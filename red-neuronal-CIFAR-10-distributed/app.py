# agent.py
import os
import io
import base64
import json
import time
import random
import logging
import threading
from typing import Dict, Any, Tuple, Optional

from flask import Flask, request, jsonify

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import psutil

# ==========================
# Config global (RAM/CPU)
# ==========================
# Limitar hilos BLAS/OpenMP (útil en VMs pequeñas)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# Torch: 1 hilo intra/inter-op
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("agent")

# Flask
app = Flask(__name__)

# ==========================
# Globals controlados
# ==========================
_init_lock = threading.Lock()
_initialized = False

_batch_manager: Optional["BatchManager"] = None
_model: Optional[nn.Module] = None
_device = torch.device("cpu")  # CPU por defecto; simple y portable

# ==========================
# Modelo
# ==========================
class SmallCIFAR(nn.Module):
    """
    CNN pequeña para CIFAR-10 (sin BatchNorm para simplificar sincronización).
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # -> (32,16,16)
        x = self.pool(F.relu(self.conv2(x)))  # -> (64,8,8)
        x = self.pool(F.relu(self.conv3(x)))  # -> (128,4,4)
        x = x.view(x.size(0), -1)            # -> (N, 128*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================
# Dataset / BatchManager (lazy)
# ==========================
class BatchManager:
    """
    Carga CIFAR-10 on-demand.
    Mantener num_workers=0 y batch_size moderado para ahorrar RAM.
    """
    def __init__(self, data_root: str, batch_size: int = 32):
        self.data_root = data_root
        self.batch_size = batch_size
        self._loaded = False
        self.train_ds = None
        self.test_ds = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        import torchvision
        import torchvision.transforms as T

        normalize = T.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        test_tf = T.Compose([T.ToTensor(), normalize])

        # Nota: descargar solo la primera vez; torchvision guarda en root
        self.train_ds = torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, download=True, transform=train_tf
        )
        self.test_ds = torchvision.datasets.CIFAR10(
            root=self.data_root, train=False, download=True, transform=test_tf
        )
        self._loaded = True
        logger.info(f"Datasets cargados: train={len(self.train_ds)} test={len(self.test_ds)}")

    def get_batch_loader(self, batch_indices: list) -> DataLoader:
        self._ensure_loaded()
        subset = Subset(self.train_ds, batch_indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def get_test_loader(self) -> DataLoader:
        self._ensure_loaded()
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def get_batch_indices(self, batch_id: int, total_batches: int) -> list:
        self._ensure_loaded()
        total_samples = len(self.train_ds)
        # Partición equitativa, último shard se queda con el resto
        samples_per_batch = total_samples // total_batches
        start_idx = batch_id * samples_per_batch
        end_idx = (batch_id + 1) * samples_per_batch if batch_id < total_batches - 1 else total_samples
        return list(range(start_idx, end_idx))

# ==========================
# Utils
# ==========================
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDNN no aplica en CPU, pero no estorba si existe
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def _maybe_cast_state(dtype_target: Optional[torch.dtype], sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Casteo opcional de tensores (por ejemplo, fp32<->fp16) para ahorrar ancho de banda.
    """
    if dtype_target is None:
        return sd
    out = {}
    for k, v in sd.items():
        if torch.is_floating_point(v):
            out[k] = v.to(dtype=dtype_target)
        else:
            out[k] = v
    return out

def _state_to_b64(state: Dict[str, torch.Tensor]) -> str:
    # Compresión opcional a fp16 para bajar payload
    serialize_fp16 = os.environ.get("MODEL_SERIALIZE_FP16", "0") == "1"
    sd_to_save = _maybe_cast_state(torch.float16 if serialize_fp16 else None, state)

    bio = io.BytesIO()
    torch.save(sd_to_save, bio)
    return base64.b64encode(bio.getvalue()).decode("ascii")

def _state_from_b64(b64: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(b64.encode("ascii"))
    bio = io.BytesIO(raw)
    # weights_only=True evita cargas peligrosas por pickle
    sd = torch.load(bio, map_location="cpu", weights_only=True)
    # Si venía en fp16 y quieres operar en fp32, castea aquí:
    cast_back_fp32 = os.environ.get("MODEL_DESERIALIZE_FP32", "1") == "1"
    return _maybe_cast_state(torch.float32 if cast_back_fp32 else None, sd)

def ensure_initialized(load_data: bool = False):
    """
    Inicializa (thread-safe). Por defecto NO carga datasets.
    Endpoints que entrenan/evalúan pasan load_data=True.
    """
    global _initialized, _batch_manager, _model
    if _initialized and (not load_data or (_batch_manager and _batch_manager._loaded)):
        return

    with _init_lock:
        if not _initialized:
            data_root = os.environ.get("DATA_ROOT", "/tmp/torch-datasets")
            batch_size = int(os.environ.get("BATCH_SIZE", "32"))
            _batch_manager = BatchManager(data_root=data_root, batch_size=batch_size)
            _model = SmallCIFAR().to(_device)
            _initialized = True
            logger.info("Componentes base listos (modelo en CPU, dataset aún no cargado).")
        if load_data and not _batch_manager._loaded:
            _batch_manager._ensure_loaded()

@torch.no_grad()
def evaluate_batch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return (total_loss / total) if total else 0.0, (correct / total) if total else 0.0

def train_one_batch(
    model: nn.Module,
    batch_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, Any]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    for images, labels in batch_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    training_time = time.time() - start_time
    return {
        "loss": (total_loss / total) if total else 0.0,
        "accuracy": (correct / total) if total else 0.0,
        "samples_processed": total,
        "training_time": training_time,
    }

# ==========================
# Endpoints
# ==========================
@app.route("/health", methods=["GET"])
def health_check():
    # Ligero: no carga dataset
    return jsonify({"status": "healthy", "message": "Node is running"}), 200

@app.route("/info", methods=["GET"])
def get_info():
    ensure_initialized(load_data=False)
    param_count = sum(p.numel() for p in _model.parameters())
    return jsonify({
        "device": str(_device),
        "model_params": int(param_count),
        "batch_size": int(os.environ.get("BATCH_SIZE", "32")),
        "datasets_loaded": bool(_batch_manager and _batch_manager._loaded),
        "serialize_fp16": os.environ.get("MODEL_SERIALIZE_FP16", "0") == "1"
    }), 200

@app.route("/warmup", methods=["POST"])
def warmup():
    """
    Precarga datasets en RAM/FS para evitar el costo de la primera llamada a /train_batch o /evaluate.
    """
    try:
        ensure_initialized(load_data=True)
        return jsonify({"ok": True, "datasets_loaded": True}), 200
    except Exception as e:
        logger.exception("Error en warmup")
        return jsonify({"error": str(e)}), 500

@app.route("/train_batch", methods=["POST"])
def train_batch_endpoint():
    try:
        data = request.get_json(force=True) or {}
        ensure_initialized(load_data=True)

        # Validaciones requeridas
        for p in ("batch_id", "total_batches"):
            if p not in data:
                return jsonify({"error": f"Missing parameter: {p}"}), 400

        batch_id = int(data["batch_id"])
        total_batches = int(data["total_batches"])
        if total_batches <= 0:
            return jsonify({"error": "total_batches must be > 0"}), 400
        if batch_id < 0 or batch_id >= total_batches:
            return jsonify({"error": "batch_id out of range"}), 400

        # Hiperparámetros
        lr = float(data.get("lr", 0.1))
        momentum = float(data.get("momentum", 0.9))
        weight_decay = float(data.get("weight_decay", 5e-4))
        seed = int(data.get("seed", 1337))
        include_state = bool(data.get("include_state", True))
        incoming_state_b64 = data.get("model_state_b64")

        set_seed(seed)

        # Cargar estado si viene
        if incoming_state_b64:
            _model.load_state_dict(_state_from_b64(incoming_state_b64))

        # Batch loader
        batch_indices = _batch_manager.get_batch_indices(batch_id, total_batches)
        if len(batch_indices) == 0:
            return jsonify({"error": "empty batch slice (check total_batches)"}), 400
        batch_loader = _batch_manager.get_batch_loader(batch_indices)

        # Optimizer por-request (stateless en el agente)
        optimizer = optim.SGD(_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        logger.info(f"Entrenando batch {batch_id}/{total_batches} (size={len(batch_indices)}) ...")
        results = train_one_batch(_model, batch_loader, optimizer, _device)

        response = {
            "batch_id": batch_id,
            "total_batches": total_batches,
            "batch_size": _batch_manager.batch_size,
            "slice_size": len(batch_indices),
            "loss": float(results["loss"]),
            "accuracy": float(results["accuracy"]),
            "samples_processed": int(results["samples_processed"]),
            "training_time": float(results["training_time"]),
            "status": "completed"
        }

        # Estado del modelo opcional en base64 binario
        if include_state:
            response["model_state_b64"] = _state_to_b64(_model.state_dict())

        logger.info(
            f"Batch {batch_id} OK - Loss: {results['loss']:.4f}, Acc: {results['accuracy']:.4f}, "
            f"samples: {results['samples_processed']}"
        )
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Error en /train_batch")
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate", methods=["POST"])
def evaluate_endpoint():
    try:
        data = request.get_json(silent=True) or {}
        ensure_initialized(load_data=True)

        incoming_state_b64 = data.get("model_state_b64")
        if incoming_state_b64:
            _model.load_state_dict(_state_from_b64(incoming_state_b64))

        loader = _batch_manager.get_test_loader()
        with torch.inference_mode():
            test_loss, test_acc = evaluate_batch(_model, loader, _device)

        return jsonify({
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "samples_evaluated": len(_batch_manager.test_ds)
        }), 200

    except Exception as e:
        logger.exception("Error en /evaluate")
        return jsonify({"error": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def metrics_endpoint():
    """
    Devuelve stats del nodo (CPU/RAM) y, opcionalmente, últimas info de dataset/modelo.
    No dispara entrenamiento.
    """
    try:
        cpu = psutil.cpu_percent(interval=0.05)  # medición rápida
        ram = psutil.virtual_memory().percent
        payload = {
            "cpu": float(cpu),
            "ram": float(ram),
            "ts": time.time(),
            "datasets_loaded": bool(_batch_manager and _batch_manager._loaded),
            "batch_size": int(os.environ.get("BATCH_SIZE", "32")),
        }
        return jsonify(payload), 200
    except Exception as e:
        logger.exception("Error en /metrics")
        return jsonify({"error": str(e)}), 500
# ==========================
# Main
# ==========================
if __name__ == "__main__":
    # Puerto configurable vía env PORT (default 6000)
    port = int(os.environ.get("PORT", "6000"))
    logger.info(f"Iniciando agent en 0.0.0.0:{port}")
    # threaded=True permite manejar concurrencia simple en endpoints
    app.run(host="0.0.0.0", port=port, threaded=True)
