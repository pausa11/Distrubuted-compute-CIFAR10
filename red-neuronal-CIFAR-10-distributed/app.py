import os
import io
import math
import base64
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

# Permitimos ajustar hilos vía ENV (por defecto 1 para estabilidad en VMs)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# Torch: hilos intra/inter-op (se puede sobreescribir por ENV AGENT_TORCH_THREADS)
_torch_threads = int(os.environ.get("AGENT_TORCH_THREADS", "1"))
torch.set_num_threads(max(1, _torch_threads))
try:
    torch.set_num_interop_threads(max(1, int(os.environ.get("AGENT_TORCH_INTEROP", str(_torch_threads)))))
except Exception:
    pass

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("agent")
logging.getLogger("werkzeug").setLevel(logging.WARNING)  # menos ruido HTTP

app = Flask(__name__)

_init_lock = threading.Lock()
_initialized = False

_batch_manager: Optional["BatchManager"] = None
_model: Optional[nn.Module] = None
_device = torch.device("cpu")

class SmallCIFAR(nn.Module):
    """
    CNN para CIFAR-10 con BatchNorm y Dropout.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, groups=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, groups=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.fc1   = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64x8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 128x4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class BatchManager:
    """
    Carga CIFAR-10 on-demand.
    Permite controlar batch_size/workers/prefetch/pin_memory y shuffle/drop_last como en tu script.
    """
    def __init__(self, data_root: str, default_batch_size: int = 32, default_workers: int = 0):
        self.data_root = data_root
        self.default_batch_size = default_batch_size
        self.default_workers = default_workers
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
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize
        ])
        test_tf = T.Compose([T.ToTensor(), normalize])

        self.train_ds = torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, download=True, transform=train_tf
        )
        self.test_ds = torchvision.datasets.CIFAR10(
            root=self.data_root, train=False, download=True, transform=test_tf
        )
        self._loaded = True
        logger.info(f"Datasets cargados: train={len(self.train_ds)} test={len(self.test_ds)}")

    def get_train_loader_for_slice(
        self,
        batch_indices: list,
        batch_size: Optional[int] = None,
        workers: Optional[int] = None,
        pin_memory: bool = False,
        prefetch_factor_train: Optional[int] = None,
    ) -> DataLoader:
        self._ensure_loaded()
        subset = Subset(self.train_ds, batch_indices)
        bs = int(batch_size or self.default_batch_size)
        nw = int(workers if workers is not None else self.default_workers)
        kwargs = {}
        if nw > 0:
            kwargs["prefetch_factor"] = int(prefetch_factor_train or 4)
        return DataLoader(
            subset,
            batch_size=bs,
            shuffle=True,          # como en tu script
            num_workers=nw,
            pin_memory=pin_memory, # CPU → False está bien; si usas MPS/GPU puedes activarlo
            persistent_workers=bool(nw > 0),
            drop_last=True,        # como en tu script
            **kwargs
        )

    def get_test_loader(self, batch_size: Optional[int] = None, workers: Optional[int] = None, pin_memory: bool = False) -> DataLoader:
        self._ensure_loaded()
        bs = int(batch_size or self.default_batch_size)
        if workers is None:
            workers = max(1, (self.default_workers // 2) or 0)
        nw = int(workers)
        kwargs = {}
        if nw > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(
            self.test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin_memory,
            persistent_workers=bool(nw > 0),
            **kwargs
        )

    def get_batch_indices(self, batch_id: int, total_batches: int) -> list:
        self._ensure_loaded()
        total_samples = len(self.train_ds)
        samples_per_batch = total_samples // total_batches
        start_idx = batch_id * samples_per_batch
        end_idx = (batch_id + 1) * samples_per_batch if batch_id < total_batches - 1 else total_samples
        return list(range(start_idx, end_idx))

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

def _maybe_cast_state(dtype_target: Optional[torch.dtype], sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
    serialize_fp16 = os.environ.get("MODEL_SERIALIZE_FP16", "0") == "1"
    sd_to_save = _maybe_cast_state(torch.float16 if serialize_fp16 else None, state)
    bio = io.BytesIO()
    torch.save(sd_to_save, bio)
    return base64.b64encode(bio.getvalue()).decode("ascii")

def _state_from_b64(b64: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(b64.encode("ascii"))
    bio = io.BytesIO(raw)
    sd = torch.load(bio, map_location="cpu", weights_only=True)
    cast_back_fp32 = os.environ.get("MODEL_DESERIALIZE_FP32", "1") == "1"
    return _maybe_cast_state(torch.float32 if cast_back_fp32 else None, sd)

def ensure_initialized(load_data: bool = False):
    """
    Inicializa (thread-safe). Por defecto NO carga datasets.
    """
    global _initialized, _batch_manager, _model
    if _initialized and (not load_data or (_batch_manager and _batch_manager._loaded)):
        return
    with _init_lock:
        if not _initialized:
            data_root = os.environ.get("DATA_ROOT", "/tmp/torch-datasets")
            default_bs = int(os.environ.get("BATCH_SIZE", "256"))   # por defecto 256 como tu script
            default_workers = int(os.environ.get("WORKERS", "0"))   # puedes poner WORKERS=cores
            _batch_manager = BatchManager(data_root=data_root, default_batch_size=default_bs, default_workers=default_workers)
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

def train_one_batch( model: nn.Module, loader: DataLoader, lr: float, momentum: float, weight_decay: float, local_epochs: int = 1, onecycle: bool = True, clip_grad_norm: float = 1.0, nesterov: bool = True, device: torch.device = torch.device("cpu"), ) -> Dict[str, Any]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    # ====== Métricas de recursos (solo al final) ======
    proc = psutil.Process(os.getpid())
    vm_total = psutil.virtual_memory().total or 1
    ct0 = proc.cpu_times()
    rss_max = proc.memory_info().rss
    rss0 = rss_max

    # Optimizador y scheduler como tu script
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=bool(nesterov)
    )

    steps_per_epoch = max(1, math.ceil(len(loader.dataset) / loader.batch_size))
    if onecycle:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=local_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    else:
        scheduler = None

    start_time = time.time()
    total_loss = 0.0
    correct = 0
    total = 0

    for ep in range(local_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if clip_grad_norm and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_norm))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # track pico de RSS
            try:
                rss_now = proc.memory_info().rss
                if rss_now > rss_max:
                    rss_max = rss_now
            except Exception:
                pass

    training_time = time.time() - start_time

    # CPU promedio del proceso durante la ventana
    try:
        ct1 = proc.cpu_times()
        cpu_time_delta = (ct1.user - ct0.user) + (ct1.system - ct0.system)
        ncpu = max(1, psutil.cpu_count(logical=True) or 1)
        avg_cpu_pct = 100.0 * cpu_time_delta / max(1e-6, training_time * ncpu)
    except Exception:
        avg_cpu_pct = 0.0

    try:
        peak_ram_pct = 100.0 * (rss_max / vm_total)
        rss1 = proc.memory_info().rss
    except Exception:
        peak_ram_pct = 0.0
        rss1 = rss_max

    return {
        "loss": (total_loss / total) if total else 0.0,
        "accuracy": (correct / total) if total else 0.0,
        "samples_processed": total,
        "training_time": training_time,
        "resources": {
            "avg_cpu_pct": float(avg_cpu_pct),
            "peak_ram_pct": float(peak_ram_pct),
            "peak_rss_bytes": int(rss_max),
            "rss_start_bytes": int(rss0),
            "rss_end_bytes": int(rss1),
        },
    }

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Node is running"}), 200

@app.route("/info", methods=["GET"])
def get_info():
    ensure_initialized(load_data=False)
    param_count = sum(p.numel() for p in _model.parameters())
    return jsonify({
        "device": str(_device),
        "model_params": int(param_count),
        "batch_size": int(os.environ.get("BATCH_SIZE", "256")),
        "workers": int(os.environ.get("WORKERS", "0")),
        "datasets_loaded": bool(_batch_manager and _batch_manager._loaded),
        "serialize_fp16": os.environ.get("MODEL_SERIALIZE_FP16", "0") == "1"
    }), 200

@app.route("/warmup", methods=["POST"])
def warmup():
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

        # Requeridos
        for p in ("batch_id", "total_batches"):
            if p not in data:
                return jsonify({"error": f"Missing parameter: {p}"}), 400

        batch_id = int(data["batch_id"])
        total_batches = int(data["total_batches"])
        if total_batches <= 0:
            return jsonify({"error": "total_batches must be > 0"}), 400
        if batch_id < 0 or batch_id >= total_batches:
            return jsonify({"error": "batch_id out of range"}), 400

        # Hiperparámetros (con defaults tipo script)
        lr = float(data.get("lr", 0.1))
        momentum = float(data.get("momentum", 0.9))
        weight_decay = float(data.get("weight_decay", 5e-4))
        seed = int(data.get("seed", 1337))
        include_state = bool(data.get("include_state", True))
        incoming_state_b64 = data.get("model_state_b64")

        # NUEVOS OPCIONALES (replican tu script)
        local_epochs = int(data.get("local_epochs", int(os.environ.get("LOCAL_EPOCHS", "1"))))
        onecycle = bool(data.get("onecycle", os.environ.get("ONECYCLE", "1") == "1"))
        clip_grad_norm = float(data.get("clip_grad_norm", float(os.environ.get("CLIP_GRAD_NORM", "1.0"))))
        nesterov = bool(data.get("nesterov", os.environ.get("NESTEROV", "1") == "1"))
        req_batch_size = data.get("batch_size")  # permite sobreescribir BS por request
        req_workers = data.get("workers")        # idem workers

        set_seed(seed)

        # Cargar estado si viene
        if incoming_state_b64:
            _model.load_state_dict(_state_from_b64(incoming_state_b64))

        # Slice del dataset
        batch_indices = _batch_manager.get_batch_indices(batch_id, total_batches)
        if len(batch_indices) == 0:
            return jsonify({"error": "empty batch slice (check total_batches)"}), 400

        # DataLoader "estilo script": shuffle=True, drop_last=True, workers/prefetch
        # En CPU: pin_memory=False (puedes activar si usas MPS/GPU)
        train_loader = _batch_manager.get_train_loader_for_slice(
            batch_indices,
            batch_size=int(req_batch_size) if req_batch_size is not None else None,
            workers=int(req_workers) if req_workers is not None else None,
            pin_memory=False,
            prefetch_factor_train=4
        )

        logger.info(f"Entrenando batch {batch_id}/{total_batches} (size={len(batch_indices)}) ...")
        results = train_one_batch(
            model=_model,
            loader=train_loader,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            local_epochs=local_epochs,
            onecycle=onecycle,
            clip_grad_norm=clip_grad_norm,
            nesterov=nesterov,
            device=_device,
        )

        response = {
            "batch_id": batch_id,
            "total_batches": total_batches,
            "batch_size": train_loader.batch_size,
            "slice_size": len(batch_indices),
            "loss": float(results["loss"]),
            "accuracy": float(results["accuracy"]),
            "samples_processed": int(results["samples_processed"]),
            "training_time": float(results["training_time"]),
            "resources": results["resources"],
            "status": "completed"
        }

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

        test_loader = _batch_manager.get_test_loader()
        with torch.inference_mode():
            test_loss, test_acc = evaluate_batch(_model, test_loader, _device)

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
    Info estática del nodo (sin CPU/RAM en tiempo real).
    """
    try:
        payload = {
            "ts": time.time(),
            "datasets_loaded": bool(_batch_manager and _batch_manager._loaded),
            "batch_size": int(os.environ.get("BATCH_SIZE", "256")),
            "workers": int(os.environ.get("WORKERS", "0")),
        }
        return jsonify(payload), 200
    except Exception as e:
        logger.exception("Error en /metrics")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "6000"))
    logger.info(f"Iniciando agent en 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
