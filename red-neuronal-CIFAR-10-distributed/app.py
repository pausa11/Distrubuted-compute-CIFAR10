# server.py
import os
import io
import base64
import json
import time
import random
import logging
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

from flask import Flask, request, jsonify

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# =========================
# Límites de hilos (RAM baja)
# =========================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cifar_api")

# =========================
# Flask
# =========================
app = Flask(__name__)

# =========================
# Modelo
# =========================
class SmallCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1   = nn.Linear(128 * 4 * 4, 256)
        self.drop  = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))   # 64x8x8
        x = self.pool(F.relu(self.conv3(x)))   # 128x4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

# =========================
# Dataset manager (lazy)
# =========================
class DataManager:
    """Carga CIFAR-10 sólo cuando hace falta. Workers=0 para ahorrar RAM."""
    def __init__(self, data_root: str, batch_size: int):
        self.data_root   = data_root
        self.batch_size  = batch_size
        self._loaded     = False
        self.train_ds    = None
        self.test_ds     = None

    def ensure_loaded(self):
        if self._loaded:
            return
        import torchvision
        import torchvision.transforms as T
        norm = T.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            norm
        ])
        test_tf = T.Compose([T.ToTensor(), norm])

        self.train_ds = torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, download=True, transform=train_tf
        )
        self.test_ds = torchvision.datasets.CIFAR10(
            root=self.data_root, train=False, download=True, transform=test_tf
        )
        self._loaded = True
        log.info(f"Datasets cargados: train={len(self.train_ds)} test={len(self.test_ds)}")

    def train_loader(self) -> DataLoader:
        self.ensure_loaded()
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

    def test_loader(self) -> DataLoader:
        self.ensure_loaded()
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=0, pin_memory=False)

# =========================
# Utilidades
# =========================
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, total, correct = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = crit(logits, labels)
        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return (total_loss / total) if total else 0.0, (correct / total) if total else 0.0

def state_dict_to_b64(state: Dict[str, torch.Tensor]) -> str:
    bio = io.BytesIO()
    torch.save(state, bio)
    return base64.b64encode(bio.getvalue()).decode("ascii")

def state_dict_from_b64(b64: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(b64.encode("ascii"))
    return torch.load(io.BytesIO(raw), map_location="cpu")

# =========================
# Estado de entrenamiento
# =========================
@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 64
    seed: int = 1337
    checkpoint_path: str = "./checkpoints/best.pt"
    eval_each_epoch: bool = True
    return_state_b64: bool = False  # si quieres devolver el modelo al terminar

TRAIN_LOCK = threading.Lock()
TRAIN_RUNNING = False
TRAIN_STOP = threading.Event()
TRAIN_PROGRESS: Dict[str, Any] = {
    "running": False,
    "epoch": 0,
    "epochs": 0,
    "train_loss": None,
    "train_acc": None,
    "val_loss": None,
    "val_acc": None,
    "started_at": None,
    "finished_at": None,
    "last_error": None,
}

# =========================
# Globals del servicio
# =========================
_model: Optional[SmallCIFAR] = None
_data: Optional[DataManager] = None
_init_lock = threading.Lock()

def ensure_initialized(load_data: bool = False, batch_size: int = 64):
    global _model, _data
    if _model is not None and _data is not None and (not load_data or _data._loaded):
        return
    with _init_lock:
        if _model is None:
            _model = SmallCIFAR().to(torch.device("cpu"))
            log.info("Modelo inicializado en CPU.")
        if _data is None:
            data_root = os.environ.get("DATA_ROOT", "/tmp/torch-datasets")
            _data = DataManager(data_root=data_root, batch_size=batch_size)
            log.info(f"DataManager listo (root={data_root}, batch_size={batch_size}).")
        if load_data and not _data._loaded:
            _data.ensure_loaded()

# =========================
# Bucle de entrenamiento
# =========================
def train_worker(cfg: TrainConfig):
    global TRAIN_RUNNING, TRAIN_PROGRESS
    device = torch.device("cpu")
    try:
        set_seed(cfg.seed)
        ensure_initialized(load_data=True, batch_size=cfg.batch_size)

        model = _model  # global
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        criterion = nn.CrossEntropyLoss()
        train_loader = _data.train_loader()
        val_loader = _data.test_loader() if cfg.eval_each_epoch else None

        with TRAIN_LOCK:
            TRAIN_RUNNING = True
            TRAIN_STOP.clear()
            TRAIN_PROGRESS.update({
                "running": True,
                "epoch": 0,
                "epochs": cfg.epochs,
                "started_at": time.time(),
                "finished_at": None,
                "last_error": None,
                "train_loss": None,
                "train_acc": None,
                "val_loss": None,
                "val_acc": None,
            })

        best_acc = -1.0
        os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)

        for epoch in range(1, cfg.epochs + 1):
            if TRAIN_STOP.is_set():
                log.warning("Entrenamiento cancelado por /stop.")
                break

            model.train()
            total, correct, running_loss = 0, 0, 0.0
            t0 = time.time()

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                pred = logits.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

            train_loss = (running_loss / total) if total else 0.0
            train_acc  = (correct / total) if total else 0.0

            val_loss, val_acc = (None, None)
            if val_loader is not None:
                val_loss, val_acc = evaluate(model, val_loader, device)
                # checkpoint sencillo
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save({"model_state": model.state_dict(),
                                "val_acc": val_acc,
                                "epoch": epoch}, cfg.checkpoint_path)

            with TRAIN_LOCK:
                TRAIN_PROGRESS.update({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time_s": round(time.time() - t0, 3),
                })

            log.info(f"[Epoch {epoch}/{cfg.epochs}] "
                     f"train_loss={train_loss:.4f} acc={train_acc:.4f} "
                     f"val_loss={val_loss if val_loss is not None else '-'} "
                     f"val_acc={val_acc if val_acc is not None else '-'}")

        with TRAIN_LOCK:
            TRAIN_RUNNING = False
            TRAIN_PROGRESS["running"] = False
            TRAIN_PROGRESS["finished_at"] = time.time()

        if cfg.return_state_b64:
            # nada que devolver por hilo; /status puede ofrecerlo bajo demanda
            pass

    except Exception as e:
        log.exception("Fallo en entrenamiento")
        with TRAIN_LOCK:
            TRAIN_RUNNING = False
            TRAIN_PROGRESS["running"] = False
            TRAIN_PROGRESS["finished_at"] = time.time()
            TRAIN_PROGRESS["last_error"] = str(e)

# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    # No fuerza carga de datasets
    return jsonify({"status": "ok", "training_running": TRAIN_RUNNING}), 200

@app.get("/status")
def status():
    # Devuelve progreso actual (si hay)
    with TRAIN_LOCK:
        out = dict(TRAIN_PROGRESS)
    # tiempos legibles
    if out.get("started_at"):
        out["started_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(out["started_at"]))
    if out.get("finished_at"):
        out["finished_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(out["finished_at"]))
    return jsonify(out), 200

@app.post("/train")
def train_endpoint():
    """
    Lanza entrenamiento asíncrono.

    JSON opcional:
    {
      "epochs": 10,
      "lr": 0.1,
      "momentum": 0.9,
      "weight_decay": 0.0005,
      "batch_size": 64,
      "seed": 1337,
      "checkpoint_path": "./checkpoints/best.pt",
      "eval_each_epoch": true,
      "return_state_b64": false
    }
    """
    if TRAIN_RUNNING:
        return jsonify({"message": "Ya hay entrenamiento en curso"}), 409

    body = request.get_json(silent=True) or {}
    try:
        cfg = TrainConfig(**{k: body[k] for k in body if k in TrainConfig.__annotations__})
    except TypeError as e:
        return jsonify({"error": f"Parámetros inválidos: {e}"}), 400

    # Inicializa base (modelo + datamanager, sin cargar datasets aún)
    ensure_initialized(load_data=False, batch_size=cfg.batch_size)

    t = threading.Thread(target=train_worker, args=(cfg,), daemon=True)
    t.start()
    return jsonify({"message": "Entrenamiento iniciado", "config": asdict(cfg)}), 202

@app.post("/stop")
def stop():
    """Solicita cancelar el entrenamiento en curso (se detiene al final de la época)."""
    if not TRAIN_RUNNING:
        return jsonify({"message": "No hay entrenamiento en curso"}), 400
    TRAIN_STOP.set()
    return jsonify({"message": "Cancelación solicitada"}), 200

@app.post("/evaluate")
def eval_endpoint():
    """Evalúa el modelo actual sobre test (carga datasets si no están)."""
    ensure_initialized(load_data=True, batch_size=int((request.json or {}).get("batch_size", 64)))
    loss, acc = evaluate(_model, _data.test_loader(), torch.device("cpu"))
    return jsonify({"test_loss": loss, "test_accuracy": acc, "samples": len(_data.test_ds)}), 200

@app.get("/model/b64")
def download_model_b64():
    """Devuelve el state_dict en base64 (útil para sincronizar nodos)."""
    ensure_initialized(load_data=False)
    b64 = state_dict_to_b64(_model.state_dict())
    return jsonify({"model_state_b64": b64}), 200

@app.post("/model/b64")
def upload_model_b64():
    """Carga un state_dict desde base64."""
    data = request.get_json(silent=True) or {}
    b64 = data.get("model_state_b64")
    if not b64:
        return jsonify({"error": "model_state_b64 requerido"}), 400
    ensure_initialized(load_data=False)
    _model.load_state_dict(state_dict_from_b64(b64))
    return jsonify({"message": "Modelo actualizado"}), 200

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Variables de entorno útiles:
    #   DATA_ROOT=/ruta/a/datasets  (default: /tmp/torch-datasets)
    #   FLASK_RUN_PORT=8000
    app.run(host="0.0.0.0", port=int(os.environ.get("FLASK_RUN_PORT", "8000")))
