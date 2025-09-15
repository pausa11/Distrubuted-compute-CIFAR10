import os
import io
import json
import time
import uuid
import base64
import logging
import threading
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("controller")

NODE_HOSTS = {
    "node-0": "http://34.132.166.106:6000",
    "node-1": "http://34.121.35.52:6000",
    "node-2": "http://34.9.209.76:6000",
    "node-3": "http://34.53.3.174:6000",
    "node-4": "http://34.11.238.13:6000",
}

HTTP_TIMEOUT_TRAIN = float(os.environ.get("HTTP_TIMEOUT_TRAIN", "400"))
HTTP_TIMEOUT_PING  = float(os.environ.get("HTTP_TIMEOUT_PING", "2"))
HTTP_TIMEOUT_METR  = float(os.environ.get("HTTP_TIMEOUT_METR", "1.5"))
SSE_HEARTBEAT_EVERY = 10
METRICS_POLL_INTERVAL = float(os.environ.get("METRICS_POLL_INTERVAL", "1.0"))

# Telemetría en vivo (desactivada por defecto)
ENABLE_REALTIME_POLL = os.environ.get("ENABLE_REALTIME_POLL", "0") == "1"

# Defaults de hiperparámetros "estilo script bueno"
DEFAULT_LOCAL_EPOCHS    = int(os.environ.get("DEFAULT_LOCAL_EPOCHS", "1"))
DEFAULT_BATCH_SIZE      = int(os.environ.get("DEFAULT_BATCH_SIZE", "256"))
DEFAULT_WORKERS         = int(os.environ.get("DEFAULT_WORKERS", "0"))
DEFAULT_ONECYCLE        = os.environ.get("DEFAULT_ONECYCLE", "1") == "1"
DEFAULT_CLIP_GRAD_NORM  = float(os.environ.get("DEFAULT_CLIP_GRAD_NORM", "1.0"))
DEFAULT_NESTEROV        = os.environ.get("DEFAULT_NESTEROV", "1") == "1"

# Checkpoints
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "./checkpoints")
BEST_CKPT_NAME  = os.environ.get("BEST_CKPT_NAME", "best.pt")

# Evaluación
EVAL_ON_TEST = os.environ.get("EVAL_ON_TEST", "1") == "1"
TEST_EVAL_BATCH_SIZE = int(os.environ.get("TEST_EVAL_BATCH_SIZE", "256"))

# Estado global en memoria
SESSIONS: Dict[str, Dict[str, Any]] = {}   # session_id -> info
METRICS:  Dict[str, Dict[str, Dict[str, Any]]] = {}  # session_id -> { node_id: last_metrics }

# Clases de CIFAR-10
CIFAR10_CLASSES = [ "airplane","automobile","bird","cat","deer", "dog","frog","horse","ship","truck" ]

# Mean/STD estándar de CIFAR-10 (asegúrate que coincidan con tu training)
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2470, 0.2435, 0.2616)

# Recursos cacheados para inferencia/evaluación
_infer_model: Optional[nn.Module] = None
_test_ds = None
_test_tf = None
_denorm_tf = None
_last_best_mtime: Optional[float] = None

# Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

#   Modelo (igual a nodos)
class SmallCIFAR(nn.Module):
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

  def forward(self, x):
      x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x16x16
      x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64x8x8
      x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 128x4x4
      x = x.view(x.size(0), -1)
      x = F.relu(self.fc1(x))
      x = self.dropout(x)
      x = self.fc2(x)
      return x

# Utils
def state_to_b64(sd: Dict[str, torch.Tensor]) -> str:
    bio = io.BytesIO()
    torch.save(sd, bio)
    return base64.b64encode(bio.getvalue()).decode("ascii")

def b64_to_state(b64s: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(b64s.encode("ascii"))
    bio = io.BytesIO(raw)
    # Nota: weights_only=True requiere PyTorch >= 2.2; en <2.2 quitar el flag.
    return torch.load(bio, map_location="cpu", weights_only=True)

def fedavg_state_dicts(sd_list: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    """Promedia state_dicts con pesos (número de muestras procesadas por nodo)."""
    assert len(sd_list) == len(weights) and len(sd_list) > 0
    total = float(sum(max(1, int(w)) for w in weights))
    out = {}
    keys = sd_list[0].keys()
    with torch.no_grad():
        for k in keys:
            acc = None
            for sd, w in zip(sd_list, weights):
                t = sd[k].float() * (max(1, int(w)) / total)
                acc = t.clone() if acc is None else acc.add_(t)
            out[k] = acc.to(sd_list[0][k].dtype)
    return out

def _resolve_best_ckpt_path() -> str:
    """Devuelve ruta existente a best.pt si existe; si no, cadena vacía."""
    abs_path = "/checkpoints/best.pt"
    if os.path.isfile(abs_path):
        return abs_path
    rel_path = os.path.join(CHECKPOINTS_DIR, BEST_CKPT_NAME)
    if os.path.isfile(rel_path):
        return rel_path
    return ""

def _best_save_path() -> str:
    """Ruta donde GUARDAR best.pt. Prefiere /checkpoints si existe, si no CHECKPOINTS_DIR."""
    if os.path.isdir("/checkpoints"):
        os.makedirs("/checkpoints", exist_ok=True)
        return "/checkpoints/best.pt"
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    return os.path.join(CHECKPOINTS_DIR, BEST_CKPT_NAME)

def _best_path_and_mtime() -> Tuple[str, float]:
    ckpt = _resolve_best_ckpt_path()
    if not ckpt or not os.path.isfile(ckpt):
        raise FileNotFoundError("best.pt not found (revisa CHECKPOINTS_DIR / BEST_CKPT_NAME)")
    return ckpt, os.path.getmtime(ckpt)

def _load_disk_best_meta() -> Tuple[float, int]:
    """Lee val_acc y epoch del best.pt en disco. Si no existe, (-inf, -1)."""
    p = _resolve_best_ckpt_path()
    if not p:
        return float("-inf"), -1
    try:
        obj = torch.load(p, map_location="cpu")
        val_acc = float(obj.get("val_acc", float("-inf")))
        epoch   = int(obj.get("epoch", -1))
        return val_acc, epoch
    except Exception as e:
        logger.warning(f"No se pudo leer meta de best.pt existente: {e}")
        return float("-inf"), -1

def _atomic_save_best(state_dict: Dict[str, torch.Tensor], epoch: int, val_acc: float, session_id: str):
    """Guarda best.pt de forma atómica."""
    target = _best_save_path()
    tmp = target + ".tmp"
    payload = {
        "model_state": state_dict,
        "epoch": int(epoch),
        "val_acc": float(val_acc),    # en porcentaje
        "updated_at": time.time(),
        "session_id": session_id
    }
    torch.save(payload, tmp)
    os.replace(tmp, target)
    logger.info(f"[BEST] Guardado nuevo best.pt en '{target}' (epoch={epoch}, val_acc={val_acc:.2f}%)")

# Inferencia / evaluación
def _ensure_infer_resources():
    """Carga (o recarga) el modelo best.pt y el dataset de test con sus transforms."""
    global _infer_model, _test_ds, _test_tf, _denorm_tf, _last_best_mtime

    # Dataset + transforms
    if _test_ds is None:
        _test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        # Denormalización para visualizar
        _denorm_tf = transforms.Normalize(
            mean=tuple(-m/s for m, s in zip(_CIFAR_MEAN, _CIFAR_STD)),
            std=tuple(1/s for s in _CIFAR_STD)
        )
        _test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=_test_tf)

    # Modelo
    ckpt, mtime = _best_path_and_mtime()
    if (_infer_model is None) or (_last_best_mtime != mtime):
        logger.info(f"Cargando modelo de {ckpt} ...")
        _last_best_mtime = mtime
        obj = torch.load(ckpt, map_location="cpu")
        if "model_state" not in obj:
            raise RuntimeError(f"Checkpoint sin 'model_state': keys={list(obj.keys())}")

        net = SmallCIFAR(num_classes=10).cpu()
        missing, unexpected = net.load_state_dict(obj["model_state"], strict=False)
        if missing:
            logger.warning(f"Faltan llaves al cargar state_dict: {missing}")
        if unexpected:
            logger.warning(f"Llaves inesperadas al cargar state_dict: {unexpected}")
        net.eval()
        _infer_model = net

def _evaluate_state_dict_on_test(state_dict: Dict[str, torch.Tensor]) -> float:
    """
    Evalúa accuracy (%) en CIFAR-10 test con CPU.
    """
    # Asegura dataset
    if _test_ds is None:
        _ensure_infer_resources()
    # Construye red y carga pesos
    model = SmallCIFAR(num_classes=10).cpu()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.debug(f"[EVAL] faltan llaves: {missing}")
    if unexpected:
        logger.debug(f"[EVAL] llaves inesperadas: {unexpected}")
    model.eval()

    loader = DataLoader(_test_ds, batch_size=TEST_EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.numel()
    acc = 100.0 * correct / max(1, total)
    return acc

def _tensor_to_png_datauri(x_norm: torch.Tensor) -> str:
    """
    x_norm: tensor [3,32,32] normalizado. Devuelve PNG base64 data URI.
    """
    with torch.no_grad():
        x = _denorm_tf(x_norm.cpu())
        x = torch.clamp(x, 0.0, 1.0)
        img = (x.permute(1,2,0).numpy() * 255).astype("uint8")
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Acepta PIL.Image; la convierte a RGB, redimensiona a 32x32 y normaliza como CIFAR-10.
    """
    img = img.convert("RGB").resize((32, 32), Image.BILINEAR)
    t = transforms.ToTensor()(img)
    t = transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)(t)
    return t

def post_json(url: str, payload: Dict[str, Any], timeout: float):
    return requests.post(url, json=payload, timeout=timeout)

# Métricas polling opcional
def _poll_metrics_loop(session_id: str):
    """Hilo que sondea /metrics en cada nodo (sin CPU/RAM en vivo en el agent)."""
    logger.info(f"[{session_id}] Iniciando poll de /metrics")
    while SESSIONS.get(session_id, {}).get("running", False):
        info = SESSIONS.get(session_id) or {}
        nodes = info.get("nodes", [])
        for node_id in nodes:
            base = NODE_HOSTS.get(node_id)
            if not base:
                continue
            try:
                r = requests.get(f"{base}/metrics", timeout=HTTP_TIMEOUT_METR)
                if r.ok:
                    data = r.json()
                    METRICS.setdefault(session_id, {}).setdefault(node_id, {})
                    METRICS[session_id][node_id].update({
                        "ts": float(data.get("ts", time.time())),
                        "datasets_loaded": bool(data.get("datasets_loaded", False)),
                        "batch_size_agent": int(data.get("batch_size", 0)),
                        "workers_agent": int(data.get("workers", 0)),
                        "node_id": node_id
                    })
                else:
                    logger.debug(f"[{session_id}] /metrics {node_id} HTTP {r.status_code}")
            except requests.exceptions.RequestException:
                pass
        time.sleep(METRICS_POLL_INTERVAL)
    logger.info(f"[{session_id}] Fin de poll de /metrics")

#  Loop de entrenamiento (FedAvg)
def _train_fedavg_loop(session_id: str, nodes: List[str], epochs: int, init_b64: str,
                       lr: float, momentum: float, weight_decay: float, seed: int,
                       shards_per_epoch: int, hp_defaults: Dict[str, Any]):
    try:
        logger.info(f"[{session_id}] Iniciando FedAvg: epochs={epochs}, shards_per_epoch={shards_per_epoch}, nodes={nodes}")
        current_b64 = init_b64
        total_batches = max(1, len(nodes) * max(1, shards_per_epoch))

        # Mejor en disco (si existe)
        best_acc_disk, best_epoch_disk = _load_disk_best_meta()
        best_acc = best_acc_disk
        best_epoch = best_epoch_disk
        logger.info(f"[{session_id}] best.pt en disco: epoch={best_epoch} val_acc={best_acc if best_acc!=float('-inf') else 'N/A'}")

        for epoch in range(1, epochs + 1):
            if not SESSIONS.get(session_id, {}).get("running", False):
                break

            subrounds = (total_batches + len(nodes) - 1) // len(nodes)
            epoch_loss_accumulator, epoch_acc_accumulator, epoch_weight_accumulator = [], [], []

            for sub in range(subrounds):
                if not SESSIONS.get(session_id, {}).get("running", False):
                    break

                futures = []
                with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
                    for idx, node_id in enumerate(nodes):
                        batch_id = sub * len(nodes) + idx
                        if batch_id >= total_batches:
                            continue
                        base = NODE_HOSTS.get(node_id)
                        if not base:
                            continue

                        payload = {
                            "batch_id": batch_id,
                            "total_batches": total_batches,
                            "lr": lr,
                            "momentum": momentum,
                            "weight_decay": weight_decay,
                            "seed": seed + epoch + sub,
                            "model_state_b64": current_b64,
                            "include_state": True,

                            # === Hiperparámetros añadidos ===
                            "local_epochs": int(hp_defaults["local_epochs"]),
                            "onecycle": bool(hp_defaults["onecycle"]),
                            "clip_grad_norm": float(hp_defaults["clip_grad_norm"]),
                            "nesterov": bool(hp_defaults["nesterov"]),
                            "batch_size": int(hp_defaults["batch_size"]) if hp_defaults["batch_size"] > 0 else None,
                            "workers": int(hp_defaults["workers"]) if hp_defaults["workers"] >= 0 else None,
                        }

                        futures.append((node_id, batch_id, ex.submit(post_json, f"{base}/train_batch", payload, HTTP_TIMEOUT_TRAIN)))

                    sd_list, weights, pernode = [], [], []
                    for node_id, batch_id, fut in futures:
                        try:
                            r = fut.result()
                            if not r.ok:
                                raise RuntimeError(f"HTTP {r.status_code}")
                            resp = r.json()

                            # Registrar métricas + recursos del nodo
                            METRICS.setdefault(session_id, {}).setdefault(node_id, {})
                            res = resp.get("resources", {}) or {}
                            METRICS[session_id][node_id].update({
                                "epoch": epoch,
                                "loss": float(resp["loss"]),
                                "acc": float(resp["accuracy"]) * 100.0,
                                "samples": int(resp["samples_processed"]),
                                "batch_id": int(resp["batch_id"]),
                                "total_batches": int(resp["total_batches"]),
                                "avg_cpu_pct": float(res.get("avg_cpu_pct", 0.0)),
                                "peak_ram_pct": float(res.get("peak_ram_pct", 0.0)),
                                "peak_rss_bytes": int(res.get("peak_rss_bytes", 0)),
                                "ts": time.time(),
                                "node_id": node_id,

                                # HP usados en el agent
                                "local_epochs": hp_defaults["local_epochs"],
                                "onecycle": hp_defaults["onecycle"],
                                "clip_grad_norm": hp_defaults["clip_grad_norm"],
                                "nesterov": hp_defaults["nesterov"],
                                "batch_size": resp.get("batch_size", hp_defaults["batch_size"]),
                            })

                            sd = b64_to_state(resp["model_state_b64"])
                            sd_list.append(sd)
                            w = max(1, int(resp["samples_processed"]))
                            weights.append(w)
                            pernode.append(resp)
                        except Exception as e:
                            logger.error(f"[{session_id}] Nodo {node_id} falló en sub-ronda {sub+1}/{subrounds}: {e}")

                    if not sd_list:
                        SESSIONS[session_id]["status"] = "failed"
                        SESSIONS[session_id]["running"] = False
                        logger.error(f"[{session_id}] No se recibieron estados en la sub-ronda {sub+1}. Abortando.")
                        return

                    new_sd = fedavg_state_dicts(sd_list, weights)
                    current_b64 = state_to_b64(new_sd)

                    wsum = float(sum(weights))
                    sub_loss = sum(w * float(m["loss"]) for w, m in zip(weights, pernode)) / wsum
                    # m["accuracy"] viene en [0..1] desde los agentes, aquí lo pasamos a %
                    sub_acc  = 100.0 * sum(w * float(m["accuracy"]) for w, m in zip(weights, pernode)) / wsum
                    epoch_loss_accumulator.append((sub_loss, wsum))
                    epoch_acc_accumulator.append((sub_acc,  wsum))
                    epoch_weight_accumulator.append(wsum)

                    METRICS[session_id]["all"] = {
                        "epoch": epoch,
                        "loss": float(sub_loss),
                        "acc": float(sub_acc),
                        "ts": time.time(),
                        "node_id": "all"
                    }

            # Métricas agregadas por epoch (promedio ponderado)
            if epoch_loss_accumulator:
                total_w = sum(epoch_weight_accumulator)
                g_loss = sum(l * w for (l, w) in epoch_loss_accumulator) / total_w
                g_acc  = sum(a * w for (a, w) in epoch_acc_accumulator) / total_w
            else:
                g_loss, g_acc = 0.0, 0.0

            METRICS[session_id]["all"] = {
                "epoch": epoch,
                "loss": float(g_loss),
                "acc": float(g_acc),
                "ts": time.time(),
                "node_id": "all"
            }
            logger.info(f"[{session_id}] Epoch {epoch}/{epochs}: loss={g_loss:.4f} acc(train)={g_acc:.2f}%")

            #   GUARDAR BEST.PT
            # 1) Reconstruimos state_dict actual para evaluar / guardar
            current_sd = b64_to_state(current_b64)

            # 2) val_acc: test o proxy de train
            if EVAL_ON_TEST:
                try:
                    val_acc = _evaluate_state_dict_on_test(current_sd)
                    logger.info(f"[{session_id}] Epoch {epoch}: val_acc(test)={val_acc:.2f}%")
                except Exception as e:
                    logger.exception(f"[{session_id}] Error evaluando en test; usando acc(train) como proxy.")
                    val_acc = float(g_acc)
            else:
                val_acc = float(g_acc)

            # 3) Si mejora, guardamos best.pt
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                _atomic_save_best(current_sd, epoch, val_acc, session_id)

        SESSIONS[session_id]["status"] = "completed"
        SESSIONS[session_id]["running"] = False
    except Exception as e:
        logger.exception(f"[{session_id}] Error en loop FedAvg")
        SESSIONS[session_id]["status"] = "failed"
        SESSIONS[session_id]["running"] = False

#  Endpoints
@app.route("/train", methods=["POST"])
def start_train():
    """
    Inicia una sesión de entrenamiento orquestado (FedAvg).
    """
    try:
        p = request.get_json(force=True) or {}
        nodes = list(p.get("nodes", []))
        if not nodes:
            return jsonify({"error": "No nodes specified"}), 400
        for node_id in nodes:
            if node_id not in NODE_HOSTS:
                return jsonify({"error": f"Unknown node: {node_id}"}), 400

        epochs = int(p.get("epochs", 25))
        lr = float(p.get("lr", 0.1))
        momentum = float(p.get("momentum", 0.9))
        weight_decay = float(p.get("weight_decay", 5e-4))
        seed = int(p.get("seed", 1337))
        shards_per_epoch = int(p.get("shards_per_epoch", 1))

        # Overrides opcionales de HP
        hp_defaults = {
            "local_epochs": int(p.get("local_epochs", DEFAULT_LOCAL_EPOCHS)),
            "batch_size": int(p.get("batch_size", DEFAULT_BATCH_SIZE)),
            "workers": int(p.get("workers", DEFAULT_WORKERS)),
            "onecycle": bool(p.get("onecycle", DEFAULT_ONECYCLE)),
            "clip_grad_norm": float(p.get("clip_grad_norm", DEFAULT_CLIP_GRAD_NORM)),
            "nesterov": bool(p.get("nesterov", DEFAULT_NESTEROV)),
        }

        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {
            "nodes": nodes,
            "running": True,
            "status": "running",
            "start_time": time.time(),
            "epochs": epochs,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "seed": seed,
            "shards_per_epoch": shards_per_epoch,
            "hp": hp_defaults,  # guardamos para consulta
        }
        METRICS[session_id] = {}

        # Modelo global inicial
        model = SmallCIFAR().cpu()
        init_b64 = state_to_b64(model.state_dict())

        # Lanzar hilos
        t_train = threading.Thread(
            target=_train_fedavg_loop,
            args=(session_id, nodes, epochs, init_b64, lr, momentum, weight_decay, seed, shards_per_epoch, hp_defaults),
            daemon=True
        )
        t_train.start()

        if ENABLE_REALTIME_POLL:
            t_poll = threading.Thread(target=_poll_metrics_loop, args=(session_id,), daemon=True)
            t_poll.start()
            logger.info(f"[{session_id}] Realtime /metrics polling ACTIVADO (ENABLE_REALTIME_POLL=1)")
        else:
            logger.info(f"[{session_id}] Realtime /metrics polling DESACTIVADO (ENABLE_REALTIME_POLL=0)")

        logger.info(f"Iniciada sesión {session_id} con nodos: {nodes} | HP: {hp_defaults}")
        return jsonify({"session_id": session_id, "status": "started", "hp": hp_defaults})
    except Exception as e:
        logger.exception("Error en /train")
        return jsonify({"error": str(e)}), 500

@app.route("/events/<session_id>")
def sse(session_id: str):
    """
    Server-Sent Events con métricas (por nodo y global "all").
    - event: connected
    - event: metrics (data: {...})
    - event: heartbeat
    - event: completed / stopped / disconnected
    """
    def stream():
        last_sent_ts: Dict[str, float] = {}  # node_id -> ts
        heartbeat_count = 0

        if session_id not in SESSIONS:
            yield f"event: error\ndata: {json.dumps({'error':'Session not found'})}\n\n"
            return

        yield "event: connected\ndata: {}\n\n"

        try:
            while SESSIONS.get(session_id, {}).get("running", False):
                session_metrics = METRICS.get(session_id, {})

                for node_id, m in list(session_metrics.items()):
                    ts = float(m.get("ts", 0.0))
                    if last_sent_ts.get(node_id) != ts:
                        last_sent_ts[node_id] = ts
                        payload = {**m, "node_id": node_id, "session_id": session_id}
                        yield f"event: metrics\ndata: {json.dumps(payload)}\n\n"

                heartbeat_count += 1
                if heartbeat_count >= SSE_HEARTBEAT_EVERY:
                    yield "event: heartbeat\ndata: {}\n\n"
                    heartbeat_count = 0

                time.sleep(0.5)

        except GeneratorExit:
            logger.info(f"SSE client disconnected from {session_id}")
        except Exception as e:
            logger.error(f"Error in SSE for {session_id}: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        status = SESSIONS.get(session_id, {}).get("status", "stopped")
        if status == "completed":
            yield f"event: completed\ndata: {json.dumps({'session_id': session_id})}\n\n"
        elif status == "failed":
            yield f"event: error\ndata: {json.dumps({'session_id': session_id, 'error':'failed'})}\n\n"
        else:
            yield f"event: stopped\ndata: {json.dumps({'session_id': session_id})}\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
            "Connection": "keep-alive",
        },
    )

@app.route("/stop/<session_id>", methods=["POST"])
def stop(session_id: str):
    """Detiene una sesión en curso."""
    try:
        sess = SESSIONS.get(session_id)
        if not sess:
            return jsonify({"error": "Session not found"}), 404
        sess["running"] = False
        sess["status"]  = "stopped"
        logger.info(f"Sesión {session_id} detenida por el usuario.")
        return jsonify({"ok": True, "session_id": session_id})
    except Exception as e:
        logger.exception("Error en /stop")
        return jsonify({"error": str(e)}), 500

@app.route("/sessions", methods=["GET"])
def list_sessions():
    return jsonify({
        "sessions": {
            sid: {
                "nodes": info.get("nodes", []),
                "status": info.get("status"),
                "running": info.get("running"),
                "start_time": info.get("start_time"),
                "epochs": info.get("epochs"),
                "lr": info.get("lr"),
                "momentum": info.get("momentum"),
                "weight_decay": info.get("weight_decay"),
                "seed": info.get("seed"),
                "shards_per_epoch": info.get("shards_per_epoch"),
                "hp": info.get("hp", {}),
            }
            for sid, info in SESSIONS.items()
        }
    })

@app.route("/session/<session_id>", methods=["GET"])
def get_session(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({
        "session_id": session_id,
        "info": sess,
        "current_metrics": METRICS.get(session_id, {})
    })

@app.route("/nodes", methods=["GET"])
def list_nodes():
    """Verifica estado de los nodos vía /health."""
    node_status = {}
    for node_id, base_url in NODE_HOSTS.items():
        try:
            r = requests.get(f"{base_url}/health", timeout=HTTP_TIMEOUT_PING)
            node_status[node_id] = {
                "host": base_url,
                "status": "online" if r.ok else "error",
                "last_check": time.time()
            }
        except Exception:
            node_status[node_id] = {
                "host": base_url,
                "status": "offline",
                "last_check": time.time()
            }
    return jsonify({"nodes": node_status})

@app.route("/best_model", methods=["GET"])
def get_best_model():
    try:
        ckpt_path = _resolve_best_ckpt_path()
        if not ckpt_path:
            return jsonify({"ok": False, "error": "best.pt not found"}), 404

        obj = torch.load(ckpt_path, map_location="cpu")

        # tu checkpoint guarda el state_dict en 'model_state'
        if "model_state" in obj:
            state_dict = obj["model_state"]
        else:
            return jsonify({"ok": False, "error": f"Formato desconocido: {list(obj.keys())}"}), 400

        b64 = state_to_b64(state_dict)
        stat = os.stat(ckpt_path)

        return jsonify({
            "ok": True,
            "model_state_b64": b64,
            "info": {
                "path": ckpt_path,
                "size_bytes": stat.st_size,
                "updated_at": stat.st_mtime,
                "extra": {k: obj[k] for k in obj if k != "model_state"}  # val_acc, epoch, session_id, updated_at...
            }
        })
    except Exception as e:
        logger.exception("Error en /best_model")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/predict_random", methods=["GET"])
def predict_random():
    """
    Devuelve muestras aleatorias del test de CIFAR-10 con predicciones top-k.
    Query params:
      - count: int (1..64)  default 12
      - topk:  int (1..10)  default 5
    """
    try:
        _ensure_infer_resources()
    except Exception as e:
        logger.exception("Error preparando recursos de inferencia:")
        return jsonify({"ok": False, "error": str(e)}), 500

    try:
        count = int(request.args.get("count", "12"))
        topk  = int(request.args.get("topk", "5"))
    except ValueError:
        return jsonify({"ok": False, "error": "Parámetros inválidos"}), 400

    count = max(1, min(count, 64))
    topk  = max(1, min(topk, 10))

    import random
    idxs = random.sample(range(len(_test_ds)), count)
    items = []

    with torch.no_grad():
        for idx in idxs:
            x_norm, y_true = _test_ds[idx]  # x_norm: [3,32,32] normalizado
            logits = _infer_model(x_norm.unsqueeze(0))  # [1,10]
            probs  = F.softmax(logits, dim=1)[0]        # [10]

            pvals, pidxs = torch.topk(probs, k=topk)
            preds = [{
                "class_index": int(ci),
                "class_name": CIFAR10_CLASSES[ci],
                "prob": float(p)
            } for p, ci in zip(pvals.tolist(), pidxs.tolist())]

            items.append({
                "image": _tensor_to_png_datauri(x_norm),
                "label_index": int(y_true),
                "label_name": CIFAR10_CLASSES[y_true],
                "predictions": preds
            })

    return jsonify({
        "ok": True,
        "count": len(items),
        "items": items,
        "ts": int(time.time())
    })

@app.route("/predict_image", methods=["POST"])
def predict_image():
    """
    Recibe una imagen y devuelve top-k predicciones.
    Formas de envío:
      - multipart/form-data con campo 'image'
      - application/json con campo 'image_b64' -> "data:image/...;base64,...." o sólo base64 puro
    Query param:
      - topk: int (1..10), default 5
    """
    try:
        _ensure_infer_resources()
    except Exception as e:
        logger.exception("Error preparando recursos de inferencia:")
        return jsonify({"ok": False, "error": str(e)}), 500

    try:
        topk = int(request.args.get("topk", "5"))
    except ValueError:
        return jsonify({"ok": False, "error": "Parámetro topk inválido"}), 400
    topk = max(1, min(topk, 10))

    pil_img = None

    # 1) multipart
    if "image" in request.files:
        f = request.files["image"]
        try:
            pil_img = Image.open(f.stream)
        except Exception:
            return jsonify({"ok": False, "error": "No se pudo abrir la imagen"}), 400

    # 2) JSON base64
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        b64s = data.get("image_b64")
        if not b64s:
            return jsonify({"ok": False, "error": "Falta image_b64"}), 400
        # soporta 'data:image/png;base64,....'
        if "," in b64s:
            b64s = b64s.split(",", 1)[1]
        try:
            raw = base64.b64decode(b64s)
            pil_img = Image.open(io.BytesIO(raw))
        except Exception:
            return jsonify({"ok": False, "error": "Base64 inválido o imagen corrupta"}), 400
    else:
        return jsonify({"ok": False, "error": "Provee 'image' (multipart) o 'image_b64' (JSON)"}), 400

    # Preproceso y predicción
    x_norm = _image_to_tensor(pil_img)  # [3,32,32]
    with torch.no_grad():
        logits = _infer_model(x_norm.unsqueeze(0))  # [1,10]
        probs  = F.softmax(logits, dim=1)[0]        # [10]
        pvals, pidxs = torch.topk(probs, k=topk)

    preds = [{
        "class_index": int(ci),
        "class_name": CIFAR10_CLASSES[ci],
        "prob": float(p)
    } for p, ci in zip(pvals.tolist(), pidxs.tolist())]

    # Para visualización del input (ya reescalado a 32x32)
    data_uri = _tensor_to_png_datauri(x_norm)

    return jsonify({
        "ok": True,
        "topk": topk,
        "predictions": preds,
        "image": data_uri,
        "ts": int(time.time())
    })

@app.errorhandler(404)
def not_found(err):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(err):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "6001"))
    logger.info(f"Iniciando controller en 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
