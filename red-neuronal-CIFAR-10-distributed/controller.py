# controller.py
import os
import io
import json
import time
import uuid
import base64
import logging
import threading
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================
# Config & Logging
# ==========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("controller")

# Ajusta aquí los hosts de tus nodos (agents)
NODE_HOSTS = {
    "node-0": "http://192.168.20.3:6000",
    "node-1": "http://10.128.0.6:6000",
    "node-2": "http://10.128.0.7:6000",
    "node-3": "http://10.128.0.8:6000",
    "node-4": "http://10.128.0.9:6000",
}

# Timeouts / intervalos
HTTP_TIMEOUT_TRAIN = float(os.environ.get("HTTP_TIMEOUT_TRAIN", "120"))  # s por /train_batch
HTTP_TIMEOUT_PING  = float(os.environ.get("HTTP_TIMEOUT_PING", "2"))     # s para /health
HTTP_TIMEOUT_METR  = float(os.environ.get("HTTP_TIMEOUT_METR", "1.5"))   # s para /metrics
SSE_HEARTBEAT_EVERY = 10  # iteraciones del loop SSE (~0.5s * 10 = 5s aprox)
METRICS_POLL_INTERVAL = float(os.environ.get("METRICS_POLL_INTERVAL", "1.0"))  # s

# NUEVO: desactiva polling de /metrics por defecto (0=off, 1=on)
ENABLE_REALTIME_POLL = os.environ.get("ENABLE_REALTIME_POLL", "0") == "1"

# ==========================
# Flask app
# ==========================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================
# Modelo (igual al del agent)
# ==========================
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

# ==========================
# Estado global en memoria
# ==========================
SESSIONS: Dict[str, Dict[str, Any]] = {}   # session_id -> info
METRICS:  Dict[str, Dict[str, Dict[str, Any]]] = {}  # session_id -> { node_id: last_metrics }

# ==========================
# Utils: (de)serialización & agregación
# ==========================
def state_to_b64(sd: Dict[str, torch.Tensor]) -> str:
    bio = io.BytesIO()
    torch.save(sd, bio)
    return base64.b64encode(bio.getvalue()).decode("ascii")

def b64_to_state(b64s: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(b64s.encode("ascii"))
    bio = io.BytesIO(raw)
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

def post_json(url: str, payload: Dict[str, Any], timeout: float):
    return requests.post(url, json=payload, timeout=timeout)

# ==========================
# Polling de /metrics (CPU/RAM)
# ==========================
def _poll_metrics_loop(session_id: str):
    """Hilo que sondea /metrics en cada nodo y fusiona CPU/RAM en METRICS[session_id]."""
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
                        "batch_size": int(data.get("batch_size", 0)),
                        "node_id": node_id
                    })
                else:
                    logger.debug(f"[{session_id}] /metrics {node_id} HTTP {r.status_code}")
            except requests.exceptions.RequestException:
                # nodos caídos o lentos: ignora
                pass
        time.sleep(METRICS_POLL_INTERVAL)
    logger.info(f"[{session_id}] Fin de poll de /metrics")

# ==========================
# Loop de entrenamiento por rondas (FedAvg)
# ==========================
def _train_fedavg_loop(session_id: str,
                       nodes: List[str],
                       epochs: int,
                       init_b64: str,
                       lr: float, momentum: float, weight_decay: float,
                       seed: int,
                       shards_per_epoch: int):
    try:
        logger.info(f"[{session_id}] Iniciando FedAvg: epochs={epochs}, shards_per_epoch={shards_per_epoch}, nodes={nodes}")
        current_b64 = init_b64
        total_batches = max(1, len(nodes) * max(1, shards_per_epoch))
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
                        }
                        futures.append((node_id, batch_id, ex.submit(post_json, f"{base}/train_batch", payload, HTTP_TIMEOUT_TRAIN)))

                    sd_list, weights, pernode = [], [], []
                    for node_id, batch_id, fut in futures:
                        try:
                            r = fut.result()
                            if not r.ok:
                                raise RuntimeError(f"HTTP {r.status_code}")
                            resp = r.json()

                            # Registrar métricas de rendimiento + recursos del nodo (sin necesidad de /metrics)
                            METRICS.setdefault(session_id, {}).setdefault(node_id, {})
                            # Recursos enviados por agent al final del batch
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
                                "node_id": node_id
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
            logger.info(f"[{session_id}] Epoch {epoch}/{epochs}: loss={g_loss:.4f} acc={g_acc:.2f}%")

        SESSIONS[session_id]["status"] = "completed"
        SESSIONS[session_id]["running"] = False
    except Exception as e:
        logger.exception(f"[{session_id}] Error en loop FedAvg")
        SESSIONS[session_id]["status"] = "failed"
        SESSIONS[session_id]["running"] = False

# ==========================
# Endpoints HTTP
# ==========================
@app.route("/train", methods=["POST"])
def start_train():
    """
    Inicia una sesión de entrenamiento orquestado (FedAvg).
    Body JSON:
    {
      "nodes": ["node-0","node-1",...],
      "epochs": 25,
      "lr": 0.1,
      "momentum": 0.9,
      "weight_decay": 0.0005,
      "seed": 1337,
      "shards_per_epoch": 1
    }
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
        }
        METRICS[session_id] = {}

        # Modelo global inicial
        model = SmallCIFAR().cpu()
        init_b64 = state_to_b64(model.state_dict())

        # Lanzar hilos: 1) entrenamiento, 2) poll de métricas (solo si está habilitado)
        t_train = threading.Thread(
            target=_train_fedavg_loop,
            args=(session_id, nodes, epochs, init_b64, lr, momentum, weight_decay, seed, shards_per_epoch),
            daemon=True
        )
        t_train.start()

        if ENABLE_REALTIME_POLL:
            t_poll = threading.Thread(target=_poll_metrics_loop, args=(session_id,), daemon=True)
            t_poll.start()
            logger.info(f"[{session_id}] Realtime /metrics polling ACTIVADO (ENABLE_REALTIME_POLL=1)")
        else:
            logger.info(f"[{session_id}] Realtime /metrics polling DESACTIVADO (ENABLE_REALTIME_POLL=0)")

        logger.info(f"Iniciada sesión {session_id} con nodos: {nodes}")
        return jsonify({"session_id": session_id, "status": "started"})
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
    """Verifica estado de los nodos vía /health (si responde OK => online)."""
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

# ==========================
# Errores HTTP
# ==========================
@app.errorhandler(404)
def not_found(err):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(err):
    return jsonify({"error": "Internal server error"}), 500

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "6001"))
    logger.info(f"Iniciando controller en 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
