# controller.py
from flask import Flask, request, Response, jsonify
import requests, threading, time, uuid

app = Flask(__name__)

# Inventario nodos -> host agent
NODE_HOSTS = {
    "node-0": "http://10.128.0.5:6000",
    "node-1": "http://10.128.0.6:6000",
    "node-2": "http://10.128.0.7:6000",
    "node-3": "http://10.128.0.8:6000",
    "node-4": "http://10.128.0.9:6000",
}

SESSIONS = {}      # session_id -> {"nodes":[...], "running": True}
METRICS = {}       # session_id -> {node_id: last_metrics}

POLL_INTERVAL = 1.0  # s

def poll_metrics_loop(session_id):
    while SESSIONS.get(session_id, {}).get("running", False):
        nodes = SESSIONS[session_id]["nodes"]
        for idx, node_id in enumerate(nodes):
            base = NODE_HOSTS[node_id]
            try:
                r = requests.get(f"{base}/metrics", timeout=0.8)
                if r.ok:
                    data = r.json()   # {cpu, ram, ts, epoch, acc, loss}
                    METRICS.setdefault(session_id, {})[node_id] = data
            except Exception:
                pass
        time.sleep(POLL_INTERVAL)

@app.route("/train", methods=["POST"])
def start_train():
    payload = request.get_json()
    nodes = payload["nodes"]
    epochs = int(payload["epochs"])
    bpp = int(payload.get("batch_per_proc", 128))

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"nodes": nodes, "running": True}
    METRICS[session_id] = {}

    # WORLD_SIZE = #nodos * nproc_per_node (aquí 1 por nodo)
    world_size = len(nodes)

    # Lanza a cada agent con su RANK
    for rank, node_id in enumerate(nodes):
        base = NODE_HOSTS[node_id]
        body = {
            "session_id": session_id,
            "rank": rank,
            "world_size": world_size,
            "epochs": epochs,
            "batch_per_proc": bpp,
            "master_addr": NODE_HOSTS[nodes[0]].split("//")[1].split(":")[0],
            "master_port": 29500,
            # Ruta del script de entrenamiento en los nodos:
            "train_script": "/home/ubuntu/main.py"
        }
        requests.post(f"{base}/start_train", json=body, timeout=3)

    # Hilo que “pull-ea” métricas
    t = threading.Thread(target=poll_metrics_loop, args=(session_id,), daemon=True)
    t.start()

    return jsonify({"session_id": session_id})

@app.route("/events/<session_id>")
def sse(session_id):
    def stream():
        last = {}
        while SESSIONS.get(session_id, {}).get("running", False):
            snap = METRICS.get(session_id, {})
            # Emitir una actualización por nodo cuando cambie
            for node_id, m in snap.items():
                key = (node_id, m.get("ts"))
                if last.get(node_id) != key:
                    last[node_id] = key
                    yield f"data: {m | {'node_id': node_id}}\n\n"
            time.sleep(0.3)
        # cierre de sesión
        yield f"data: { {'node_id':'all','done':True} }\n\n"

    return Response(stream(), mimetype="text/event-stream")

@app.route("/stop/<session_id>", methods=["POST"])
def stop(session_id):
    sess = SESSIONS.get(session_id)
    if not sess: return jsonify({"ok": True})
    sess["running"] = False
    for node_id in sess["nodes"]:
        base = NODE_HOSTS[node_id]
        try:
            requests.post(f"{base}/stop", json={"session_id": session_id}, timeout=1)
        except Exception:
            pass
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
