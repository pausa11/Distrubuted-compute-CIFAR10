import { useEffect, useRef, useState, useCallback } from "react";

const NODES = [
  { id: "node-0", host: "34.132.166.106" },
  { id: "node-1", host: "34.121.35.52" },
  { id: "node-2", host: "34.9.209.76" },
  { id: "node-3", host: "34.53.3.174" },
  { id: "node-4", host: "34.11.238.13" },
];

const API_BASE = "https://sms-adequate-attention-intervals.trycloudflare.com";

function PredictionCard({ item }) {
  const { image, label_name, predictions } = item;
  return (
    <div className="border rounded-lg overflow-hidden bg-white">
      <img
        src={image}
        alt={label_name}
        className="w-full h-40 object-contain bg-gray-50"
      />
      <div className="p-3">
        <div className="text-xs text-gray-500">Etiqueta real</div>
        <div className="font-semibold mb-2">{label_name}</div>
        <div className="text-xs text-gray-500 mb-1">Top-K</div>
        <ul className="text-sm space-y-1">
          {predictions.map((p, i) => (
            <li key={i} className="flex justify-between">
              <span>{p.class_name}</span>
              <span className="tabular-nums">
                {(p.prob * 100).toFixed(2)}%
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default function App() {
  const [selected, setSelected] = useState([NODES[0].id]);
  const [epochs, setEpochs] = useState(25);
  const [batchPerProc, setBatchPerProc] = useState(128);
  const [session, setSession] = useState(null);
  const [sessionStatus, setSessionStatus] = useState("idle");
  const [stats, setStats] = useState({});
  const [connectionStatus, setConnectionStatus] = useState("disconnected");
  const [error, setError] = useState(null);
  const [trainingProgress, setTrainingProgress] = useState({});
  const evtRef = useRef(null);

  // --- Modelo global (best.pt) ---
  const [bestModelB64, setBestModelB64] = useState(null);
  const [bestModelInfo, setBestModelInfo] = useState(null);
  const [bestLoading, setBestLoading] = useState(false);

  // --- NUEVO: Predicciones aleatorias CIFAR-10 ---
  const [samples, setSamples] = useState([]);
  const [predLoading, setPredLoading] = useState(false);
  const [predCount, setPredCount] = useState(12);
  const [predTopK, setPredTopK] = useState(5);

  // --- NUEVO: Predicción por imagen subida ---
  const [uploadPreview, setUploadPreview] = useState(null);
  const [uploadPreds, setUploadPreds] = useState(null); // respuesta completa del backend
  const [uploadTopK, setUploadTopK] = useState(5);
  const [uploadLoading, setUploadLoading] = useState(false);
  const fileInputRef = useRef(null);

  const toggleNode = (id) => {
    if (sessionStatus === "running") return;
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const connectSSE = useCallback(
    (sessionId) => {
      if (evtRef.current) {
        evtRef.current.close();
      }

      console.log(`Conectando SSE para sesión: ${sessionId}`);
      const es = new EventSource(`${API_BASE}/events/${sessionId}`);
      let reconnectAttempts = 0;
      const maxReconnectAttempts = 5;

      es.onopen = () => {
        console.log("SSE conectado exitosamente");
        setConnectionStatus("connected");
        setError(null);
        reconnectAttempts = 0;
      };

      es.onmessage = (e) => {
        try {
          const payload = JSON.parse(e.data);
          if (payload.node_id && payload.node_id !== "all") {
            setStats((prev) => ({
              ...prev,
              [payload.node_id]: payload,
            }));

            if (payload.epoch) {
              setTrainingProgress((prev) => ({
                ...prev,
                [payload.node_id]: {
                  epoch: payload.epoch,
                  totalEpochs: epochs,
                  progress: (payload.epoch / epochs) * 100,
                },
              }));
            }
          }
        } catch (err) {
          console.error("Error parsing SSE data:", err);
          setError(`Error procesando datos: ${err.message}`);
        }
      };

      es.addEventListener("connected", () => {
        console.log("Evento SSE: connected");
        setConnectionStatus("connected");
      });

      es.addEventListener("completed", () => {
        console.log("Evento SSE: completed");
        setSessionStatus("completed");
        setConnectionStatus("disconnected");
        es.close();
      });

      es.addEventListener("heartbeat", () => {});

      es.addEventListener("metrics", (e) => {
        try {
          const payload = JSON.parse(e.data);
          if (payload.node_id) {
            setStats((prev) => ({
              ...prev,
              [payload.node_id]: payload,
            }));
          }
        } catch (err) {
          console.error("Error parsing metrics event:", err);
        }
      });

      es.onerror = () => {
        console.error("SSE error event");
        if (es.readyState === EventSource.CONNECTING) {
          console.log("SSE reconnecting...");
          setConnectionStatus("reconnecting");
        } else if (es.readyState === EventSource.CLOSED) {
          console.log("SSE connection closed");
          setConnectionStatus("disconnected");

          if (
            sessionStatus === "running" &&
            reconnectAttempts < maxReconnectAttempts
          ) {
            reconnectAttempts++;
            console.log(
              `Reintentando conexión SSE (${reconnectAttempts}/${maxReconnectAttempts})...`
            );

            setTimeout(() => {
              if (sessionStatus === "running") {
                connectSSE(sessionId);
              }
            }, 2000 * reconnectAttempts);
          } else if (reconnectAttempts >= maxReconnectAttempts) {
            setError(
              "No se pudo restablecer la conexión con el servidor. Recarga la página."
            );
            setConnectionStatus("failed");
          }
        } else {
          setConnectionStatus("error");
          setError("Error de conexión SSE");
        }
      };

      evtRef.current = es;
    },
    [epochs, sessionStatus]
  );

  const startTraining = async () => {
    if (selected.length === 0) {
      setError("Selecciona al menos un nodo");
      return;
    }

    setError(null);
    setSessionStatus("starting");

    try {
      const body = {
        nodes: selected,
        epochs: Number(epochs),
        batch_per_proc: Number(batchPerProc),
      };

      const res = await fetch(`${API_BASE}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setSession(data.session_id);
      setSessionStatus("running");

      setStats({});
      setTrainingProgress({});

      connectSSE(data.session_id);
    } catch (err) {
      setError(`Error iniciando entrenamiento: ${err.message}`);
      setSessionStatus("failed");
    }
  };

  const stopTraining = async () => {
    if (!session) return;

    setSessionStatus("stopping");

    try {
      const res = await fetch(`${API_BASE}/stop/${session}`, {
        method: "POST",
      });

      if (res.ok) {
        setSessionStatus("stopped");
        if (evtRef.current) {
          evtRef.current.close();
        }
        setConnectionStatus("disconnected");
      }
    } catch (err) {
      setError(`Error deteniendo entrenamiento: ${err.message}`);
    }
  };

  useEffect(() => {
    return () => {
      if (evtRef.current) {
        evtRef.current.close();
      }
    };
  }, []);

  // ----- Utilidades Modelo Global -----
  function base64ToBlob(base64) {
    const byteChars = atob(base64);
    const byteNumbers = new Array(byteChars.length);
    for (let i = 0; i < byteChars.length; i++)
      byteNumbers[i] = byteChars.charCodeAt(i);
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: "application/octet-stream" });
  }

  const fetchBestModel = useCallback(async () => {
    setError(null);
    setBestLoading(true);
    try {
      const res = await fetch(`${API_BASE}/best_model`, { method: "GET" });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data?.error || `HTTP ${res.status}`);
      }
      setBestModelB64(data.model_state_b64 || null);

      let meta = { ...(data.info || {}) };
      if (data.info?.extra) {
        if (typeof data.info.extra.val_acc !== "undefined")
          meta.val_acc = data.info.extra.val_acc;
        if (typeof data.info.extra.epoch !== "undefined")
          meta.epoch = data.info.extra.epoch;
      }
      setBestModelInfo(meta);
    } catch (e) {
      setError(`No se pudo obtener el modelo: ${e.message}`);
      setBestModelB64(null);
      setBestModelInfo(null);
    } finally {
      setBestLoading(false);
    }
  }, []);

  const downloadBestModel = useCallback(() => {
    if (!bestModelB64) return;
    const blob = base64ToBlob(bestModelB64);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "best_from_controller.pt";
    document.body.appendChild(a);
    a.click();
    URL.revokeObjectURL(url);
    a.remove();
  }, [bestModelB64]);

  const copyBestModelB64 = useCallback(async () => {
    if (!bestModelB64) return;
    try {
      await navigator.clipboard.writeText(bestModelB64);
    } catch {
      setError("No se pudo copiar el modelo en base64.");
    }
  }, [bestModelB64]);

  // ----- NUEVO: Obtener muestras aleatorias -----
  const fetchRandomPreds = useCallback(async () => {
    setPredLoading(true);
    setError(null);
    try {
      const url = `${API_BASE}/predict_random?count=${predCount}&topk=${predTopK}`;
      const res = await fetch(url);
      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data?.error || `HTTP ${res.status}`);
      }
      setSamples(data.items || []);
    } catch (e) {
      setError(`No se pudieron obtener predicciones: ${e.message}`);
      setSamples([]);
    } finally {
      setPredLoading(false);
    }
  }, [predCount, predTopK]);

  // ----- NUEVO: Manejo de upload imagen -----
  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    setUploadPreds(null);
    if (!f) {
      setUploadPreview(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => setUploadPreview(ev.target.result);
    reader.readAsDataURL(f);
  };

  const predictFromFile = useCallback(async () => {
    if (!fileInputRef.current?.files?.[0]) {
      setError("Selecciona una imagen primero.");
      return;
    }
    setUploadLoading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("image", fileInputRef.current.files[0]);
      const res = await fetch(
        `${API_BASE}/predict_image?topk=${uploadTopK}`,
        { method: "POST", body: fd }
      );
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data?.error || `HTTP ${res.status}`);
      setUploadPreds(data); // { image, predictions, topk, ts }
    } catch (e) {
      setError(`No se pudo predecir la imagen: ${e.message}`);
      setUploadPreds(null);
    } finally {
      setUploadLoading(false);
    }
  }, [uploadTopK]);

  const isTrainingActive = ["starting", "running"].includes(sessionStatus);
  const canModifySettings = !isTrainingActive;

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      <header className="border-b pb-4">
        <h1 className="text-3xl font-bold text-gray-900">
          Sistema de Cómputo Distribuido
        </h1>
        <div className="flex items-center gap-4 mt-2">
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                connectionStatus === "connected"
                  ? "bg-green-500"
                  : connectionStatus === "reconnecting"
                  ? "bg-yellow-500 animate-pulse"
                  : connectionStatus === "error" || connectionStatus === "failed"
                  ? "bg-red-500"
                  : "bg-gray-400"
              }`}
            ></div>
            <span className="text-sm text-gray-600">
              {connectionStatus === "reconnecting"
                ? "Reconectando..."
                : `Conexión: ${connectionStatus}`}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                sessionStatus === "running"
                  ? "bg-blue-500 animate-pulse"
                  : sessionStatus === "completed"
                  ? "bg-green-500"
                  : sessionStatus === "failed"
                  ? "bg-red-500"
                  : "bg-gray-400"
              }`}
            ></div>
            <span className="text-sm text-gray-600">Estado: {sessionStatus}</span>
          </div>
        </div>
      </header>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <div className="text-red-600 text-sm">{error}</div>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-600 hover:text-red-800"
            >
              ×
            </button>
          </div>
        </div>
      )}

      <section className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Configuración de Nodos</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {NODES.map((n) => (
            <label
              key={n.id}
              className={`flex items-center gap-3 p-4 border rounded-lg cursor-pointer transition-colors ${
                selected.includes(n.id)
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-200"
              } ${
                !canModifySettings
                  ? "opacity-50 cursor-not-allowed"
                  : "hover:border-gray-300"
              }`}
            >
              <input
                type="checkbox"
                checked={selected.includes(n.id)}
                onChange={() => toggleNode(n.id)}
                disabled={!canModifySettings}
                className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <div>
                <div className="font-medium">{n.id}</div>
                <div className="text-sm text-gray-500">{n.host}</div>
              </div>
            </label>
          ))}
        </div>
      </section>

      <section className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Hiperparámetros</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Epochs
            </label>
            <input
              type="number"
              min="1"
              max="1000"
              className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              value={epochs}
              onChange={(e) => setEpochs(e.target.value)}
              disabled={!canModifySettings}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Batch por proceso
            </label>
            <input
              type="number"
              min="32"
              step="32"
              className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              value={batchPerProc}
              onChange={(e) => setBatchPerProc(e.target.value)}
              disabled={!canModifySettings}
            />
          </div>
          <div className="flex items-end">
            {sessionStatus === "idle" ||
            sessionStatus === "completed" ||
            sessionStatus === "failed" ? (
              <button
                onClick={startTraining}
                disabled={selected.length === 0}
                className="w-full bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Iniciar Entrenamiento
              </button>
            ) : sessionStatus === "starting" ? (
              <button
                disabled
                className="w-full bg-gray-400 text-white px-4 py-2 rounded-lg cursor-not-allowed"
              >
                Iniciando...
              </button>
            ) : (
              <button
                onClick={stopTraining}
                className="w-full bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
              >
                Detener Entrenamiento
              </button>
            )}
          </div>
        </div>
      </section>

      {session && (
        <section className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Progreso del Entrenamiento</h2>
            <div className="text-sm text-gray-600">
              Sesión: {session.substring(0, 8)}...
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {selected.map((nodeId) => {
              const progress = trainingProgress[nodeId];
              return (
                <div key={nodeId} className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium">{nodeId}</h3>
                    {progress && (
                      <span className="text-sm text-gray-600">
                        {progress.epoch}/{progress.totalEpochs}
                      </span>
                    )}
                  </div>
                  {progress && (
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${Math.min(progress.progress, 100)}%` }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      <section className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Métricas en Tiempo Real</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {selected.map((id) => {
            const s = stats[id];
            const hasData = s && Object.keys(s).length > 0;

            return (
              <div key={id} className="border border-gray-200 rounded-lg p-4 bg-gray-50">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold text-lg">{id}</h3>
                  <div
                    className={`w-3 h-3 rounded-full ${
                      hasData ? "bg-green-500" : "bg-gray-400"
                    }`}
                  />
                </div>

                {hasData ? (
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-white rounded p-2">
                        <div className="text-xs text-gray-500">CPU</div>
                        <div className="text-lg font-medium">
                          {s.cpu?.toFixed?.(1) ??
                            s.avg_cpu_pct?.toFixed?.(1) ??
                            0}
                          %
                        </div>
                      </div>
                      <div className="bg-white rounded p-2">
                        <div className="text-xs text-gray-500">RAM</div>
                        <div className="text-lg font-medium">
                          {s.ram?.toFixed?.(1) ??
                            s.peak_ram_pct?.toFixed?.(1) ??
                            0}
                          %
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-2">
                      <div className="bg-white rounded p-2 text-center">
                        <div className="text-xs text-gray-500">Epoch</div>
                        <div className="font-medium">{s.epoch ?? "-"}</div>
                      </div>
                      <div className="bg-white rounded p-2 text-center">
                        <div className="text-xs text-gray-500">Loss</div>
                        <div className="font-medium">
                          {s.loss?.toFixed?.(4) ?? "-"}
                        </div>
                      </div>
                      <div className="bg-white rounded p-2 text-center">
                        <div className="text-xs text-gray-500">Acc</div>
                        <div className="font-medium">
                          {s.acc?.toFixed?.(2) ?? "-"}%
                        </div>
                      </div>
                    </div>

                    {s.ts && (
                      <div className="text-xs text-gray-500 text-center pt-2 border-t">
                        Última actualización:{" "}
                        {new Date(s.ts * 1000).toLocaleTimeString()}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    <div className="text-2xl mb-2">📊</div>
                    <div>Esperando datos...</div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {selected.length === 0 && (
          <div className="text-center text-gray-500 py-12">
            <div className="text-4xl mb-4">🖥️</div>
            <div className="text-lg">Selecciona nodos para ver las métricas</div>
          </div>
        )}
      </section>

      {/* === Sección Modelo Global best.pt === */}
      <section className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Modelo Global (best.pt)</h2>
          <div className="flex gap-2">
            <button
              onClick={fetchBestModel}
              className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              disabled={bestLoading}
            >
              {bestLoading ? "Cargando..." : "Obtener modelo"}
            </button>
            <button
              onClick={downloadBestModel}
              className="bg-gray-800 text-white px-4 py-2 rounded-lg hover:bg-gray-900 disabled:opacity-50"
              disabled={!bestModelB64}
            >
              Descargar .pt
            </button>
            <button
              onClick={copyBestModelB64}
              className="bg-gray-200 text-gray-800 px-4 py-2 rounded-lg hover:bg-gray-300 disabled:opacity-50"
              disabled={!bestModelB64}
            >
              Copiar base64
            </button>
          </div>
        </div>

        {bestModelInfo ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="bg-gray-50 p-3 rounded border">
              <div className="text-gray-500">Ruta</div>
              <div className="font-medium break-all">
                {bestModelInfo.path || "-"}
              </div>
            </div>
            <div className="bg-gray-50 p-3 rounded border">
              <div className="text-gray-500">Tamaño</div>
              <div className="font-medium">
                {bestModelInfo.size_bytes?.toLocaleString?.() ?? "-"} bytes
              </div>
            </div>
            <div className="bg-gray-50 p-3 rounded border">
              <div className="text-gray-500">Actualizado</div>
              <div className="font-medium">
                {bestModelInfo.updated_at
                  ? new Date(bestModelInfo.updated_at * 1000).toLocaleString()
                  : "-"}
              </div>
            </div>
            {"val_acc" in bestModelInfo && (
              <div className="bg-gray-50 p-3 rounded border">
                <div className="text-gray-500">Val Acc</div>
                <div className="font-medium">
                  {typeof bestModelInfo.val_acc === "number"
                    ? `${(bestModelInfo.val_acc * 100).toFixed(2)}%`
                    : String(bestModelInfo.val_acc)}
                </div>
              </div>
            )}
            {"epoch" in bestModelInfo && (
              <div className="bg-gray-50 p-3 rounded border">
                <div className="text-gray-500">Epoch</div>
                <div className="font-medium">{bestModelInfo.epoch}</div>
              </div>
            )}
            <div className="bg-gray-50 p-3 rounded border md:col-span-3">
              <div className="text-gray-500">Estado</div>
              <div className="font-medium">
                {bestModelB64
                  ? "Modelo cargado (base64 listo para usar)"
                  : "Aún no cargado"}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-gray-500">Aún no has solicitado el modelo.</div>
        )}
      </section>

      {/* === NUEVO: Probar modelo con CIFAR-10 (aleatorio) === */}
      <section className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Probar modelo con CIFAR-10</h2>
          <div className="flex gap-2">
            <input
              type="number"
              min="1"
              max="64"
              value={predCount}
              onChange={(e) => setPredCount(Number(e.target.value))}
              className="w-24 border border-gray-300 rounded px-2 py-1"
              disabled={predLoading}
              title="Número de muestras"
            />
            <input
              type="number"
              min="1"
              max="10"
              value={predTopK}
              onChange={(e) => setPredTopK(Number(e.target.value))}
              className="w-24 border border-gray-300 rounded px-2 py-1"
              disabled={predLoading}
              title="Top-K"
            />
            <button
              onClick={fetchRandomPreds}
              className="bg-emerald-600 text-white px-4 py-2 rounded-lg hover:bg-emerald-700 disabled:opacity-50"
              disabled={predLoading}
            >
              {predLoading ? "Cargando..." : "Obtener muestras"}
            </button>
          </div>
        </div>

        {samples.length === 0 ? (
          <div className="text-gray-500">Aún no has solicitado predicciones.</div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {samples.map((it, idx) => (
              <PredictionCard key={idx} item={it} />
            ))}
          </div>
        )}
      </section>

      {/* === NUEVO: Probar con una imagen subida === */}
      <section className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Probar con una imagen</h2>

        <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-end">
          <div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-700 file:mr-3 file:py-2 file:px-3 file:rounded file:border-0 file:bg-gray-100 file:text-gray-700 hover:file:bg-gray-200"
            />
            {uploadPreview && (
              <img
                src={uploadPreview}
                alt="preview"
                className="mt-2 w-32 h-32 object-contain border rounded bg-gray-50"
              />
            )}
          </div>

          <div className="flex gap-2 items-center">
            <label className="text-sm text-gray-700">Top-K</label>
            <input
              type="number"
              min="1"
              max="10"
              value={uploadTopK}
              onChange={(e) => setUploadTopK(Number(e.target.value))}
              className="w-24 border border-gray-300 rounded px-2 py-1"
              disabled={uploadLoading}
            />
            <button
              onClick={predictFromFile}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
              disabled={uploadLoading || !fileInputRef.current}
            >
              {uploadLoading ? "Prediciendo..." : "Predecir imagen"}
            </button>
          </div>
        </div>

        {uploadPreds && (
          <div className="mt-6">
            <PredictionCard
              item={{
                image: uploadPreds.image,
                label_name: "Entrada",
                predictions: uploadPreds.predictions || [],
              }}
            />
          </div>
        )}
      </section>
    </div>
  );
}
