import { useEffect, useRef, useState, useCallback } from "react";

const NODES = [
  { id: "node-0", host: "34.132.166.106" },
  { id: "node-1", host: "34.121.35.52" },
  { id: "node-2", host: "34.9.209.76" },
  { id: "node-3", host: "34.53.3.174" },
  { id: "node-4", host: "34.11.238.13" },
];

const API_BASE = "http://34.135.213.57:6001";

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

  const toggleNode = (id) => {
    if (sessionStatus === "running") return;
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const connectSSE = useCallback((sessionId) => {
    if (evtRef.current) {
      evtRef.current.close();
    }

    console.log(`Conectando SSE para sesión: ${sessionId}`);
    
    const es = new EventSource(`${API_BASE}/events/${sessionId}`);
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    
    es.onopen = (event) => {
      console.log("SSE conectado exitosamente");
      setConnectionStatus("connected");
      setError(null);
      reconnectAttempts = 0;
    };

    es.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data);
        console.log("Datos SSE recibidos:", payload);
        
        if (payload.node_id && payload.node_id !== "all") {
          setStats((prev) => ({
            ...prev,
            [payload.node_id]: payload
          }));
          
          if (payload.epoch) {
            setTrainingProgress(prev => ({
              ...prev,
              [payload.node_id]: {
                epoch: payload.epoch,
                totalEpochs: epochs,
                progress: (payload.epoch / epochs) * 100
              }
            }));
          }
        }
      } catch (err) {
        console.error("Error parsing SSE data:", err);
        setError(`Error procesando datos: ${err.message}`);
      }
    };

    es.addEventListener('connected', (e) => {
      console.log("Evento SSE: connected");
      setConnectionStatus("connected");
    });

    es.addEventListener('completed', (e) => {
      console.log("Evento SSE: completed");
      setSessionStatus("completed");
      setConnectionStatus("disconnected");
      es.close();
    });

    es.addEventListener('heartbeat', (e) => {
      // Mantener conexión viva
    });

    es.addEventListener('metrics', (e) => {
      try {
        const payload = JSON.parse(e.data);
        if (payload.node_id) {
          setStats((prev) => ({
            ...prev,
            [payload.node_id]: payload
          }));
        }
      } catch (err) {
        console.error("Error parsing metrics event:", err);
      }
    });

    es.onerror = (event) => {
      console.error("SSE error event:", event);
      
      if (es.readyState === EventSource.CONNECTING) {
        console.log("SSE reconnecting...");
        setConnectionStatus("reconnecting");
      } else if (es.readyState === EventSource.CLOSED) {
        console.log("SSE connection closed");
        setConnectionStatus("disconnected");
        
        if (sessionStatus === "running" && reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          console.log(`Reintentando conexión SSE (${reconnectAttempts}/${maxReconnectAttempts})...`);
          
          setTimeout(() => {
            if (sessionStatus === "running") {
              connectSSE(sessionId);
            }
          }, 2000 * reconnectAttempts);
        } else if (reconnectAttempts >= maxReconnectAttempts) {
          setError("No se pudo restablecer la conexión con el servidor. Recarga la página.");
          setConnectionStatus("failed");
        }
      } else {
        setConnectionStatus("error");
        setError("Error de conexión SSE");
      }
    };

    evtRef.current = es;
  }, [epochs, sessionStatus]);

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
        method: "POST"
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

  const isTrainingActive = ["starting", "running"].includes(sessionStatus);
  const canModifySettings = !isTrainingActive;

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      <header className="border-b pb-4">
        <h1 className="text-3xl font-bold text-gray-900">Sistema de Cómputo Distribuido</h1>
        <div className="flex items-center gap-4 mt-2">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${
              connectionStatus === "connected" ? "bg-green-500" : 
              connectionStatus === "reconnecting" ? "bg-yellow-500 animate-pulse" :
              connectionStatus === "error" || connectionStatus === "failed" ? "bg-red-500" : "bg-gray-400"
            }`}></div>
            <span className="text-sm text-gray-600">
              {connectionStatus === "reconnecting" ? "Reconectando..." : `Conexión: ${connectionStatus}`}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${
              sessionStatus === "running" ? "bg-blue-500 animate-pulse" :
              sessionStatus === "completed" ? "bg-green-500" :
              sessionStatus === "failed" ? "bg-red-500" : "bg-gray-400"
            }`}></div>
            <span className="text-sm text-gray-600">
              Estado: {sessionStatus}
            </span>
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
                selected.includes(n.id) ? "border-blue-500 bg-blue-50" : "border-gray-200"
              } ${!canModifySettings ? "opacity-50 cursor-not-allowed" : "hover:border-gray-300"}`}
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
            {sessionStatus === "idle" || sessionStatus === "completed" || sessionStatus === "failed" ? (
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
                      ></div>
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
                  <div className={`w-3 h-3 rounded-full ${
                    hasData ? "bg-green-500" : "bg-gray-400"
                  }`}></div>
                </div>
                
                {hasData ? (
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-white rounded p-2">
                        <div className="text-xs text-gray-500">CPU</div>
                        <div className="text-lg font-medium">
                          {s.cpu?.toFixed?.(1) ?? 0}%
                        </div>
                      </div>
                      <div className="bg-white rounded p-2">
                        <div className="text-xs text-gray-500">RAM</div>
                        <div className="text-lg font-medium">
                          {s.ram?.toFixed?.(1) ?? 0}%
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
                        Última actualización: {new Date(s.ts * 1000).toLocaleTimeString()}
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
    </div>
  );
}