// App.jsx
import { useEffect, useRef, useState } from "react";

const NODES = [
  { id: "node-0", host: "10.128.0.5" },
  { id: "node-1", host: "10.128.0.6" },
  { id: "node-2", host: "10.128.0.7" },
  { id: "node-3", host: "10.128.0.8" },
  { id: "node-4", host: "10.128.0.9" },
];

export default function App() {
  const [selected, setSelected] = useState([NODES[0].id]);
  const [epochs, setEpochs] = useState(25);
  const [batchPerProc, setBatchPerProc] = useState(128);
  const [session, setSession] = useState(null);
  const [stats, setStats] = useState({}); // { nodeId: { cpu, ram, ts, ... } }
  const evtRef = useRef(null);

  const toggleNode = (id) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const startTraining = async () => {
    const body = {
      nodes: selected,
      epochs: Number(epochs),
      batch_per_proc: Number(batchPerProc),
    };
    const res = await fetch("http://CONTROLLER_HOST:5000/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();             
    setSession(data.session_id);

    // Cerrar SSE anterior si existía
    if (evtRef.current) evtRef.current.close();

    // Abrir SSE para métricas de esta sesión
    const es = new EventSource(
      `http://CONTROLLER_HOST:5000/events/${data.session_id}`
    );
    es.onmessage = (e) => {
      const payload = JSON.parse(e.data); // { node_id, cpu, ram, ts, epoch, acc, loss }
      setStats((prev) => ({ ...prev, [payload.node_id]: payload }));
    };
    es.onerror = () => {
      console.warn("SSE closed");
      es.close();
    };
    evtRef.current = es;
  };

  useEffect(() => {
    return () => { if (evtRef.current) evtRef.current.close(); };
  }, []);

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold">Sistema de cómputo distribuido</h1>

      <section className="space-y-2">
        <h2 className="text-xl font-semibold">Elige nodos</h2>
        <div className="flex gap-4 flex-wrap">
          {NODES.map((n) => (
            <label key={n.id} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={selected.includes(n.id)}
                onChange={() => toggleNode(n.id)}
              />
              <span>{n.id} ({n.host})</span>
            </label>
          ))}
        </div>
      </section>

      <section className="space-y-2">
        <h2 className="text-xl font-semibold">Hiperparámetros</h2>
        <div className="flex gap-6 items-end flex-wrap">
          <div>
            <label className="block text-sm font-medium">Epochs</label>
            <input type="number" min="1" className="border p-2 rounded"
              value={epochs} onChange={(e)=>setEpochs(e.target.value)} />
          </div>
          <div>
            <label className="block text-sm font-medium">Batch por proceso</label>
            <input type="number" min="32" step="32" className="border p-2 rounded"
              value={batchPerProc} onChange={(e)=>setBatchPerProc(e.target.value)} />
          </div>
          <button onClick={startTraining}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
            Entrenar
          </button>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold">Stats en vivo</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {selected.map((id) => {
            const s = stats[id];
            return (
              <div key={id} className="border rounded p-3">
                <div className="font-semibold">{id}</div>
                <div className="text-sm text-gray-600">session: {session ?? "-"}</div>
                <div className="mt-2">CPU: {s?.cpu?.toFixed?.(1) ?? 0}%</div>
                <div>RAM: {s?.ram?.toFixed?.(1) ?? 0}%</div>
                <div>Epoch: {s?.epoch ?? "-"}</div>
                <div>Loss: {s?.loss?.toFixed?.(4) ?? "-"}</div>
                <div>Acc: {s?.acc?.toFixed?.(2) ?? "-"}%</div>
                <div className="text-xs text-gray-500 mt-1">
                  {s?.ts ? new Date(s.ts).toLocaleTimeString() : ""}
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
