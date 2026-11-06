import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { cliModels, serverModels } from "../api";

type Props = { kind: "server" | "cli"; url?: string; tool?: string; value: string; onChange(v: string): void; disabled?: boolean; placeholder?: string };
const cache = new Map<string, string[]>();

export default function ModelSelect({ kind, url = "", tool = "", value, onChange, disabled, placeholder }: Props) {
  const key = kind === "server" ? `server:${url}` : `cli:${tool}`;
  const [models, setModels] = useState<string[]>(cache.get(key) ?? []);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  const [custom, setCustom] = useState(false);
  const fetchModels = async () => {
    if (kind === "server" ? !url.trim() : !tool) { setModels([]); return; }
    setLoading(true); setFailed(false);
    try {
      const result = kind === "server" ? await serverModels(url) : await cliModels(tool);
      cache.set(key, result); setModels(result);
    } catch { setFailed(true); setModels([]); }
    finally { setLoading(false); }
  };
  useEffect(() => {
    setModels(cache.get(key) ?? []); setCustom(false);
    const timer = setTimeout(() => { void fetchModels(); }, kind === "server" ? 350 : 0);
    return () => clearTimeout(timer);
  }, [key]);
  const fallback = failed || models.length === 0 || custom;
  const hint = kind === "server" ? "Couldn't list models from this server — type a model name" : "This tool can't list its models — type one";
  return <div className="model-select-control">
    {fallback ? <input className="composer-model-input" value={value} placeholder={placeholder ?? "Default"} disabled={disabled} onChange={e => onChange(e.target.value)} /> : <select className="composer-model-select" value={value} disabled={disabled || loading} onChange={e => e.target.value === "__custom" ? setCustom(true) : onChange(e.target.value)}>
      <option value="">— default —</option>
      {value && !models.includes(value) && <option value={value}>{value}</option>}
      {models.map(model => <option key={model} value={model}>{model}</option>)}
      <option value="__custom">Custom…</option>
    </select>}
    <button type="button" className="ghost model-select-refresh" title="Refresh models" aria-label="Refresh models" disabled={disabled || loading} onClick={() => { setCustom(false); void fetchModels(); }}><RefreshCw size={15} strokeWidth={2} className={loading ? "model-select-spinning" : ""} /></button>
    {loading && <span className="muted small">Loading…</span>}
    {fallback && !loading && <span className="muted small">{hint}</span>}
  </div>;
}
