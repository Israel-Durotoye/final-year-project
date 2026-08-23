import { useEffect, useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { MapPreview } from "@/components/MapPreview";
import { TrendCharts } from "@/components/TrendCharts";
import {
  Radio, Leaf, FlaskConical, Droplets, CloudRain, Thermometer, Sprout, MapPin, Satellite, Bot
} from "lucide-react";
import { cn } from "@/lib/utils";
import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || (process.env.VITE_SUPABASE_URL as string) || "";
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || (process.env.VITE_SUPABASE_ANON_KEY as string) || "";
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

const Dashboard = () => {
  const [rows, setRows] = useState<Array<any>>([]);
  const [latestNodes, setLatestNodes] = useState<Array<any>>([]);
  const [selected, setSelected] = useState<string | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [statusLoading, setStatusLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    const fetchRecent = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data, error: sbError } = await supabase
          .from("capstone_dataset")
          .select("*")
          .order("Timestamp", { ascending: false })
          .limit(300);
        if (sbError) throw sbError;
        const arr = Array.isArray(data) ? data : [];
        if (!mounted) return;
        setRows(arr);

        // Reduce to latest per node_id
        const seen = new Map<string, any>();
        for (const r of arr) {
          const id = r.Node_ID;
          if (!seen.has(id)) seen.set(id, r);
        }
        const latest = Array.from(seen.values());
        setLatestNodes(latest);
        if (!selected && latest.length) setSelected(latest[0].Node_ID);
      } catch (err: any) {
        setError(err.message || String(err));
        setRows([]);
        setLatestNodes([]);
      } finally {
        setLoading(false);
      }
    };
    fetchRecent();
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    if (!selected || !latestNodes.length) return;
    const nodeData = latestNodes.find(n => n.Node_ID === selected);
    if (!nodeData) return;

    let mounted = true;
    const fetchStatus = async () => {
      setStatusLoading(true);
      setStatusMessage(null);
      try {
        const telemetryContext = `Telemetry: Nitrogen=${nodeData.Nitrogen_mg_k}, Phosphorus=${nodeData.Phosphorus_m}, Potassium=${nodeData.Potassium_mg_}, Moisture=${nodeData["Moisture_%"]}%, Temp=${nodeData.Temperature_C}°C`;
        const query = `Provide a single, clean sentence showing the status and any needed action for node ${selected}. ${telemetryContext}`;
        
        const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1";
        const res = await fetch(`${API_BASE}/chat/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, top_k: 1 })
        });
        
        if (!res.ok) throw new Error("Failed to fetch status");
        const data = await res.json();
        
        if (mounted) {
          // Clean up formatting to ensure it's a single clean sentence
          let msg = data.answer.replace(/\*\*[^*]+\*\*/g, '').replace(/\n/g, ' ').trim();
          setStatusMessage(msg);
        }
      } catch (err) {
        if (mounted) setStatusMessage("Status currently unavailable.");
      } finally {
        if (mounted) setStatusLoading(false);
      }
    };
    
    fetchStatus();
    
    return () => { mounted = false; };
  }, [selected, latestNodes]);

  const node = latestNodes.find((n) => n.Node_ID === selected) ?? latestNodes[0];

  return (
    <>
      <PageHeader title="Dashboard" subtitle="Real-time field intelligence across all nodes" />
      <div className="p-6 space-y-6">
        {/* Top metrics — computed from Supabase rows */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <MetricCard icon={Radio} label="Total Nodes" value={latestNodes.length} trend="Unique nodes" tone="info" />
          <MetricCard icon={Droplets} label="Avg Soil Moisture" value={rows.length ? `${(rows.reduce((s, r) => s + Number(r["Moisture_%"] || 0), 0) / rows.length).toFixed(1)}%` : "-"} trend="Network average" tone="success" />
          <MetricCard icon={Thermometer} label="Avg Temp" value={rows.length ? `${(rows.reduce((s, r) => s + Number(r.Temperature_C || 0), 0) / rows.length).toFixed(1)}°C` : "-"} trend="Network average" tone="warning" />
        </div>

        {/* AI Status Panel */}
        <section className="bg-card border border-border rounded-xl shadow-card overflow-hidden">
          <div className="p-5 flex items-start gap-4">
            <div className="bg-primary/10 p-3 rounded-full text-primary mt-1">
              <Bot className="h-6 w-6" />
            </div>
            <div className="flex-1">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                AI Node Status
                {selected && (
                  <span className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded-full font-mono">{selected}</span>
                )}
              </h2>
              <div className="mt-2 text-muted-foreground text-sm leading-relaxed">
                {statusLoading ? (
                  <span className="animate-pulse">Analyzing latest telemetry...</span>
                ) : (
                  <span>{statusMessage || "Select a node to view its AI-interpreted status."}</span>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Node parameters */}
        <section className="bg-card border border-border rounded-xl shadow-card overflow-hidden">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-5 border-b border-border">
            <div>
              <h2 className="text-lg font-semibold">Node Parameters</h2>
              <p className="text-sm text-muted-foreground">Live readings · {node?.Target_Crop ?? "-"} — {node?.Season ?? "-"}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {latestNodes.map((n) => (
                <button
                  key={n.Node_ID}
                  onClick={() => setSelected(n.Node_ID)}
                  className={cn(
                    "px-3 py-1.5 rounded-lg text-[10px] font-mono font-bold tracking-wider uppercase border transition-all",
                    selected === n.Node_ID
                      ? "bg-primary text-primary-foreground border-primary shadow-glow"
                      : "bg-secondary text-secondary-foreground border-white/5 hover:border-primary/50"
                  )}
                >
                  {n.Node_ID}
                </button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-4 gap-4 p-5">
            <MetricCard icon={Leaf} label="Nitrogen" value={node ? (node.Nitrogen_mg_k ?? "-") : "-"} unit="mg/kg" tone="success" />
            <MetricCard icon={FlaskConical} label="Phosphorus" value={node ? (node.Phosphorus_m ?? "-") : "-"} unit="mg/kg" tone="info" />
            <MetricCard icon={Sprout} label="Potassium" value={node ? (node.Potassium_mg_ ?? "-") : "-"} unit="mg/kg" tone="success" />
            <MetricCard icon={Droplets} label="Soil Moisture" value={node ? (node["Moisture_%"] ?? "-") : "-"} unit="%" tone="info" />
            <MetricCard icon={CloudRain} label="Humidity" value={node ? (node["Humidity_%"] ?? "-") : "-"} unit="%" tone="default" />
            <MetricCard icon={Thermometer} label="Temperature" value={node ? (node.Temperature_C ?? "-") : "-"} unit="°C" tone="warning" />
            <MetricCard icon={MapPin} label="Altitude" value={node ? (node.Altitude_m ?? "-") : "-"} unit="m" tone="default" />
            <MetricCard icon={Satellite} label="Satellites" value={node ? (node.Satellites ?? "-") : "-"} unit="" tone="info" />
          </div>
        </section>

        {/* Historical Trends */}
        <section>
          <div className="mb-4">
            <h2 className="text-lg font-semibold">Historical Trends</h2>
            <p className="text-sm text-muted-foreground">Recent telemetry variations for {selected}</p>
          </div>
          <TrendCharts rows={rows} selectedId={selected} />
        </section>

        {/* Map */}
        <section className="bg-card border border-border rounded-xl shadow-card p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold">Field Map</h2>
              <p className="text-sm text-muted-foreground">Live node locations — click a marker to inspect</p>
            </div>
          </div>
          {loading ? (
            <div className="h-96 flex items-center justify-center">Loading map…</div>
          ) : latestNodes.length === 0 ? (
            <div className="h-96 flex items-center justify-center text-muted-foreground">No location data available.</div>
          ) : (
            <MapPreview nodes={latestNodes} selectedId={selected} onSelect={setSelected} height="h-96" />
          )}
        </section>
      </div>
    </>
  );
};

export default Dashboard;
