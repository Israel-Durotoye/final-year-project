import { useEffect, useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { MapPreview } from "@/components/MapPreview";
import { TrendCharts } from "@/components/TrendCharts";
import {
  Radio, Leaf, FlaskConical, Droplets, CloudRain, Thermometer, Sprout,
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

  useEffect(() => {
    let mounted = true;
    const fetchRecent = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data, error: sbError } = await supabase
          .from("sensor_telemetry")
          .select("*")
          .order("timestamp_utc", { ascending: false })
          .limit(300);
        if (sbError) throw sbError;
        const arr = Array.isArray(data) ? data : [];
        if (!mounted) return;
        setRows(arr);

        // Reduce to latest per node_id
        const seen = new Map<string, any>();
        for (const r of arr) {
          const id = r.node_id;
          if (!seen.has(id)) seen.set(id, r);
        }
        const latest = Array.from(seen.values());
        setLatestNodes(latest);
        if (!selected && latest.length) setSelected(latest[0].node_id);
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

  const node = latestNodes.find((n) => n.node_id === selected) ?? latestNodes[0];

  return (
    <>
      <PageHeader title="Dashboard" subtitle="Real-time field intelligence across all nodes" />
      <div className="p-6 space-y-6">
        {/* Top metrics — computed from Supabase rows */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <MetricCard icon={Radio} label="Total Nodes" value={latestNodes.length} trend="Unique nodes" tone="info" />
          <MetricCard icon={Droplets} label="Avg Soil Moisture" value={rows.length ? `${(rows.reduce((s, r) => s + Number(r.soil_moisture || 0), 0) / rows.length).toFixed(1)}%` : "-"} trend="Network average" tone="success" />
          <MetricCard icon={Leaf} label="Avg pH" value={rows.length ? (rows.reduce((s, r) => s + Number(r.ph || 0), 0) / rows.length).toFixed(2) : "-"} trend="Network average" tone="info" />
        </div>

        {/* Node parameters */}
        <section className="bg-card border border-border rounded-xl shadow-card overflow-hidden">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-5 border-b border-border">
            <div>
              <h2 className="text-lg font-semibold">Node Parameters</h2>
              <p className="text-sm text-muted-foreground">Live readings · {node?.farm ?? "-"} — {node?.location ?? "-"}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {latestNodes.map((n) => (
                <button
                  key={n.node_id}
                  onClick={() => setSelected(n.node_id)}
                  className={cn(
                    "px-3 py-1.5 rounded-lg text-[10px] font-mono font-bold tracking-wider uppercase border transition-all",
                    selected === n.node_id
                      ? "bg-primary text-primary-foreground border-primary shadow-glow"
                      : "bg-secondary text-secondary-foreground border-white/5 hover:border-primary/50"
                  )}
                >
                  {n.node_id}
                </button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4 p-5">
            <MetricCard icon={Leaf} label="Nitrogen" value={node ? (node.nitrogen_ppm ?? node.nitrogen ?? "-") : "-"} unit="ppm" tone="success" />
            <MetricCard icon={FlaskConical} label="Phosphorus" value={node ? (node.phosphorus_ppm ?? node.phosphorus ?? "-") : "-"} unit="ppm" tone="info" />
            <MetricCard icon={Sprout} label="Potassium" value={node ? (node.potassium_ppm ?? node.potassium ?? "-") : "-"} unit="ppm" tone="success" />
            <MetricCard icon={Droplets} label="Soil Moisture" value={node ? (node.soil_moisture ?? node.moisture ?? "-") : "-"} unit="%" tone="info" />
            <MetricCard icon={CloudRain} label="Humidity" value={node ? (node.humidity ?? "-") : "-"} unit="%" tone="default" />
            <MetricCard icon={Thermometer} label="Temperature" value={node ? (node.soil_temperature_c ?? node.temperature ?? "-") : "-"} unit="°C" tone="warning" />
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
