import { PageHeader } from "@/components/layout/PageHeader";
import { Server, AlertTriangle, TrendingUp, Radio, Leaf, FlaskConical, Sprout, Droplets, CloudRain, Thermometer, Stethoscope, LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { createClient } from "@supabase/supabase-js";

type Tone = "good" | "fair" | "poor";

// Score thresholds for each metric (min-good, min-fair)
const ranges: Record<string, { good: [number, number]; fair: [number, number]; unit: string }> = {
  nitrogen:    { good: [25, 50], fair: [15, 60], unit: "ppm" },
  phosphorus:  { good: [40, 65], fair: [30, 75], unit: "ppm" },
  potassium:   { good: [180, 230], fair: [150, 260], unit: "ppm" },
  moisture:    { good: [35, 55], fair: [25, 65], unit: "%" },
  humidity:    { good: [60, 75], fair: [50, 80], unit: "%" },
  temperature: { good: [22, 26], fair: [18, 30], unit: "°C" },
};

const scoreMetric = (key: string, v: number): Tone => {
  const r = ranges[key];
  if (v >= r.good[0] && v <= r.good[1]) return "good";
  if (v >= r.fair[0] && v <= r.fair[1]) return "fair";
  return "poor";
};

const toneClasses: Record<Tone, { bar: string; text: string; badgeBg: string; badgeText: string; label: string }> = {
  good: { bar: "bg-primary", text: "text-primary", badgeBg: "bg-primary-soft", badgeText: "text-primary", label: "GOOD" },
  fair: { bar: "bg-warning", text: "text-warning", badgeBg: "bg-warning/15", badgeText: "text-warning", label: "FAIR" },
  poor: { bar: "bg-destructive", text: "text-destructive", badgeBg: "bg-destructive/15", badgeText: "text-destructive", label: "POOR" },
};

// Compute fill % within the fair envelope
const fillPct = (key: string, v: number) => {
  const [lo, hi] = ranges[key].fair;
  return Math.max(8, Math.min(100, ((v - lo) / (hi - lo)) * 100));
};

const StatCard = ({ icon: Icon, label, value, tone = "default" }: { icon: LucideIcon; label: string; value: string | number; tone?: "default" | "warning" | "primary" }) => {
  const toneStyle = tone === "warning"
    ? "bg-destructive/10 text-destructive"
    : tone === "primary"
    ? "bg-primary/20 text-primary"
    : "bg-secondary text-secondary-foreground";
  const valueColor = tone === "warning" ? "text-destructive" : tone === "primary" ? "text-primary" : "text-foreground";
  return (
    <div className="glass-card rounded-xl p-5 shadow-card hover:shadow-elevated transition-all">
      <div className="flex items-center gap-4">
        <div className={cn("h-12 w-12 rounded-xl flex items-center justify-center shrink-0 border border-white/5", toneStyle)}>
          <Icon className="h-6 w-6" />
        </div>
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-widest text-muted-foreground font-medium">{label}</p>
          <p className={cn("text-3xl font-mono font-bold tracking-tight mt-0.5", valueColor)}>{value}</p>
        </div>
      </div>
    </div>
  );
};

const MetricCell = ({ icon: Icon, label, value, mKey, val }: { icon: LucideIcon; label: string; value: number | string; mKey: string; val: number }) => {
  const tone = scoreMetric(mKey, val);
  const t = toneClasses[tone];
  return (
    <div className="space-y-1.5">
      <div className="flex items-end justify-between text-xs mb-1">
        <div className="flex items-center gap-1.5 pb-[2px]">
          <Icon className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          <span className="text-muted-foreground font-medium">{label}</span>
        </div>
        <div className="flex items-baseline justify-end">
          <span className={cn("font-mono font-bold text-sm tracking-tight text-right", t.text)}>
            {value}
          </span>
          <span className="text-muted-foreground font-mono text-[10px] w-7 ml-1 text-left">
            {ranges[mKey]?.unit ?? ""}
          </span>
        </div>
      </div>
      <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
        <div className={cn("h-full rounded-full transition-all", t.bar)} style={{ width: `${fillPct(mKey, val)}%` }} />
      </div>
    </div>
  );
};

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || (process.env.VITE_SUPABASE_URL as string) || "";
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || (process.env.VITE_SUPABASE_ANON_KEY as string) || "";
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

const Nodes = () => {
  const navigate = useNavigate();
  const [latestNodes, setLatestNodes] = useState<Array<any>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Helper to determine status
  const calculateNodeStatus = (node: any): string => {
    // OFFLINE if communication_ok is explicitly false
    if (node.communication_ok === false || node.communication_ok === 0 || node.communication_ok === "false") return "OFFLINE";
    const n = Number(node.nitrogen_ppm ?? node.nitrogen ?? 0);
    const moisture = Number(node.soil_moisture ?? node.moisture ?? 0);
    if (n < 20 || moisture < 25) return "POOR";
    if ((n >= 20 && n <= 30) || (moisture >= 25 && moisture <= 35)) return "FAIR";
    return "GOOD";
  };

  useEffect(() => {
    let mounted = true;
    const fetchLatest = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data, error: sbError } = await supabase
          .from("sensor_telemetry")
          .select("*")
          .order("timestamp_utc", { ascending: false })
          .limit(100);
        if (sbError) throw sbError;
        const rows = Array.isArray(data) ? data : [];
        // Reduce to latest row per node_id
        const reduced = rows.reduce((acc: Map<string, any>, row: any) => {
          const id = row.node_id;
          if (!acc.has(id)) acc.set(id, row);
          return acc;
        }, new Map<string, any>());
        const latest = Array.from(reduced.values());
        if (mounted) setLatestNodes(latest);
      } catch (err: any) {
        setError(err.message || String(err));
        setLatestNodes([]);
      } finally {
        setLoading(false);
      }
    };
    fetchLatest();
    return () => { mounted = false; };
  }, []);

  const totalNodes = latestNodes.length;
  const attention = latestNodes.filter((n) => calculateNodeStatus(n) !== "GOOD").length;
  const yieldForecast = "--"; // placeholder

  return (
    <>
      <PageHeader title="Nodes" subtitle={`${totalNodes} sensor nodes deployed`} />

      <div className="p-6 space-y-6">
        {/* Top stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <StatCard icon={Server} label="Total Nodes" value={totalNodes} />
          <StatCard icon={AlertTriangle} label="Nodes Requiring Attention" value={attention} tone="warning" />
          <StatCard icon={TrendingUp} label="Avg Farm Yield Forecast" value={`${yieldForecast}%`} tone="primary" />
        </div>

        {/* Node grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {latestNodes.map((n) => {
            const status = calculateNodeStatus(n);
            const toneKey: Tone = status === "GOOD" ? "good" : status === "FAIR" ? "fair" : "poor";
            const t = toneClasses[toneKey];
            const isOnline = !(n.communication_ok === false || n.communication_ok === 0 || n.communication_ok === "false");
            const badgeLabel = status;

            return (
              <div key={n.node_id} className="bg-card border border-border rounded-xl p-5 shadow-card hover:shadow-elevated hover:-translate-y-0.5 transition-all flex flex-col">
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className={cn("h-11 w-11 rounded-xl flex items-center justify-center", isOnline ? "bg-primary-soft text-primary" : "bg-muted text-muted-foreground")}>
                      <Radio className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="font-bold tracking-tight text-base">{n.node_id}</h3>
                      <p className="text-xs text-muted-foreground">{n.farm ?? "-"} · {n.location ?? "-"}</p>
                    </div>
                  </div>
                  <span className={cn("inline-flex items-center px-2.5 py-1 rounded-full text-[10px] font-bold tracking-wider", status === "OFFLINE" ? "bg-muted text-muted-foreground" : t.badgeBg, status === "OFFLINE" ? "text-muted-foreground" : t.badgeText)}>
                    {badgeLabel}
                  </span>
                </div>

                {/* Metrics list */}
                <div className="flex flex-col gap-y-5 mb-6">
                  <MetricCell icon={Leaf} label="Nitrogen" value={n.nitrogen_ppm ?? n.nitrogen ?? "-"} mKey="nitrogen" val={Number(n.nitrogen_ppm ?? n.nitrogen ?? 0)} />
                  <MetricCell icon={FlaskConical} label="Phosphorus" value={n.phosphorus_ppm ?? n.phosphorus ?? "-"} mKey="phosphorus" val={Number(n.phosphorus_ppm ?? n.phosphorus ?? 0)} />
                  <MetricCell icon={Sprout} label="Potassium" value={n.potassium_ppm ?? n.potassium ?? "-"} mKey="potassium" val={Number(n.potassium_ppm ?? n.potassium ?? 0)} />
                  <MetricCell icon={Droplets} label="Moisture" value={n.soil_moisture ?? n.moisture ?? "-"} mKey="moisture" val={Number(n.soil_moisture ?? n.moisture ?? 0)} />
                  <MetricCell icon={CloudRain} label="Humidity" value={n.humidity ?? "-"} mKey="humidity" val={Number(n.humidity ?? 0)} />
                  <MetricCell icon={Thermometer} label="Temp" value={n.soil_temperature_c ?? n.temperature ?? "-"} mKey="temperature" val={Number(n.soil_temperature_c ?? n.temperature ?? 0)} />
                </div>

                {/* CTA */}
                <Button onClick={() => navigate('/ai-doctor', { state: { autoQuery: `Run a full diagnostic on ${n.node_id} using its latest telemetry.` } })} className="w-full mt-auto gradient-primary text-primary-foreground hover:opacity-95 shadow-glow h-11 text-sm tracking-wide">
                  <Stethoscope className="h-4 w-4 mr-2" />
                  Diagnose Node
                </Button>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
};

export default Nodes;
