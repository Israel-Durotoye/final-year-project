import { PageHeader } from "@/components/layout/PageHeader";
import { nodes, SensorNode } from "@/lib/mockData";
import { Server, AlertTriangle, TrendingUp, Radio, Leaf, FlaskConical, Sprout, Droplets, CloudRain, Thermometer, Stethoscope, LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { evaluateTelemetry } from "@/lib/api";

type Tone = "good" | "fair" | "poor";

// Score thresholds for each metric (min-good, min-fair)
const ranges: Record<keyof SensorNode["metrics"], { good: [number, number]; fair: [number, number]; unit: string }> = {
  nitrogen:    { good: [25, 50], fair: [15, 60], unit: "ppm" },
  phosphorus:  { good: [40, 65], fair: [30, 75], unit: "ppm" },
  potassium:   { good: [180, 230], fair: [150, 260], unit: "ppm" },
  moisture:    { good: [35, 55], fair: [25, 65], unit: "%" },
  humidity:    { good: [60, 75], fair: [50, 80], unit: "%" },
  temperature: { good: [22, 26], fair: [18, 30], unit: "°C" },
};

const scoreMetric = (key: keyof SensorNode["metrics"], v: number): Tone => {
  const r = ranges[key];
  if (v >= r.good[0] && v <= r.good[1]) return "good";
  if (v >= r.fair[0] && v <= r.fair[1]) return "fair";
  return "poor";
};

const nodeStatus = (n: SensorNode): Tone => {
  const scores = (Object.keys(n.metrics) as Array<keyof SensorNode["metrics"]>).map((k) => scoreMetric(k, n.metrics[k]));
  if (scores.includes("poor")) return "poor";
  if (scores.includes("fair")) return "fair";
  return "good";
};

const toneClasses: Record<Tone, { bar: string; text: string; badgeBg: string; badgeText: string; label: string }> = {
  good: { bar: "bg-primary", text: "text-primary", badgeBg: "bg-primary-soft", badgeText: "text-primary", label: "GOOD" },
  fair: { bar: "bg-warning", text: "text-warning", badgeBg: "bg-warning/15", badgeText: "text-warning", label: "FAIR" },
  poor: { bar: "bg-destructive", text: "text-destructive", badgeBg: "bg-destructive/15", badgeText: "text-destructive", label: "POOR" },
};

// Compute fill % within the fair envelope
const fillPct = (key: keyof SensorNode["metrics"], v: number) => {
  const [lo, hi] = ranges[key].fair;
  return Math.max(8, Math.min(100, ((v - lo) / (hi - lo)) * 100));
};

const StatCard = ({ icon: Icon, label, value, tone = "default" }: { icon: LucideIcon; label: string; value: string | number; tone?: "default" | "warning" | "primary" }) => {
  const toneStyle = tone === "warning"
    ? "bg-destructive/10 text-destructive"
    : tone === "primary"
    ? "bg-primary-soft text-primary"
    : "bg-secondary text-secondary-foreground";
  const valueColor = tone === "warning" ? "text-destructive" : tone === "primary" ? "text-primary" : "text-foreground";
  return (
    <div className="bg-card border border-border rounded-xl p-5 shadow-card hover:shadow-elevated transition-all">
      <div className="flex items-center gap-4">
        <div className={cn("h-12 w-12 rounded-xl flex items-center justify-center shrink-0", toneStyle)}>
          <Icon className="h-6 w-6" />
        </div>
        <div className="min-w-0">
          <p className="text-sm text-muted-foreground font-medium">{label}</p>
          <p className={cn("text-3xl font-bold tracking-tight mt-0.5", valueColor)}>{value}</p>
        </div>
      </div>
    </div>
  );
};

const MetricCell = ({ icon: Icon, label, value, mKey, val }: { icon: LucideIcon; label: string; value: number | string; mKey: keyof SensorNode["metrics"]; val: number }) => {
  const tone = scoreMetric(mKey, val);
  const t = toneClasses[tone];
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5 text-xs">
        <Icon className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        <span className="text-muted-foreground font-medium">{label}</span>
        <span className={cn("ml-auto font-bold", t.text)}>
          {value} <span className="text-muted-foreground font-medium text-[11px]">{ranges[mKey].unit}</span>
        </span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
        <div className={cn("h-full rounded-full transition-all", t.bar)} style={{ width: `${fillPct(mKey, val)}%` }} />
      </div>
    </div>
  );
};

const Nodes = () => {
  const navigate = useNavigate();
  const [evaluations, setEvaluations] = useState<Record<string, any>>({});

  const totalNodes = nodes.length;
  const attention = nodes.filter((n) => n.status === "offline" || nodeStatus(n) !== "good").length;
  const goodCount = nodes.filter((n) => n.status === "online" && nodeStatus(n) === "good").length;
  const yieldForecast = Math.round(50 + (goodCount / totalNodes) * 45);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const evals: Record<string, any> = {};
        await Promise.all(nodes.map(async (n) => {
          const payload: Record<string, any> = {
            nitrogen_ppm: n.metrics.nitrogen,
            phosphorus_ppm: n.metrics.phosphorus,
            potassium_ppm: n.metrics.potassium,
            soil_moisture: n.metrics.moisture,
            ambient_humidity: n.metrics.humidity,
            ambient_temperature: n.metrics.temperature,
          };
          try {
            const res = await evaluateTelemetry(payload);
            evals[n.id] = res;
          } catch (err) {
            console.error("Evaluate error for", n.id, err);
          }
        }));
        if (mounted) setEvaluations(evals);
      } catch (e) {
        console.error("Failed to load evaluations", e);
      }
    })();
    return () => { mounted = false; };
  }, []);

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
          {nodes.map((n) => {
            const status = n.status === "offline" ? "poor" : nodeStatus(n);
            const t = toneClasses[status];
            const remote = evaluations[n.id];
            const healthBand = remote?.summary?.health_band ?? t.label;

            return (
              <div key={n.id} className="bg-card border border-border rounded-xl p-5 shadow-card hover:shadow-elevated hover:-translate-y-0.5 transition-all flex flex-col">
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className={cn("h-11 w-11 rounded-xl flex items-center justify-center", n.status === "online" ? "bg-primary-soft text-primary" : "bg-muted text-muted-foreground")}>
                      <Radio className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="font-bold tracking-tight text-base">{n.id}</h3>
                      <p className="text-xs text-muted-foreground">{n.farm} · {n.location}</p>
                    </div>
                  </div>
                  <span className={cn("inline-flex items-center px-2.5 py-1 rounded-full text-[10px] font-bold tracking-wider", t.badgeBg, t.badgeText)}>
                    {n.status === "offline" ? "OFFLINE" : healthBand}
                  </span>
                </div>

                {/* Metrics grid */}
                <div className="grid grid-cols-3 gap-x-4 gap-y-3 mb-4">
                  <MetricCell icon={Leaf} label="N" value={n.metrics.nitrogen} mKey="nitrogen" val={n.metrics.nitrogen} />
                  <MetricCell icon={FlaskConical} label="P" value={n.metrics.phosphorus} mKey="phosphorus" val={n.metrics.phosphorus} />
                  <MetricCell icon={Sprout} label="K" value={n.metrics.potassium} mKey="potassium" val={n.metrics.potassium} />
                  <MetricCell icon={Droplets} label="Moisture" value={n.metrics.moisture} mKey="moisture" val={n.metrics.moisture} />
                  <MetricCell icon={CloudRain} label="Humidity" value={n.metrics.humidity} mKey="humidity" val={n.metrics.humidity} />
                  <MetricCell icon={Thermometer} label="Temp" value={n.metrics.temperature} mKey="temperature" val={n.metrics.temperature} />
                </div>

                {/* CTA */}
                <Button onClick={() => { window.dispatchEvent(new CustomEvent("ai:diagnose", { detail: { nodeId: n.id } })); navigate("/ai-doctor"); }} className="w-full mt-auto gradient-primary text-primary-foreground hover:opacity-95 shadow-glow">
                  <Stethoscope className="h-4 w-4 mr-2" />
                  Diagnose
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
