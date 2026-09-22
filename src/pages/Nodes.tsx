import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, CloudRain, Droplets, FlaskConical, Leaf, Radio, RefreshCw, Server, Sprout, Stethoscope, Thermometer } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { getNigerianSeason } from "@/lib/season";
import { METRICS, evaluateNodeThresholds, loadAlertThresholds } from "@/lib/alerting";
import { getSpatialAssessment, isMapNodeOnline } from "@/lib/mapSpatial";
import { fetchTelemetry, latestTelemetryByNode, TelemetryRow } from "@/lib/telemetry";
import { fetchCropRecommendations, isTentativeCropRecommendation, CropRecommendation } from "@/lib/cropRecommendation";

const SENSOR_ROWS = [
  { key: "nitrogen", label: "Nitrogen", icon: Leaf, max: 1999 },
  { key: "phosphorus", label: "Phosphorus", icon: FlaskConical, max: 1999 },
  { key: "potassium", label: "Potassium", icon: Sprout, max: 1999 },
  { key: "moisture", label: "Moisture", icon: Droplets, max: 100 },
  { key: "humidity", label: "Humidity", icon: CloudRain, max: 100 },
  { key: "temperature", label: "Temperature", icon: Thermometer, max: 100 },
] as const;

const LEVEL_STYLES = {
  good: { text: "text-green-600 dark:text-green-400", bar: "bg-green-500", badge: "bg-green-500/10 text-green-700 dark:text-green-400" },
  fair: { text: "text-orange-600 dark:text-orange-400", bar: "bg-orange-500", badge: "bg-orange-500/10 text-orange-700 dark:text-orange-400" },
  poor: { text: "text-red-600 dark:text-red-400", bar: "bg-red-500", badge: "bg-red-500/10 text-red-700 dark:text-red-400" },
  unknown: { text: "text-muted-foreground", bar: "bg-muted-foreground/35", badge: "bg-muted text-muted-foreground" },
};

const readingAge = (timestamp: string, now: Date) => {
  const age = Math.floor((now.getTime() - Date.parse(timestamp)) / 60_000);
  if (!Number.isFinite(age) || age < 0) return "Reading time unavailable";
  if (age < 1) return "Updated just now";
  if (age < 60) return `Updated ${age} min ago`;
  if (age < 1440) return `Updated ${Math.floor(age / 60)}h ago`;
  return `Last reading ${new Date(timestamp).toLocaleDateString(undefined, { day: "numeric", month: "short" })}`;
};

const measurement = (value: unknown) => {
  if ((typeof value !== "number" && typeof value !== "string") || String(value).trim() === "") return "—";
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Number(parsed.toFixed(1)).toLocaleString() : "—";
};

const Nodes = () => {
  const [nodes, setNodes] = useState<TelemetryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => new Date());
  const [thresholds, setThresholds] = useState(loadAlertThresholds);
  const [recommendations, setRecommendations] = useState<Record<string, CropRecommendation>>({});
  const [predicting, setPredicting] = useState<Record<string, boolean>>({});
  const mounted = useRef(false);
  const pendingPredictions = useRef(new Set<string>());
  const predictedTimestamps = useRef(new Map<string, string>());

  useEffect(() => {
    mounted.current = true;
    let inFlight = false;
    const refresh = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const latest = latestTelemetryByNode(await fetchTelemetry({ limit: 200 }));
        if (mounted.current) { setNodes(latest); setError(null); }
      } catch (err) {
        if (mounted.current) setError(err instanceof Error ? err.message : "Please try again shortly.");
      } finally {
        inFlight = false;
        if (mounted.current) { setLoading(false); setNow(new Date()); }
      }
    };
    void refresh();
    const timer = window.setInterval(() => { setNow(new Date()); void refresh(); }, 30_000);
    const updateLimits = () => setThresholds(loadAlertThresholds());
    window.addEventListener("soilnet:thresholds-updated", updateLimits);
    window.addEventListener("storage", updateLimits);
    return () => {
      mounted.current = false;
      window.clearInterval(timer);
      window.removeEventListener("soilnet:thresholds-updated", updateLimits);
      window.removeEventListener("storage", updateLimits);
    };
  }, []);

  const predictCrop = async (nodeId: string) => {
    if (pendingPredictions.current.has(nodeId)) return;
    pendingPredictions.current.add(nodeId);
    setPredicting((current) => ({ ...current, [nodeId]: true }));
    try {
      const results = await fetchCropRecommendations([nodeId]);
      if (mounted.current) setRecommendations((current) => ({ ...current, ...results }));
    } finally {
      pendingPredictions.current.delete(nodeId);
      if (mounted.current) setPredicting((current) => ({ ...current, [nodeId]: false }));
    }
  };

  useEffect(() => {
    for (const node of nodes) {
      if (predictedTimestamps.current.get(node.Node_ID) === node.Timestamp || pendingPredictions.current.has(node.Node_ID)) continue;
      predictedTimestamps.current.set(node.Node_ID, node.Timestamp);
      void predictCrop(node.Node_ID);
    }
  }, [nodes]);

  const assessed = nodes.map((node) => ({ node, assessment: getSpatialAssessment("health", node, thresholds, now), online: isMapNodeOnline(node, now) }));
  const activeCount = assessed.filter((item) => item.online).length;
  const attentionCount = assessed.filter((item) => item.assessment.status !== "within").length;
  return (
    <>
      <PageHeader title="Nodes" subtitle={`${nodes.length} sensors on your farm`} />
      <div className="space-y-6 py-5 sm:px-1">
        {error && <p role="alert" className="rounded-lg border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">Unable to refresh readings. {nodes.length ? "Showing the last available readings. " : ""}{error}</p>}

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3" aria-label="Network overview">
          {[
            { label: "Active Nodes", value: activeCount, icon: Radio, tone: "text-primary bg-primary/10" },
            { label: "Needs Attention", value: attentionCount, icon: AlertTriangle, tone: "text-warning bg-warning/10" },
            { label: "Total Nodes", value: nodes.length, icon: Server, tone: "text-primary bg-primary/10" },
          ].map(({ label, value, icon: Icon, tone }) => <div key={label} className="flex items-center gap-4 rounded-xl border border-border bg-card p-5 shadow-card">
            <div className={cn("flex h-11 w-11 shrink-0 items-center justify-center rounded-xl", tone)}><Icon className="h-5 w-5" /></div>
            <div><p className="text-xs font-medium text-muted-foreground">{label}</p><p className="mt-1 text-3xl font-semibold tabular-nums tracking-tight">{loading ? "—" : value}</p></div>
          </div>)}
        </div>

        {loading ? <div role="status" className="rounded-xl border border-border bg-card p-10 text-center text-sm text-muted-foreground">Loading sensor nodes…</div>
          : nodes.length === 0 ? <div className="rounded-xl border border-dashed border-border bg-card p-10 text-center text-sm text-muted-foreground">No sensor nodes are reporting yet.</div>
          : <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
            {assessed.map(({ node, assessment, online }) => {
              const recommendation = recommendations[node.Node_ID];
              const busy = predicting[node.Node_ID];
              const tentative = isTentativeCropRecommendation(recommendation);
              const recordedCrop = typeof node.Target_Crop === "string" ? node.Target_Crop.trim() : "";
              // Reuse the Alerts page's warning/critical bands, considering only valid, current readings.
              const alerts = evaluateNodeThresholds(node, thresholds).filter((alert) => {
                const result = getSpatialAssessment(alert.metric, node, thresholds, now);
                return result.status === "low" || result.status === "high";
              });
              const level = alerts.some((alert) => alert.severity === "critical") ? "poor"
                : alerts.length ? "fair" : assessment.status === "within" ? "good" : "unknown";
              const status = !online ? "OFFLINE" : level === "unknown" ? "INCOMPLETE" : level.toUpperCase();
              const tone = LEVEL_STYLES[level].badge;
              return <article key={node.Node_ID} aria-label={node.Node_ID} className="flex min-w-0 flex-col rounded-xl border border-border bg-card p-5 shadow-card transition-shadow hover:shadow-elevated">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-3">
                    <div className={cn("flex h-10 w-10 shrink-0 items-center justify-center rounded-xl", online ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground")}><Radio className="h-5 w-5" /></div>
                    <div className="min-w-0">
                      <h3 className="text-base font-semibold tracking-tight">{node.Node_ID}</h3>
                      <p className="mt-0.5 text-[11px] leading-relaxed text-muted-foreground">{node.Data_Source === "hardware" ? "Hardware" : "Simulator"} · {getNigerianSeason(node.Timestamp)}</p>
                    </div>
                  </div>
                  <span title={assessment.explanation} className={cn("shrink-0 rounded-full px-2 py-1 text-[10px] font-semibold tracking-wide", tone)}>{status}</span>
                </div>

                <div className="mb-5 mt-4 border-y border-border/70 py-3" aria-label={`Crop recommendation for ${node.Node_ID}`}>
                  {recordedCrop && <p className="mb-1.5 text-xs text-muted-foreground">Recorded planting <span className="ml-1 font-medium text-foreground">{recordedCrop}</span></p>}
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-1 text-xs" aria-live="polite">
                      <span className="text-muted-foreground">Suggested crop</span>
                      <span className="font-semibold text-foreground">{recommendation?.crop || (busy || !recommendation ? "Checking…" : recommendation.status === "insufficient_data" ? "More readings needed" : "Unavailable")}</span>
                      {tentative && <span className="text-[10px] text-warning">Tentative match</span>}
                    </div>
                    <button type="button" onClick={() => void predictCrop(node.Node_ID)} disabled={busy} aria-label={`Refresh crop prediction for ${node.Node_ID}`} title="Refresh crop prediction" className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-primary/10 hover:text-primary disabled:opacity-50"><RefreshCw className={cn("h-3.5 w-3.5", busy && "animate-spin")} /></button>
                  </div>
                  {recommendation?.crop ? <details className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                    <summary className="w-fit cursor-pointer rounded-sm hover:text-foreground">Prediction details</summary>
                    <div className="space-y-1 pt-2">
                      <p>{`${recommendation.readingsUsed ?? 24} recent readings${recommendation.confidence != null ? ` · ${Math.round(recommendation.confidence * 100)}% model score` : ""}`}</p>
                      {!!recommendation.imputedValues && <p>Some missing or out-of-range sensor values were estimated.</p>}
                      {tentative && <p>Collect more valid readings before choosing what to plant.</p>}
                      {!online && <p>Based on older readings; refresh once the sensor reconnects.</p>}
                      <p>This is a suggested planting, not an identified crop.</p>
                    </div>
                  </details> : recommendation?.status === "insufficient_data" ? <p className="mt-1 text-[11px] text-muted-foreground">{recommendation.readingsUsed ?? 0} of {recommendation.readingsRequired ?? 24} readings collected.</p> : null}
                </div>

                <dl className="mb-5 space-y-4" aria-label={`Sensor readings for ${node.Node_ID}`}>
                  {SENSOR_ROWS.map(({ key, label, icon: Icon, max }) => {
                    const metric = METRICS.find((item) => item.key === key)!;
                    const result = getSpatialAssessment(key, node, thresholds, now);
                    const value = measurement(node[metric.column]);
                    const numeric = value === "—" ? null : Number(node[metric.column]);
                    const fill = numeric === null ? 0 : Math.max(0, Math.min(100, numeric / max * 100));
                    const alert = alerts.find((item) => item.metric === key);
                    const readingLevel = !online || numeric === null || result.status === "missing" ? "unknown"
                      : alert?.severity === "critical" ? "poor" : alert ? "fair" : "good";
                    const colors = LEVEL_STYLES[readingLevel];
                    return <div key={key}>
                      <div className="mb-2 flex items-center justify-between gap-3">
                        <dt className="flex items-center gap-2 text-xs font-medium text-muted-foreground"><Icon className="h-3.5 w-3.5 shrink-0" />{label}</dt>
                        <dd className="flex shrink-0 items-baseline gap-1.5"><span className={cn("font-mono text-sm font-semibold tabular-nums", colors.text)}>{value}</span><span className="w-9 text-[10px] text-muted-foreground">{metric.unit}</span></dd>
                      </div>
                      <dd>
                        <div role="meter" aria-label={`${label} at ${node.Node_ID}`} aria-valuemin={0} aria-valuemax={max} aria-valuenow={numeric === null ? undefined : Math.max(0, Math.min(max, numeric))} aria-valuetext={numeric === null ? "No recorded value" : `${value} ${metric.unit} · ${online ? `${readingLevel.toUpperCase()} · ${result.label}` : "Last reading; sensor offline"}`} className="h-1.5 overflow-hidden rounded-full bg-muted" title={online ? `${readingLevel.toUpperCase()}: ${result.explanation}` : "Last recorded value; sensor offline"}>
                          <div className={cn("h-full rounded-full transition-all", colors.bar)} style={{ width: `${fill}%` }} />
                        </div>
                        <div aria-hidden="true" className="mt-1 flex justify-between font-mono text-[9px] text-muted-foreground/70"><span>0</span><span>{max} {metric.unit}</span></div>
                      </dd>
                    </div>;
                  })}
                </dl>

                <div className="mt-auto space-y-3 pt-1">
                  <p className="text-[11px] text-muted-foreground" title={node.Timestamp}>{readingAge(node.Timestamp, now)}</p>
                  <Button asChild className="h-10 w-full rounded-lg gradient-primary text-primary-foreground hover:opacity-95"><Link to="/ai-doctor"><Stethoscope className="mr-2 h-4 w-4" />Diagnose Node</Link></Button>
                </div>
              </article>;
            })}
          </div>}
      </div>
    </>
  );
};

export default Nodes;
