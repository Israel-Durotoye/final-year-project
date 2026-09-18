import { useEffect, useState } from "react";
import { Circle } from "react-leaflet";
import { Activity, Droplets, Leaf, Wifi } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { MapPreview } from "@/components/MapPreview";
import { cn } from "@/lib/utils";
import { METRICS, loadAlertThresholds } from "@/lib/alerting";
import {
  formatSpatialValue, getMapCoordinate, getNearestSensorReadings, getSpatialAssessment,
  getSpatialComparison, getSpatialLayerColor, getSpatialLayerRadiusMeters,
  spatialMetric, SPATIAL_COLORS, SpatialLayerType,
} from "@/lib/mapSpatial";
import { fetchTelemetry, HARDWARE_NODE_IDS, latestTelemetryByNode, SIMULATOR_NODE_IDS, TelemetryRow } from "@/lib/telemetry";
import { fetchCropRecommendations, formatCropRecommendation, CropRecommendation } from "@/lib/cropRecommendation";

const MAP_REFRESH_INTERVAL_MS = 60_000;
const layers = [
  { id: "coverage", label: "Coverage", icon: Wifi },
  ...METRICS.map((metric) => ({ id: metric.key, label: metric.label, icon: metric.key === "moisture" ? Droplets : Leaf })),
  { id: "health", label: "Condition check", icon: Activity },
] as const;

const MapView = () => {
  const [selected, setSelected] = useState<string>();
  const [nodesLatest, setNodesLatest] = useState<TelemetryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeLayer, setActiveLayer] = useState<SpatialLayerType>("moisture");
  const [thresholds, setThresholds] = useState(loadAlertThresholds);
  const [now, setNow] = useState(() => new Date());
  const [cropRecommendations, setCropRecommendations] = useState<Record<string, CropRecommendation>>({});

  useEffect(() => {
    const update = () => setThresholds(loadAlertThresholds());
    window.addEventListener("soilnet:thresholds-updated", update);
    window.addEventListener("storage", update);
    return () => {
      window.removeEventListener("soilnet:thresholds-updated", update);
      window.removeEventListener("storage", update);
    };
  }, []);

  useEffect(() => {
    if (!nodesLatest.length) return;
    let cancelled = false;
    void fetchCropRecommendations(nodesLatest.map((item) => item.Node_ID)).then((recommendations) => {
      if (!cancelled) setCropRecommendations(recommendations);
    });
    return () => { cancelled = true; };
  }, [nodesLatest]);

  useEffect(() => {
    let cancelled = false;
    const fetchLatestPerNode = async () => {
      try {
        const latest = latestTelemetryByNode(await fetchTelemetry({ limit: 1000 }));
        if (cancelled) return;
        setError(null);
        setNodesLatest(latest);
        setSelected((current) => latest.some((item) => item.Node_ID === current)
          ? current : (latest.find((item) => getMapCoordinate(item)) ?? latest[0])?.Node_ID);
      } catch (err: unknown) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) { setLoading(false); setNow(new Date()); }
      }
    };
    void fetchLatestPerNode();
    const timer = window.setInterval(() => { setNow(new Date()); void fetchLatestPerNode(); }, MAP_REFRESH_INTERVAL_MS);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, []);

  const node = nodesLatest.find((item) => item.Node_ID === selected);
  const knownNodeIds = [...HARDWARE_NODE_IDS, ...SIMULATOR_NODE_IDS];
  const missingNodeIds = knownNodeIds.filter((nodeId) => !nodesLatest.some((item) => item.Node_ID === nodeId));
  const metric = spatialMetric(activeLayer);
  const comparison = getSpatialComparison(activeLayer, nodesLatest, thresholds, now);
  const assessment = node ? getSpatialAssessment(activeLayer, node, thresholds, now) : null;
  const nearby = node ? getNearestSensorReadings(node, nodesLatest, activeLayer, thresholds, now) : [];
  const labels = Object.fromEntries(comparison.readings.map(({ node: item, assessment: result }) =>
    [String(item.Node_ID), `${result.reading}${result.value !== null ? ` · ${result.label}` : ""}`]));
  const colors = Object.fromEntries(comparison.readings.map(({ node: item }) =>
    [String(item.Node_ID), getSpatialLayerColor(activeLayer, item, nodesLatest, now, thresholds)]));
  const legend = activeLayer === "coverage"
    ? [{ color: SPATIAL_COLORS.within, label: "Active" }, { color: SPATIAL_COLORS.high, label: "Stale / offline" }]
    : activeLayer === "health"
      ? [{ color: SPATIAL_COLORS.within, label: "Within limits" }, { color: SPATIAL_COLORS.high, label: "Outside limits" }, { color: SPATIAL_COLORS.unknown, label: "Stale / incomplete" }]
      : [{ color: SPATIAL_COLORS.low, label: "Below limit" }, { color: SPATIAL_COLORS.within, label: "Within limits" }, { color: SPATIAL_COLORS.high, label: "Above limit" }, { color: SPATIAL_COLORS.unknown, label: "No current value" }];

  return (
    <>
      <PageHeader title="Map View" subtitle="See where conditions differ and what needs attention" />
      <div className="p-6 grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_360px] gap-6">
        <section className="min-w-0 space-y-4" aria-label="Sensor map and comparison">
          {error && <p role="alert" className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">Unable to refresh readings: {error}{nodesLatest.length ? ". Showing the last available readings." : ""}</p>}
          <div className="bg-card border border-border rounded-xl p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="font-semibold">{layers.find((layer) => layer.id === activeLayer)?.label ?? "Sensor locations"}</h2>
              {!loading && <span className="text-xs text-muted-foreground">{comparison.current} of {comparison.readings.length} mapped nodes with current {metric ? "values" : "readings"}</span>}
            </div>
            {metric && <p className="text-sm text-muted-foreground">Your alert range: <strong className="text-foreground">{formatSpatialValue(thresholds[metric.key].min, metric.unit)}–{formatSpatialValue(thresholds[metric.key].max, metric.unit)}</strong> · {comparison.low} below · {comparison.high} above</p>}
            {metric && comparison.minimum !== null && comparison.maximum !== null && <p className="text-sm">Measured range across current mapped nodes: {formatSpatialValue(comparison.minimum, metric.unit)}–{formatSpatialValue(comparison.maximum, metric.unit)}.</p>}
            {activeLayer === "health" && <p className="text-sm">{comparison.high} mapped location(s) have readings outside your alert limits.</p>}
            {activeLayer !== "none" && <div className="flex flex-wrap gap-x-4 gap-y-2" aria-label="Map legend">
              {legend.map((item) => <span key={item.label} className="inline-flex items-center gap-1.5 text-xs text-muted-foreground"><span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />{item.label}</span>)}
            </div>}
          </div>

          {loading ? <div role="status" className="h-96 flex items-center justify-center">Loading map…</div> : <MapPreview
            nodes={nodesLatest} selectedId={selected} onSelect={setSelected}
            height="h-[60vh] min-h-[420px]" nodeLabels={activeLayer === "none" ? undefined : labels}
            nodeColors={activeLayer === "none" ? undefined : colors} showNetworkBoundary={false}
          >
            {activeLayer !== "none" && comparison.readings.map(({ node: item }) => <Circle
              key={`circle-${item.Node_ID}`} center={getMapCoordinate(item)!}
              radius={getSpatialLayerRadiusMeters(activeLayer)}
              pathOptions={{ color: colors[String(item.Node_ID)], fillColor: colors[String(item.Node_ID)], fillOpacity: 0.13, weight: 1, interactive: false }}
            />)}
          </MapPreview>}
          <p className="text-xs text-muted-foreground">Hardware and simulated sensors are labelled below; colours show readings at each sensor, not measured conditions between sensors.</p>
          {missingNodeIds.length > 0 && (
            <p className="rounded-lg border border-warning/30 bg-warning/10 p-3 text-xs text-warning-foreground">
              Not reporting, so no marker is plotted: {missingNodeIds.join(", ")}. Check the hardware connection or wait for the simulator to send a reading.
            </p>
          )}
          {!loading && !comparison.readings.length && <p className="text-sm text-muted-foreground">No sensor locations are available to plot.</p>}
          {comparison.unmapped > 0 && <p className="text-xs text-muted-foreground">{comparison.unmapped} node(s) have no valid coordinates and are excluded from map comparisons.</p>}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3" aria-label="Compare sensor locations">
            {nodesLatest.map((item) => {
              const result = getSpatialAssessment(activeLayer, item, thresholds, now);
              return <button key={item.Node_ID} type="button" onClick={() => setSelected(item.Node_ID)} aria-pressed={selected === item.Node_ID}
                className={cn("text-left rounded-xl border bg-card p-4", selected === item.Node_ID ? "border-primary ring-1 ring-primary" : "border-border")}>
                <span className="flex justify-between gap-2"><strong className="text-sm">{item.Node_ID}</strong><span className="text-xs text-muted-foreground">{item.Data_Source === "hardware" ? "Hardware" : "Simulated"}</span></span>
                <span className="block text-lg font-semibold mt-1">{result.reading}</span>
                <span className="text-xs" style={{ color: result.color }}>{result.label}</span>
                {!getMapCoordinate(item) && <span className="block text-xs text-muted-foreground">Location unavailable</span>}
              </button>;
            })}
          </div>
        </section>

        <div className="space-y-5">
          <aside className="bg-card border border-border rounded-xl p-5">
            <h2 className="font-semibold">Spatial Analysis</h2>
            <p className="text-xs text-muted-foreground mt-1 mb-4">Choose a measurement to compare locations.</p>
            <div className="grid grid-cols-2 gap-2">
              {layers.map(({ id, label, icon: Icon }) => <button key={id} type="button" onClick={() => setActiveLayer(id)} aria-pressed={activeLayer === id}
                className={cn("flex items-center gap-2 p-2.5 rounded-lg border text-xs font-medium text-left", activeLayer === id ? "bg-primary text-primary-foreground border-primary" : "bg-secondary/30 border-border hover:bg-secondary")}>
                <Icon className="h-4 w-4 shrink-0" />{label}
              </button>)}
            </div>
            <button type="button" onClick={() => setActiveLayer("none")} aria-pressed={activeLayer === "none"} className="mt-3 text-xs text-muted-foreground underline underline-offset-4">Clear overlays</button>
            <p className="text-xs text-muted-foreground mt-4">Comparisons use your alert limits in Settings; these are screening limits, not a crop diagnosis.</p>
          </aside>

          <aside className="bg-card border border-border rounded-xl p-5" aria-label="Selected sensor interpretation">
            <h2 className="font-semibold">What this location means</h2>
            {node && assessment ? <div className="mt-4 space-y-4">
              <div><h3 className="text-xl font-semibold">{node.Node_ID}</h3><p className="text-xs text-muted-foreground">Recorded: {String(node.Target_Crop || "none")} · Best crop: {formatCropRecommendation(cropRecommendations[node.Node_ID])} · {node.Data_Source === "hardware" ? "Hardware sensor" : "Simulated sensor"}</p></div>
              <div className="rounded-lg border border-border p-3 space-y-2" style={{ borderLeftColor: assessment.color, borderLeftWidth: 3 }}>
                <p className="text-sm font-semibold">{metric?.label ?? (activeLayer === "health" ? "Condition check" : "Connection")}: {assessment.label}</p>
                <p className="text-sm leading-relaxed">{assessment.explanation}</p>
                <p className="text-sm text-muted-foreground leading-relaxed">{assessment.action}</p>
              </div>
              <div className="grid grid-cols-2 gap-2" aria-label="Selected sensor measurements">
                {METRICS.map((entry) => {
                  const result = getSpatialAssessment(entry.key, node, thresholds, now);
                  return <button key={entry.key} type="button" onClick={() => setActiveLayer(entry.key)} aria-label={`Inspect ${entry.label}`} className={cn("text-left rounded-lg border p-2.5", activeLayer === entry.key ? "border-primary bg-primary/5" : "border-border")}>
                    <span className="block text-xs text-muted-foreground">{entry.label}</span><span className="block text-sm font-semibold mt-1">{result.reading}</span><span className="block text-[11px] mt-1" style={{ color: result.color }}>{result.label}</span>
                  </button>;
                })}
              </div>
              {metric && <div className="border-t border-border pt-4 space-y-2">
                <h3 className="text-sm font-semibold">Nearest sensor comparisons</h3>
                <p className="text-xs text-muted-foreground">Current {node.Data_Source === "hardware" ? "hardware" : "simulated"} readings, ordered by distance.</p>
                {nearby.length ? nearby.map((other) => <button type="button" key={String(other.node.Node_ID)} onClick={() => setSelected(String(other.node.Node_ID))} className="block w-full text-left rounded-lg bg-secondary/40 p-3 text-xs">
                  <strong>{String(other.node.Node_ID)}</strong> · {other.distance < 1000 ? `${Math.round(other.distance)} m` : `${(other.distance / 1000).toFixed(1)} km`} away
                  <span className="block mt-1">{other.assessment.reading} · {other.assessment.label}</span>
                  <span className="block mt-1 text-muted-foreground">{other.assessment.value === assessment.value ? "Same reading as this location" : `${Number(Math.abs(assessment.value! - other.assessment.value!).toFixed(1))} ${metric.unit === "%" ? "percentage points" : metric.unit} ${other.assessment.value! > assessment.value! ? "higher" : "lower"} than this location`}</span>
                </button>) : <p className="text-xs text-muted-foreground">No other current, located sensors of this type are available for comparison.</p>}
              </div>}
            </div> : <p className="mt-3 text-sm text-muted-foreground">Select a sensor marker or location card to interpret its readings.</p>}
          </aside>
        </div>
      </div>
    </>
  );
};

export default MapView;
