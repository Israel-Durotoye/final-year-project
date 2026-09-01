import { PageHeader } from "@/components/layout/PageHeader";
import { getNigerianSeason } from "@/lib/season";
import { useEffect, useState } from "react";
import { createClient } from "@supabase/supabase-js";
import { MapPreview } from "@/components/MapPreview";
import { Circle } from "react-leaflet";
import { Droplets, Leaf, Wifi, Activity } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  getMapCoordinate,
  getSpatialLayerColor,
  getSpatialLayerRadiusMeters,
  SpatialLayerType,
} from "@/lib/mapSpatial";

// Initialize Supabase client using Vite env vars (or fall back to process.env)
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || (process.env.VITE_SUPABASE_URL as string) || "";
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || (process.env.VITE_SUPABASE_ANON_KEY as string) || "";
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
const MAP_REFRESH_INTERVAL_MS = 60_000;

const MapView = () => {
  const [selected, setSelected] = useState<string | undefined>();
  const [nodesLatest, setNodesLatest] = useState<Array<any>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeLayer, setActiveLayer] = useState<SpatialLayerType>("coverage");

  useEffect(() => {
    let cancelled = false;
    let initialLoad = true;

    const fetchLatestPerNode = async () => {
      if (initialLoad) setLoading(true);
      setError(null);
      try {
        const { data, error: sbError } = await supabase
          .from("capstone_dataset")
          .select("*")
          .order("Timestamp", { ascending: false })
          .limit(1000);

        if (sbError) throw sbError;
        if (cancelled) return;

        const arr = Array.isArray(data) ? data : [];

        // Reduce to one latest row per distinct node_id
        const seen = new Map<string, any>();
        for (const row of arr) {
          const id = String(row.Node_ID);
          if (!seen.has(id)) seen.set(id, row);
        }

        const latest = Array.from(seen.values()).sort((a, b) => (
          String(a.Node_ID).localeCompare(String(b.Node_ID))
        ));
        setNodesLatest(latest);
        setSelected((current) => (
          current && latest.some((item) => String(item.Node_ID) === current)
            ? current
            : latest[0]?.Node_ID
        ));
      } catch (err: any) {
        if (cancelled) return;
        setError(err.message || String(err));
        if (initialLoad) setNodesLatest([]);
      } finally {
        if (!cancelled && initialLoad) setLoading(false);
        initialLoad = false;
      }
    };

    fetchLatestPerNode();
    const intervalId = window.setInterval(fetchLatestPerNode, MAP_REFRESH_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  const node = nodesLatest.find((n) => String(n.Node_ID) === String(selected));

  const layers = [
    { id: "none", label: "None", icon: MapPreview }, // placeholder icon
    { id: "coverage", label: "Coverage", icon: Wifi },
    { id: "moisture", label: "Moisture", icon: Droplets },
    { id: "nitrogen", label: "Nitrogen", icon: Leaf },
    { id: "health", label: "Health", icon: Activity },
  ] as const;

  const layerGuidance: Record<Exclude<SpatialLayerType, "none">, {
    description: string;
    legend: Array<{ color: string; label: string }>;
  }> = {
    coverage: {
      description: "Recent telemetry coverage using the 60-minute active-node window.",
      legend: [{ color: "#10b981", label: "Active" }, { color: "#ef4444", label: "Stale/offline" }],
    },
    moisture: {
      description: "Relative soil-moisture distribution across the sensor network.",
      legend: [{ color: "#f59e0b", label: "Lower" }, { color: "#06b6d4", label: "Mid-range" }, { color: "#2563eb", label: "Higher" }],
    },
    nitrogen: {
      description: "Relative nitrogen distribution across the sensor network.",
      legend: [{ color: "#facc15", label: "Lower" }, { color: "#84cc16", label: "Mid-range" }, { color: "#15803d", label: "Higher" }],
    },
    health: {
      description: "Combined communication and basic sensor-condition overview.",
      legend: [{ color: "#10b981", label: "Good" }, { color: "#f59e0b", label: "Watch" }, { color: "#ef4444", label: "Attention" }],
    },
  };

  return (
    <>
      <PageHeader title="Map View" subtitle="Geospatial overview of the entire sensor network" />
      <div className="p-6 grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6">
        <div className="bg-card border border-border rounded-xl shadow-card p-4">
          {error && (
            <div className="mb-3 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              Map telemetry could not be loaded: {error}
            </div>
          )}
          {loading ? (
            <div className="h-[calc(100vh-220px)] flex items-center justify-center">Loading map…</div>
          ) : (
            <MapPreview 
              nodes={nodesLatest} 
              selectedId={selected} 
              onSelect={setSelected} 
              height="h-[calc(100vh-220px)]"
              interactive={true}
            >
              {activeLayer !== "none" && nodesLatest.map((n) => {
                const coordinate = getMapCoordinate(n);
                if (!coordinate) return null;
                const color = getSpatialLayerColor(activeLayer, n, nodesLatest);
                return (
                  <Circle
                    key={`circle-${n.Node_ID}`}
                    center={coordinate}
                    radius={getSpatialLayerRadiusMeters(activeLayer)}
                    pathOptions={{
                      color,
                      fillColor: color,
                      fillOpacity: activeLayer === "coverage" ? 0.16 : 0.3,
                      weight: 2,
                      dashArray: activeLayer === "coverage" ? "5 5" : undefined,
                      interactive: false,
                    }}
                  />
                );
              })}
            </MapPreview>
          )}
        </div>
        
        <div className="flex flex-col gap-6">
          <aside className="glass-card border border-border rounded-xl shadow-card p-5">
            <h2 className="font-semibold mb-1">Spatial Analysis</h2>
            <p className="text-xs text-muted-foreground mb-4">Compare network conditions geographically</p>
            <div className="flex flex-col gap-2">
              {layers.map((l) => {
                if (l.id === "none") return null;
                const Icon = l.icon;
                const isActive = activeLayer === l.id;
                return (
                  <button
                    key={l.id}
                    onClick={() => setActiveLayer(l.id)}
                    className={cn(
                      "flex items-center gap-3 px-3 py-2.5 rounded-lg border transition-all text-sm font-medium",
                      isActive 
                        ? "bg-primary text-primary-foreground border-primary shadow-glow" 
                        : "bg-black/20 border-white/5 text-foreground hover:bg-white/5"
                    )}
                  >
                    <Icon className={cn("h-4 w-4", isActive ? "text-primary-foreground" : "text-primary")} />
                    {l.label} Overlay
                  </button>
                );
              })}
              <button
                onClick={() => setActiveLayer("none")}
                className={cn(
                  "flex items-center justify-center mt-2 px-3 py-2 rounded-lg border transition-all text-xs font-mono uppercase tracking-wider",
                  activeLayer === "none" 
                    ? "bg-secondary text-foreground border-white/20" 
                    : "bg-transparent border-transparent text-muted-foreground hover:text-foreground"
                )}
              >
                Clear Overlays
              </button>
            </div>
            {activeLayer !== "none" && (
              <div className="mt-4 border-t border-border/70 pt-4">
                <p className="text-xs leading-relaxed text-muted-foreground">
                  {layerGuidance[activeLayer].description}
                </p>
                <div className="mt-3 flex flex-wrap gap-x-3 gap-y-2">
                  {layerGuidance[activeLayer].legend.map((item) => (
                    <span key={item.label} className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                      <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                      {item.label}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </aside>

          <aside className="glass-card border border-border rounded-xl shadow-card p-5 flex-1">
            <h2 className="font-semibold mb-1">Sensor Details</h2>
            <p className="text-xs text-muted-foreground mb-4">Click a marker to view details</p>
            {node ? (
              <div className="space-y-3">
                <div>
                  <p className="text-[10px] font-sans font-bold uppercase tracking-widest text-muted-foreground">Selected</p>
                  <p className="text-xl font-display font-bold uppercase tracking-wider">{node.Node_ID}</p>
                  <p className="text-xs font-mono text-muted-foreground">{node.Target_Crop ?? "-"} · {getNigerianSeason(node.Timestamp)}</p>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-4 border-t border-white/10">
                  {Object.entries(node).filter(([k]) => k !== "id" && k !== "Node_ID").map(([k, v]) => (
                    <div key={k} className="bg-black/20 rounded-lg p-2.5 border border-white/5">
                      <p className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground">{k.replace(/_/g, ' ')}</p>
                      <p className="text-sm font-mono font-bold mt-0.5 truncate" title={String(v)}>{String(v)}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-40 flex items-center justify-center border border-dashed border-white/10 rounded-xl bg-black/10">
                <p className="text-sm font-mono text-muted-foreground italic">No node selected.</p>
              </div>
            )}
          </aside>
        </div>
      </div>
    </>
  );
};

export default MapView;
