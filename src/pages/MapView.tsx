import { PageHeader } from "@/components/layout/PageHeader";
import { useEffect, useState } from "react";
import { createClient } from "@supabase/supabase-js";
import { MapPreview } from "@/components/MapPreview";
import { CircleMarker } from "react-leaflet";
import { Droplets, Leaf, ShieldAlert, Wifi, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

// Initialize Supabase client using Vite env vars (or fall back to process.env)
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || (process.env.VITE_SUPABASE_URL as string) || "";
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || (process.env.VITE_SUPABASE_ANON_KEY as string) || "";
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

type LayerType = "none" | "coverage" | "moisture" | "nitrogen" | "health";

const MapView = () => {
  const [selected, setSelected] = useState<string | undefined>();
  const [nodesLatest, setNodesLatest] = useState<Array<any>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeLayer, setActiveLayer] = useState<LayerType>("coverage");

  useEffect(() => {
    const fetchLatestPerNode = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data, error: sbError } = await supabase
          .from("sensor_telemetry")
          .select("*")
          .order("timestamp_utc", { ascending: false })
          .limit(1000);

        if (sbError) throw sbError;

        const arr = Array.isArray(data) ? data : [];

        // Reduce to one latest row per distinct node_id
        const seen = new Map<string, any>();
        for (const row of arr) {
          const id = String(row.node_id);
          if (!seen.has(id)) seen.set(id, row);
        }

        setNodesLatest(Array.from(seen.values()));
      } catch (err: any) {
        setError(err.message || String(err));
        setNodesLatest([]);
      } finally {
        setLoading(false);
      }
    };

    fetchLatestPerNode();
  }, []);

  const node = nodesLatest.find((n) => String(n.node_id) === String(selected));

  const getLayerColor = (layer: LayerType, n: any) => {
    const isOnline = !(n.communication_ok === false || n.communication_ok === 0 || n.communication_ok === "false");

    if (layer === "moisture") {
      const v = Number(n.soil_moisture ?? n.moisture ?? 0);
      if (v < 30) return "#ef4444"; // red (low moisture)
      return "#10b981"; // green (good moisture)
    }
    if (layer === "nitrogen") {
      const v = Number(n.nitrogen_ppm ?? n.nitrogen ?? 0);
      if (v < 25) return "#ef4444"; // red (low nitrogen)
      return "#10b981"; // green (good nitrogen)
    }
    if (layer === "coverage") {
      return isOnline ? "#10b981" : "#ef4444"; // green if connected, red if disconnected
    }
    if (layer === "health") {
      if (!isOnline) return "#ef4444"; // red (offline)
      
      const nit = Number(n.nitrogen_ppm ?? n.nitrogen ?? 0);
      const moisture = Number(n.soil_moisture ?? n.moisture ?? 0);
      
      if (nit < 20 || moisture < 25) return "#ef4444"; // red (poor)
      if ((nit >= 20 && nit <= 30) || (moisture >= 25 && moisture <= 35)) return "#f59e0b"; // amber (fair)
      return "#10b981"; // emerald (good)
    }
    return "#10b981"; // default
  };

  const layers = [
    { id: "none", label: "None", icon: MapPreview }, // placeholder icon
    { id: "coverage", label: "Coverage", icon: Wifi },
    { id: "moisture", label: "Moisture", icon: Droplets },
    { id: "nitrogen", label: "Nitrogen", icon: Leaf },
    { id: "health", label: "Health", icon: Activity },
  ] as const;

  return (
    <>
      <PageHeader title="Map View" subtitle="Geospatial overview of the entire sensor network" />
      <div className="p-6 grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6">
        <div className="bg-card border border-border rounded-xl shadow-card p-4">
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
                const lat = Number(n.latitude ?? n.lat);
                const lng = Number(n.longitude ?? n.lng);
                if (isNaN(lat) || isNaN(lng)) return null;
                return (
                  <CircleMarker
                    key={`circle-${n.node_id}`}
                    center={[lat, lng]}
                    radius={45} // 45 pixels radius
                    pathOptions={{
                      color: getLayerColor(activeLayer, n),
                      fillColor: getLayerColor(activeLayer, n),
                      fillOpacity: 0.35,
                      weight: 2,
                      dashArray: "4 4",
                      interactive: false // Allows clicks to pass through to the Marker underneath
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
            <p className="text-xs text-muted-foreground mb-4">Overlay real-time heatmaps</p>
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
                    {l.label} Heatmap
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
          </aside>

          <aside className="glass-card border border-border rounded-xl shadow-card p-5 flex-1">
            <h2 className="font-semibold mb-1">Node Inspector</h2>
            <p className="text-xs text-muted-foreground mb-4">Click a marker to view details</p>
            {node ? (
              <div className="space-y-3">
                <div>
                  <p className="text-[10px] font-sans font-bold uppercase tracking-widest text-muted-foreground">Selected</p>
                  <p className="text-xl font-display font-bold uppercase tracking-wider">{node.node_id}</p>
                  <p className="text-xs font-mono text-muted-foreground">{node.farm ?? "-"} · {node.location ?? "-"}</p>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-4 border-t border-white/10">
                  {Object.entries(node).filter(([k]) => k !== "id" && k !== "node_id").map(([k, v]) => (
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
