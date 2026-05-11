import { PageHeader } from "@/components/layout/PageHeader";
import { MapPreview } from "@/components/MapPreview";
import { nodes } from "@/lib/mockData";
import { useState } from "react";

const MapView = () => {
  const [selected, setSelected] = useState<string>();
  const node = nodes.find((n) => n.id === selected);
  return (
    <>
      <PageHeader title="Map View" subtitle="Geospatial overview of the entire sensor network" />
      <div className="p-6 grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6">
        <div className="bg-card border border-border rounded-xl shadow-card p-4">
          <MapPreview nodes={nodes} selectedId={selected} onSelect={setSelected} height="h-[calc(100vh-220px)]" />
        </div>
        <aside className="bg-card border border-border rounded-xl shadow-card p-5">
          <h2 className="font-semibold mb-1">Node Inspector</h2>
          <p className="text-xs text-muted-foreground mb-4">Click a marker to view details</p>
          {node ? (
            <div className="space-y-3">
              <div>
                <p className="text-xs uppercase tracking-wider text-muted-foreground">Selected</p>
                <p className="text-lg font-semibold">{node.id}</p>
                <p className="text-sm text-muted-foreground">{node.farm} · {node.location}</p>
              </div>
              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-border">
                {Object.entries(node.metrics).map(([k, v]) => (
                  <div key={k} className="bg-secondary/50 rounded-lg p-2.5">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{k}</p>
                    <p className="text-sm font-semibold">{v}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground italic">No node selected.</p>
          )}
        </aside>
      </div>
    </>
  );
};

export default MapView;
