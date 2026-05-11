import { MapPin, Radio } from "lucide-react";
import { SensorNode } from "@/lib/mockData";
import { cn } from "@/lib/utils";

interface Props {
  nodes: SensorNode[];
  height?: string;
  selectedId?: string;
  onSelect?: (id: string) => void;
}

// Decorative map preview (no external map service required)
export const MapPreview = ({ nodes, height = "h-80", selectedId, onSelect }: Props) => {
  // Normalize lat/lng to plotting %
  const lats = nodes.map((n) => n.lat);
  const lngs = nodes.map((n) => n.lng);
  const minLat = Math.min(...lats), maxLat = Math.max(...lats);
  const minLng = Math.min(...lngs), maxLng = Math.max(...lngs);
  const dLat = maxLat - minLat || 1;
  const dLng = maxLng - minLng || 1;

  return (
    <div className={cn("relative w-full rounded-xl border border-border overflow-hidden", height)}
      style={{
        background: "radial-gradient(ellipse at top left, hsl(var(--primary-soft)), hsl(var(--background))), repeating-linear-gradient(0deg, hsl(var(--border)) 0 1px, transparent 1px 48px), repeating-linear-gradient(90deg, hsl(var(--border)) 0 1px, transparent 1px 48px)"
      }}>
      {/* Decorative field shapes */}
      <svg className="absolute inset-0 w-full h-full opacity-40" preserveAspectRatio="none" viewBox="0 0 400 300">
        <path d="M40,60 Q120,40 200,80 T380,90 L380,150 Q280,140 180,170 T20,180 Z" fill="hsl(var(--primary) / 0.08)" />
        <path d="M20,200 Q140,180 240,210 T390,230 L390,290 L20,290 Z" fill="hsl(var(--primary) / 0.12)" />
      </svg>

      {nodes.map((n) => {
        const x = ((n.lng - minLng) / dLng) * 80 + 10;
        const y = (1 - (n.lat - minLat) / dLat) * 70 + 12;
        const isSelected = selectedId === n.id;
        return (
          <button
            key={n.id}
            onClick={() => onSelect?.(n.id)}
            style={{ left: `${x}%`, top: `${y}%` }}
            className="absolute -translate-x-1/2 -translate-y-1/2 group"
          >
            <div className={cn(
              "relative h-9 w-9 rounded-full flex items-center justify-center shadow-lg transition-transform",
              n.status === "online" ? "bg-primary text-primary-foreground" : "bg-muted-foreground/60 text-background",
              isSelected && "scale-125 ring-4 ring-primary/30"
            )}>
              {n.status === "online" && (
                <span className="absolute inset-0 rounded-full bg-primary/40 live-dot" />
              )}
              <Radio className="h-4 w-4 relative z-10" />
            </div>
            <div className="absolute left-1/2 -translate-x-1/2 mt-1.5 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity bg-card border border-border rounded-md px-2 py-1 text-xs font-medium shadow-md pointer-events-none">
              {n.id}
            </div>
          </button>
        );
      })}

      <div className="absolute bottom-3 left-3 bg-card/90 backdrop-blur border border-border rounded-lg px-3 py-2 flex items-center gap-2 text-xs">
        <MapPin className="h-3.5 w-3.5 text-primary" />
        <span className="font-medium">{nodes.length} nodes</span>
        <span className="text-muted-foreground">· Fresno, CA</span>
      </div>
    </div>
  );
};
