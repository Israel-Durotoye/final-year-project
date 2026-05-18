import { useState, useMemo, useEffect } from "react";
import { MapContainer, TileLayer, Marker, Tooltip, Polyline, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Layers, Map as MapIcon, Maximize } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  nodes: any[];
  height?: string;
  selectedId?: string;
  onSelect?: (id: string) => void;
  interactive?: boolean;
  children?: React.ReactNode;
}

// Center map helper component
const MapCenterer = ({ center, zoom }: { center: [number, number], zoom: number }) => {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
};

const ZoomResetControl = ({ coordinates }: { coordinates: [number, number][] }) => {
  const map = useMap();
  if (coordinates.length === 0) return null;
  return (
    <div className="leaflet-top leaflet-left" style={{ top: '80px' }}>
      <div className="leaflet-control leaflet-bar">
        <button
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            const bounds = L.latLngBounds(coordinates);
            map.fitBounds(bounds, { padding: [50, 50] });
          }}
          className="w-[34px] h-[34px] flex items-center justify-center bg-background border-border text-foreground hover:bg-muted transition-colors"
          title="Reset View to All Nodes"
          type="button"
        >
          <Maximize className="h-[14px] w-[14px]" />
        </button>
      </div>
    </div>
  );
};

export const MapPreview = ({ nodes, height = "h-80", selectedId, onSelect, interactive = true, children }: Props) => {
  const [mapStyle, setMapStyle] = useState<"street" | "satellite">("satellite");

  // Normalize nodes (handles both mock data and Supabase schema)
  const normalizedNodes = useMemo(() => {
    return nodes.map((n) => {
      // Prioritize node_id. If a table has an 'id' primary key, we don't want it to override the node_id
      const id = n.node_id || n.id;
      const lat = Number(n.lat ?? n.latitude);
      const lng = Number(n.lng ?? n.longitude);
      const isOnline = n.status === "online" || n.communication_ok === true || n.communication_ok === 1 || n.communication_ok === "true" || n.communication_ok === undefined;
      return { ...n, id: String(id), lat, lng, isOnline };
    }).filter((n) => !isNaN(n.lat) && !isNaN(n.lng));
  }, [nodes]);

  const defaultCenter: [number, number] = normalizedNodes.length > 0 
    ? [normalizedNodes[0].lat, normalizedNodes[0].lng] 
    : [36.7378, -119.7871]; // Default to Fresno area if empty

  // Custom marker icon using HTML
  const createMarkerIcon = (isSelected: boolean, isOnline: boolean) => {
    const bgColor = isOnline ? "var(--primary)" : "var(--muted)";
    const textColor = isOnline ? "var(--primary-foreground)" : "var(--muted-foreground)";
    const ring = isSelected ? `box-shadow: 0 0 0 4px hsl(var(--primary) / 0.3); transform: scale(1.15);` : "";
    
    return L.divIcon({
      className: "custom-node-marker",
      html: `
        <div style="
          width: 28px; 
          height: 28px; 
          background-color: hsl(${bgColor}); 
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: hsl(${textColor});
          transition: all 0.3s ease;
          ${ring}
        ">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="2"/><path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14"/></svg>
        </div>
      `,
      iconSize: [28, 28],
      iconAnchor: [14, 14],
      popupAnchor: [0, -14],
      tooltipAnchor: [0, -14]
    });
  };

  const coordinates = normalizedNodes.map(n => [n.lat, n.lng] as [number, number]);

  return (
    <div className={cn("relative w-full rounded-xl border border-border overflow-hidden isolate", height)}>
      <MapContainer
        center={defaultCenter}
        zoom={normalizedNodes.length > 0 ? 15 : 4}
        className="h-full w-full z-0"
        zoomControl={interactive}
        dragging={interactive}
        scrollWheelZoom={interactive}
        doubleClickZoom={interactive}
      >
        <MapCenterer center={defaultCenter} zoom={normalizedNodes.length > 0 ? 15 : 4} />
        {interactive && <ZoomResetControl coordinates={coordinates} />}
        
        {mapStyle === "street" ? (
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> contributors'
          />
        ) : (
          <TileLayer
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            attribution="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
          />
        )}

        {/* Dotted lines connecting nodes */}
        {coordinates.length > 1 && (
          <Polyline 
            positions={coordinates} 
            pathOptions={{ color: "hsl(var(--primary))", dashArray: "6, 8", weight: 2, opacity: 0.8, interactive: false }} 
          />
        )}

        {/* Dynamic Overlay Layers Injected by Parent */}
        {children}

        {normalizedNodes.map((n) => (
          <Marker 
            key={n.id} 
            position={[n.lat, n.lng]}
            icon={createMarkerIcon(selectedId === n.id, n.isOnline)}
            eventHandlers={{
              click: () => onSelect?.(n.id)
            }}
          >
            {interactive && (
              <Tooltip direction="top" offset={[0, -10]} opacity={1} permanent={false} interactive={false}>
                <span className="font-bold font-display text-xs uppercase tracking-wider text-black">{n.id}</span>
              </Tooltip>
            )}
          </Marker>
        ))}
      </MapContainer>

      {/* Aesthetic Layer Toggle */}
      {interactive && (
        <div className="absolute top-4 right-4 z-[400] flex bg-background/80 backdrop-blur-md border border-white/10 rounded-lg p-1 shadow-lg">
          <button
            onClick={() => setMapStyle("street")}
            className={cn(
              "px-3 py-1.5 rounded-md text-[10px] font-mono font-bold uppercase tracking-wider flex items-center gap-1.5 transition-all",
              mapStyle === "street" ? "bg-primary text-primary-foreground shadow-sm" : "text-foreground/60 hover:text-foreground"
            )}
            type="button"
          >
            <MapIcon className="h-3 w-3" /> Map
          </button>
          <button
            onClick={() => setMapStyle("satellite")}
            className={cn(
              "px-3 py-1.5 rounded-md text-[10px] font-mono font-bold uppercase tracking-wider flex items-center gap-1.5 transition-all",
              mapStyle === "satellite" ? "bg-primary text-primary-foreground shadow-sm" : "text-foreground/60 hover:text-foreground"
            )}
            type="button"
          >
            <Layers className="h-3 w-3" /> Sat
          </button>
        </div>
      )}
    </div>
  );
};
