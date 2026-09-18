import { useMemo, useEffect } from "react";
import { MapContainer, TileLayer, Marker, Tooltip, Polygon, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Maximize } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  getMapCoordinate,
  isMapNodeOnline,
  orderCoordinatesAroundCenter,
  SpatialNode,
} from "@/lib/mapSpatial";

interface Props {
  nodes: SpatialNode[];
  height?: string;
  selectedId?: string;
  onSelect?: (id: string) => void;
  interactive?: boolean;
  children?: React.ReactNode;
  nodeLabels?: Record<string, string>;
  nodeColors?: Record<string, string>;
  showNetworkBoundary?: boolean;
}

// Center map helper component
const DEFAULT_CENTER: [number, number] = [9.5325, 6.4525];
export const OPENSTREETMAP_TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
export const OPENSTREETMAP_ATTRIBUTION = (
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
);

const MapViewport = ({ coordinates }: { coordinates: [number, number][] }) => {
  const map = useMap();

  useEffect(() => {
    const frameId = window.requestAnimationFrame(() => {
      map.invalidateSize({ pan: false });

      if (coordinates.length === 0) {
        map.setView(DEFAULT_CENTER, 13);
      } else if (coordinates.length === 1) {
        map.setView(coordinates[0], 17);
      } else {
        map.fitBounds(L.latLngBounds(coordinates), { padding: [48, 48], maxZoom: 17 });
      }
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [coordinates, map]);

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

export const MapPreview = ({ nodes, height = "h-80", selectedId, onSelect, interactive = true, children, nodeLabels, nodeColors, showNetworkBoundary = true }: Props) => {
  // Normalize nodes (handles Supabase schema)
  const normalizedNodes = useMemo(() => {
    return nodes.map((n) => {
      // Prioritize node_id. If a table has an 'id' primary key, we don't want it to override the node_id
      const id = n.Node_ID || n.id;
      const coordinate = getMapCoordinate(n);
      if (!coordinate || !id) return null;
      return {
        ...n,
        id: String(id),
        lat: coordinate[0],
        lng: coordinate[1],
        isOnline: isMapNodeOnline(n),
      };
    }).filter((node): node is NonNullable<typeof node> => node !== null);
  }, [nodes]);

  const coordinates = useMemo(
    () => normalizedNodes.map((node) => [node.lat, node.lng] as [number, number]),
    [normalizedNodes],
  );
  const perimeterCoordinates = useMemo(
    () => orderCoordinatesAroundCenter(coordinates),
    [coordinates],
  );

  // Custom marker icon using HTML
  const createMarkerIcon = (isSelected: boolean, isOnline: boolean, color?: string) => {
    const bgColor = isOnline ? "var(--primary)" : "var(--muted)";
    const textColor = isOnline ? "var(--primary-foreground)" : "var(--muted-foreground)";
    const ring = isSelected ? `box-shadow: 0 0 0 4px hsl(var(--primary) / 0.3); transform: scale(1.15);` : "";
    
    return L.divIcon({
      className: "custom-node-marker",
      html: `
        <div style="
          width: 28px; 
          height: 28px; 
          background-color: ${color ?? `hsl(${bgColor})`};
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

  return (
    <div className={cn("relative w-full rounded-xl border border-border overflow-hidden isolate", height)}>
      <MapContainer
        center={DEFAULT_CENTER}
        zoom={13}
        className="h-full w-full z-0"
        zoomControl={interactive}
        dragging={interactive}
        scrollWheelZoom={interactive}
        doubleClickZoom={interactive}
      >
        <MapViewport coordinates={coordinates} />
        {interactive && <ZoomResetControl coordinates={coordinates} />}
        
        <TileLayer
          url={OPENSTREETMAP_TILE_URL}
          attribution={OPENSTREETMAP_ATTRIBUTION}
          maxZoom={19}
        />

        {/* Dotted lines connecting nodes */}
        {showNetworkBoundary && perimeterCoordinates.length > 2 && (
          <Polygon
            positions={perimeterCoordinates}
            pathOptions={{
              color: "hsl(var(--primary))",
              dashArray: "6, 8",
              weight: 2,
              opacity: 0.9,
              fillColor: "hsl(var(--primary))",
              fillOpacity: 0.06,
              interactive: false,
            }}
          />
        )}

        {/* Dynamic Overlay Layers Injected by Parent */}
        {children}

        {normalizedNodes.map((n) => (
          <Marker 
            key={n.id} 
            position={[n.lat, n.lng]}
            title={`Select ${n.id}`}
            icon={createMarkerIcon(selectedId === n.id, n.isOnline, nodeColors?.[n.id])}
            eventHandlers={{
              click: () => onSelect?.(n.id)
            }}
          >
            {interactive && (
              <Tooltip direction="top" offset={[0, -10]} opacity={1} permanent={Boolean(nodeLabels)} interactive={false}>
                <span className="font-bold font-display text-xs uppercase tracking-wider text-black">{n.id}</span>
                {nodeLabels?.[n.id] && <span className="block text-xs text-black">{nodeLabels[n.id]}</span>}
              </Tooltip>
            )}
          </Marker>
        ))}
      </MapContainer>

    </div>
  );
};
