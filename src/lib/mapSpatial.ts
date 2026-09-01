import { isNodeActive } from "@/lib/nodeActivity";

export type SpatialLayerType = "none" | "coverage" | "moisture" | "nitrogen" | "health";

export type SpatialNode = {
  id?: unknown;
  Node_ID?: unknown;
  Latitude?: unknown;
  Longitude?: unknown;
  lat?: unknown;
  lng?: unknown;
  Timestamp?: unknown;
  status?: unknown;
  communication_ok?: unknown;
  Nitrogen_mg_k?: unknown;
  "Moisture_%"?: unknown;
};

export type MapCoordinate = [number, number];

const communicationIsAvailable = (node: SpatialNode): boolean => (
  node.communication_ok !== false
  && node.communication_ok !== 0
  && node.communication_ok !== "false"
);

export const isMapNodeOnline = (node: SpatialNode, now: Date = new Date()): boolean => (
  communicationIsAvailable(node)
  && (
    isNodeActive(node, now)
    || (node.Timestamp == null && node.status === "online")
  )
);

export const getMapCoordinate = (node: SpatialNode): MapCoordinate | null => {
  const latitude = Number(node.Latitude ?? node.lat);
  const longitude = Number(node.Longitude ?? node.lng);

  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) return null;

  return [latitude, longitude];
};

export const orderCoordinatesAroundCenter = (coordinates: MapCoordinate[]): MapCoordinate[] => {
  if (coordinates.length < 3) return [...coordinates];

  const center = coordinates.reduce(
    (acc, [latitude, longitude]) => [acc[0] + latitude, acc[1] + longitude] as MapCoordinate,
    [0, 0] as MapCoordinate,
  ).map((value) => value / coordinates.length) as MapCoordinate;

  return [...coordinates].sort((a, b) => (
    Math.atan2(a[0] - center[0], a[1] - center[1])
    - Math.atan2(b[0] - center[0], b[1] - center[1])
  ));
};

const finiteNumber = (value: unknown): number | null => {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const numericValues = (nodes: SpatialNode[], field: "Moisture_%" | "Nitrogen_mg_k"): number[] => (
  nodes
    .map((node) => finiteNumber(node[field]))
    .filter((value): value is number => value !== null)
);

const relativeColor = (value: number, values: number[], palette: [string, string, string]): string => {
  if (!Number.isFinite(value) || values.length === 0) return "#64748b";

  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (minimum === maximum) return palette[1];

  const position = (value - minimum) / (maximum - minimum);
  if (position <= 1 / 3) return palette[0];
  if (position >= 2 / 3) return palette[2];
  return palette[1];
};

export const getSpatialLayerColor = (
  layer: SpatialLayerType,
  node: SpatialNode,
  allNodes: SpatialNode[],
  now: Date = new Date(),
): string => {
  if (layer === "coverage") {
    return isMapNodeOnline(node, now) ? "#10b981" : "#ef4444";
  }

  if (layer === "moisture") {
    const moisture = finiteNumber(node["Moisture_%"]);
    return relativeColor(
      moisture ?? Number.NaN,
      numericValues(allNodes, "Moisture_%"),
      ["#f59e0b", "#06b6d4", "#2563eb"],
    );
  }

  if (layer === "nitrogen") {
    const nitrogen = finiteNumber(node.Nitrogen_mg_k);
    return relativeColor(
      nitrogen ?? Number.NaN,
      numericValues(allNodes, "Nitrogen_mg_k"),
      ["#facc15", "#84cc16", "#15803d"],
    );
  }

  if (layer === "health") {
    if (!isMapNodeOnline(node, now)) return "#ef4444";

    const nitrogen = finiteNumber(node.Nitrogen_mg_k);
    const moisture = finiteNumber(node["Moisture_%"]);
    if (nitrogen === null || moisture === null) return "#64748b";
    if (nitrogen < 20 || moisture < 25) return "#ef4444";
    if (nitrogen <= 30 || moisture <= 35) return "#f59e0b";
    return "#10b981";
  }

  return "#64748b";
};

export const getSpatialLayerRadiusMeters = (layer: SpatialLayerType): number => (
  layer === "coverage" ? 90 : 65
);
