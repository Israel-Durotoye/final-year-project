import { isNodeActive } from "@/lib/nodeActivity";
import { AlertThresholds, DEFAULT_THRESHOLDS, METRICS, MetricKey } from "@/lib/alerting";

export type SpatialLayerType = "none" | "coverage" | "health" | MetricKey;

export type SpatialNode = {
  [key: string]: unknown;
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
  const latitude = finiteNumber(node.Latitude ?? node.lat);
  const longitude = finiteNumber(node.Longitude ?? node.lng);

  if (latitude === null || longitude === null) return null;
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
  if ((typeof value !== "number" && typeof value !== "string") || String(value).trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

export const SPATIAL_COLORS = { low: "#f59e0b", within: "#10b981", high: "#ef4444", unknown: "#64748b" };

export const spatialMetric = (layer: SpatialLayerType) => METRICS.find((metric) => metric.key === layer);
export const formatSpatialValue = (value: number, unit: string) => `${Number(value.toFixed(1))}${unit === "%" || unit === "°C" ? "" : " "}${unit}`;

export type SpatialAssessment = {
  status: "low" | "within" | "high" | "missing" | "stale" | "active" | "attention";
  label: string;
  color: string;
  value: number | null;
  reading: string;
  explanation: string;
  action: string;
};

export const getSpatialAssessment = (
  layer: SpatialLayerType, node: SpatialNode, thresholds: AlertThresholds = DEFAULT_THRESHOLDS, now = new Date(),
): SpatialAssessment => {
  const unknown = { color: SPATIAL_COLORS.unknown, value: null, reading: "No valid reading" };
  if (!communicationIsAvailable(node) || !isNodeActive(node, now)) return {
    ...unknown, status: "stale", label: "Stale / offline", reading: "No recent reading",
    explanation: "No current reading with a working connection in the last 60 minutes.",
    action: "Check the node connection and obtain a fresh reading before making field decisions.",
  };
  if (layer === "coverage" || layer === "none") return {
    status: "active", label: "Active", color: SPATIAL_COLORS.within, value: null, reading: "Active",
    explanation: "This sensor has reported within the last 60 minutes.", action: "Choose a sensor layer to interpret its measurements.",
  };
  if (layer === "health") {
    const assessments = METRICS.map((metric) => getSpatialAssessment(metric.key, node, thresholds, now));
    const flagged = assessments.filter((item) => item.status === "low" || item.status === "high");
    if (flagged.length) return {
      ...unknown, status: "attention", label: "Outside limits", color: SPATIAL_COLORS.high,
      reading: `${flagged.length} outside limits`,
      explanation: METRICS.filter((_, index) => ["low", "high"].includes(assessments[index].status))
        .map((metric) => `${metric.label.toLowerCase()} is ${getSpatialAssessment(metric.key, node, thresholds, now).label.toLowerCase()}`).join("; ") + ".",
      action: "Select a flagged measurement below to see what needs checking.",
    };
    if (assessments.some((item) => item.status === "missing")) return {
      ...unknown, status: "missing", label: "Incomplete readings", explanation: "Some measurements are missing or invalid; a complete condition check is unavailable.",
      action: "Check the missing measurements before treating this area as within limits.",
    };
    return { ...unknown, status: "within", label: "Within limits", reading: "Within limits", color: SPATIAL_COLORS.within,
      explanation: "All six measurements are within your configured alert limits.", action: "Continue field observation; these limits do not establish overall crop health." };
  }
  const metric = spatialMetric(layer)!;
  const value = finiteNumber(node[metric.column]);
  const invalid = value === null || ((metric.unit === "%") && (value < 0 || value > 100))
    || (metric.unit === "mg/kg" && value < 0);
  if (invalid) return { ...unknown, status: "missing", label: "No valid reading",
    explanation: `${metric.label} is missing or invalid for this sensor.`, action: "Check the sensor and obtain a valid reading before comparing this location." };
  const range = thresholds[metric.key];
  const status = value < range.min ? "low" : value > range.max ? "high" : "within";
  const label = status === "low" ? "Below limit" : status === "high" ? "Above limit" : "Within limits";
  const reading = formatSpatialValue(value, metric.unit);
  const relation = status === "low" ? "below" : status === "high" ? "above" : "within";
  const nutrient = metric.unit === "mg/kg";
  const action = status === "within" ? "No correction is indicated by this alert limit; continue observing the field."
    : nutrient ? "Confirm the reading against the crop's needs and soil conditions before changing fertiliser inputs."
    : status === "low" ? metric.lowRecommendation : metric.highRecommendation;
  return { status, label, value, reading, color: SPATIAL_COLORS[status], action,
    explanation: `${metric.label} is ${reading}, ${relation} your ${formatSpatialValue(range.min, metric.unit)}–${formatSpatialValue(range.max, metric.unit)} alert range.` };
};

export const getSpatialLayerColor = (
  layer: SpatialLayerType,
  node: SpatialNode,
  _allNodes: SpatialNode[],
  now: Date = new Date(),
  thresholds: AlertThresholds = DEFAULT_THRESHOLDS,
): string => {
  if (layer === "none") return SPATIAL_COLORS.unknown;
  const assessment = getSpatialAssessment(layer, node, thresholds, now);
  if (layer === "coverage" && assessment.status === "stale") return SPATIAL_COLORS.high;
  return assessment.color;
};

export const getSpatialComparison = (layer: SpatialLayerType, nodes: SpatialNode[], thresholds = DEFAULT_THRESHOLDS, now = new Date()) => {
  const mapped = nodes.filter((node) => getMapCoordinate(node));
  const readings = mapped.map((node) => ({ node, assessment: getSpatialAssessment(layer, node, thresholds, now) }));
  const valid = readings.filter(({ assessment }) => assessment.value !== null);
  return {
    readings, unmapped: nodes.length - mapped.length,
    low: readings.filter(({ assessment }) => assessment.status === "low").length,
    high: readings.filter(({ assessment }) => ["high", "attention"].includes(assessment.status)).length,
    current: readings.filter(({ assessment }) => !["stale", "missing"].includes(assessment.status)).length,
    minimum: valid.length ? Math.min(...valid.map(({ assessment }) => assessment.value!)) : null,
    maximum: valid.length ? Math.max(...valid.map(({ assessment }) => assessment.value!)) : null,
  };
};

export const distanceMeters = (a: MapCoordinate, b: MapCoordinate): number => {
  const radians = (degrees: number) => degrees * Math.PI / 180;
  const h = Math.sin(radians(b[0] - a[0]) / 2) ** 2
    + Math.cos(radians(a[0])) * Math.cos(radians(b[0])) * Math.sin(radians(b[1] - a[1]) / 2) ** 2;
  return 6371000 * 2 * Math.asin(Math.sqrt(Math.min(1, Math.max(0, h))));
};

export const getNearestSensorReadings = (node: SpatialNode, nodes: SpatialNode[], layer: SpatialLayerType, thresholds = DEFAULT_THRESHOLDS, now = new Date()) => {
  const origin = getMapCoordinate(node);
  if (!origin || !spatialMetric(layer) || getSpatialAssessment(layer, node, thresholds, now).value === null) return [];
  return nodes.flatMap((other) => {
    const coordinate = getMapCoordinate(other);
    const assessment = getSpatialAssessment(layer, other, thresholds, now);
    if (!coordinate || other.Node_ID === node.Node_ID || assessment.value === null || other.Data_Source !== node.Data_Source) return [];
    return [{ node: other, assessment, distance: distanceMeters(origin, coordinate) }];
  }).sort((a, b) => a.distance - b.distance).slice(0, 2);
};

export const getSpatialLayerRadiusMeters = (layer: SpatialLayerType): number => (
  layer === "coverage" ? 90 : 65
);
