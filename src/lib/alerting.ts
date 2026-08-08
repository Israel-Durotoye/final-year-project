export type MetricKey = "nitrogen" | "phosphorus" | "potassium" | "moisture" | "temperature" | "humidity";

export type ThresholdRange = {
  min: number;
  max: number;
};

export type AlertThresholds = Record<MetricKey, ThresholdRange>;

export type ThresholdAlert = {
  id: string;
  nodeId: string;
  metric: MetricKey;
  severity: "critical" | "warning";
  direction: "low" | "high";
  value: number;
  threshold: number;
  timestamp: string | null;
  title: string;
  message: string;
  recommendation: string;
};

export const THRESHOLD_STORAGE_KEY = "soilnet-alert-thresholds";

export const METRICS: Array<{
  key: MetricKey;
  label: string;
  unit: string;
  column: string;
  lowRecommendation: string;
  highRecommendation: string;
}> = [
  { key: "nitrogen", label: "Nitrogen", unit: "mg/kg", column: "Nitrogen_mg_k", lowRecommendation: "Inspect nutrient availability and plan a measured nitrogen application.", highRecommendation: "Pause further nitrogen application and check for over-fertilisation or runoff risk." },
  { key: "phosphorus", label: "Phosphorus", unit: "mg/kg", column: "Phosphorus_m", lowRecommendation: "Confirm the soil test and plan phosphorus correction for the affected zone.", highRecommendation: "Avoid additional phosphorus until the next soil review." },
  { key: "potassium", label: "Potassium", unit: "mg/kg", column: "Potassium_mg_", lowRecommendation: "Review potassium availability and consider a targeted correction.", highRecommendation: "Hold potassium inputs and confirm the reading with a follow-up test." },
  { key: "moisture", label: "Soil moisture", unit: "%", column: "Moisture_%", lowRecommendation: "Inspect irrigation and plant stress in this zone before watering.", highRecommendation: "Check for standing water, drainage issues, and avoid additional irrigation." },
  { key: "temperature", label: "Temperature", unit: "°C", column: "Temperature_C", lowRecommendation: "Inspect crops for cold stress and protect vulnerable plants where possible.", highRecommendation: "Check for heat stress and review irrigation timing or shading." },
  { key: "humidity", label: "Humidity", unit: "%", column: "Humidity_%", lowRecommendation: "Check whether dry air is increasing crop water demand.", highRecommendation: "Monitor for conditions that can increase disease pressure." },
];

export const DEFAULT_THRESHOLDS: AlertThresholds = {
  nitrogen: { min: 20, max: 50 },
  phosphorus: { min: 30, max: 75 },
  potassium: { min: 150, max: 260 },
  moisture: { min: 25, max: 65 },
  temperature: { min: 18, max: 30 },
  humidity: { min: 30, max: 85 },
};

export function loadAlertThresholds(): AlertThresholds {
  if (typeof window === "undefined") return DEFAULT_THRESHOLDS;

  try {
    const stored = JSON.parse(window.localStorage.getItem(THRESHOLD_STORAGE_KEY) ?? "{}");
    return METRICS.reduce((thresholds, metric) => {
      const candidate = stored?.[metric.key];
      const min = Number(candidate?.min);
      const max = Number(candidate?.max);
      thresholds[metric.key] = Number.isFinite(min) && Number.isFinite(max) && min < max
        ? { min, max }
        : DEFAULT_THRESHOLDS[metric.key];
      return thresholds;
    }, {} as AlertThresholds);
  } catch {
    return DEFAULT_THRESHOLDS;
  }
}

export function saveAlertThresholds(thresholds: AlertThresholds) {
  window.localStorage.setItem(THRESHOLD_STORAGE_KEY, JSON.stringify(thresholds));
  window.dispatchEvent(new Event("soilnet:thresholds-updated"));
}

const formatValue = (value: number, unit: string) => `${Number(value.toFixed(1))}${unit ? ` ${unit}` : ""}`;

export function evaluateNodeThresholds(row: Record<string, unknown>, thresholds: AlertThresholds): ThresholdAlert[] {
  const nodeId = String(row.Node_ID ?? "Unknown node");
  const timestamp = typeof row.Timestamp === "string" ? row.Timestamp : null;

  return METRICS.flatMap((metric) => {
    const value = Number(row[metric.column]);
    if (!Number.isFinite(value)) return [];

    const range = thresholds[metric.key];
    const direction = value < range.min ? "low" : value > range.max ? "high" : null;
    if (!direction) return [];

    const threshold = direction === "low" ? range.min : range.max;
    const margin = Math.max(Math.abs(threshold) * 0.2, 1);
    const severity = direction === "low"
      ? (value <= threshold - margin ? "critical" : "warning")
      : (value >= threshold + margin ? "critical" : "warning");
    const relation = direction === "low" ? "below" : "above";

    return [{
      id: `${nodeId}-${metric.key}-${timestamp ?? "latest"}`,
      nodeId,
      metric: metric.key,
      severity,
      direction,
      value,
      threshold,
      timestamp,
      title: `${metric.label} is ${direction} on ${nodeId}`,
      message: `${formatValue(value, metric.unit)} is ${relation} your ${formatValue(threshold, metric.unit)} alert threshold.`,
      recommendation: direction === "low" ? metric.lowRecommendation : metric.highRecommendation,
    }];
  });
}

export function latestReadingsByNode(rows: Array<Record<string, unknown>>) {
  const latest = new Map<string, Record<string, unknown>>();
  for (const row of rows) {
    const nodeId = String(row.Node_ID ?? "");
    if (nodeId && !latest.has(nodeId)) latest.set(nodeId, row);
  }
  return [...latest.values()];
}
