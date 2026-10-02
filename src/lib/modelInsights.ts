const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1";

export type ForecastRisk = {
  sensor: string;
  direction: "below_reference_range" | "above_reference_range";
  horizon: string;
  weak: boolean;
};

export type ModelInsights = {
  anomalyStatus: string;
  isAnomalous: boolean;
  unusualSensors: string[];
  forecastStatus: string;
  forecastHorizon: string | null;
  samplesAvailable?: number;
  samplesRequired?: number;
  risks: ForecastRisk[];
};

/** Fetch the forecaster and anomaly-screen results for one node. */
export async function fetchModelInsights(nodeId: string): Promise<ModelInsights | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 45_000);
  try {
    const response = await fetch(`${API_BASE}/ml/temporal/${encodeURIComponent(nodeId)}`, { signal: controller.signal });
    if (!response.ok) return null;
    const data = await response.json();
    const anomaly = data?.anomaly_screening ?? {};
    const horizons = Object.keys(data?.forecast ?? {});
    const risks: ForecastRisk[] = (data?.forecast_outlook?.risks ?? []).map((risk: Record<string, unknown>) => ({
      sensor: String(risk.sensor ?? ""),
      direction: risk.direction === "below_reference_range" ? "below_reference_range" : "above_reference_range",
      horizon: String(risk.first_horizon ?? ""),
      weak: String(risk.forecast_reliability ?? "").includes("level estimate"),
    }));
    return {
      anomalyStatus: String(anomaly.status ?? "unavailable"),
      isAnomalous: anomaly.is_anomalous === true,
      unusualSensors: Array.isArray(anomaly.unusual_sensors) && anomaly.unusual_sensors.length
        ? anomaly.unusual_sensors : (anomaly.largest_contributors ?? []),
      forecastStatus: String(data?.forecast_status ?? "unavailable"),
      forecastHorizon: horizons.length ? horizons[horizons.length - 1] : null,
      samplesAvailable: data?.history?.contiguous_tail_samples,
      samplesRequired: data?.samples_required_for_forecast,
      risks,
    };
  } catch {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

const horizonText = (label: string) => label.replace(/(\d+(?:\.\d+)?)m$/, "$1 min").replace(/(\d+(?:\.\d+)?)h$/, "$1 h").replace(/(\d+(?:\.\d+)?)d$/, "$1 days");

/** Plain-language line for the anomaly screen. */
export const describePatternCheck = (insights?: ModelInsights | null) => {
  if (!insights) return "Unavailable";
  if (insights.anomalyStatus === "success") {
    if (!insights.isAnomalous) return "Normal";
    return insights.unusualSensors.length ? `Unusual: ${insights.unusualSensors.join(", ")}` : "Unusual readings";
  }
  if (insights.anomalyStatus === "insufficient_history" || insights.anomalyStatus === "incomplete_window") return "More readings needed";
  return "Unavailable";
};

/** Plain-language line for the forecast outlook. */
export const describeOutlook = (insights?: ModelInsights | null) => {
  if (!insights) return "Unavailable";
  if (insights.forecastStatus === "success") {
    const horizon = insights.forecastHorizon ? horizonText(insights.forecastHorizon) : "the forecast period";
    if (!insights.risks.length) return `No reading expected to leave its range in the next ${horizon}`;
    return insights.risks.map((risk) =>
      `${risk.sensor} may go ${risk.direction === "below_reference_range" ? "below" : "above"} its range within ${horizonText(risk.horizon)}`
    ).join("; ");
  }
  if (insights.forecastStatus === "insufficient_history") {
    return insights.samplesRequired
      ? `Needs ${insights.samplesRequired} continuous readings (${insights.samplesAvailable ?? 0} so far)`
      : "More continuous readings needed";
  }
  if (insights.forecastStatus === "cadence_mismatch") return "Not available for this sensor's reporting interval";
  return "No forecast available";
};
