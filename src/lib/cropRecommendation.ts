const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1";

export type CropRecommendation = {
  crop: string | null;
  confidence: number | null;
  status?: "ready" | "insufficient_data" | "unavailable";
  readingsUsed?: number;
  readingsRequired?: number;
  windowStart?: string;
  windowEnd?: string;
  imputedValues?: number;
  totalValues?: number;
};

export async function fetchCropRecommendations(nodeIds: string[]): Promise<Record<string, CropRecommendation>> {
  const entries = await Promise.all(nodeIds.map(async (nodeId) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 45_000);
    try {
      const response = await fetch(`${API_BASE}/ml/classify-suitability`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: nodeId }),
        signal: controller.signal,
      });
      if (!response.ok) return [nodeId, { crop: null, confidence: null }] as const;
      const data = await response.json();
      return [nodeId, {
        crop: typeof data?.predicted_crop === "string" ? data.predicted_crop : null,
        confidence: typeof data?.crop_confidence === "number" ? data.crop_confidence : null,
        status: data?.crop_status,
        readingsUsed: data?.readings_used,
        readingsRequired: data?.readings_required,
        windowStart: data?.crop_prediction?.window_start,
        windowEnd: data?.crop_prediction?.window_end,
        imputedValues: data?.crop_prediction?.imputed_values,
        totalValues: data?.crop_prediction?.total_values,
      }] as const;
    } catch {
      return [nodeId, { crop: null, confidence: null }] as const;
    } finally {
      clearTimeout(timeout);
    }
  }));

  return Object.fromEntries(entries);
}

export const formatCropRecommendation = (recommendation?: CropRecommendation) => {
  if (!recommendation) return "Checking sensor history…";
  if (recommendation.status === "insufficient_data") return `Collecting readings (${recommendation.readingsUsed ?? 0}/${recommendation.readingsRequired ?? 24})`;
  if (!recommendation?.crop) return "Recommendation unavailable";
  const confidence = recommendation.confidence === null ? "" : ` (${Math.round(recommendation.confidence * 100)}% model score)`;
  return `${recommendation.crop}${isTentativeCropRecommendation(recommendation) ? " · tentative" : ""}${confidence}`;
};

export const isTentativeCropRecommendation = (recommendation?: CropRecommendation) => !!recommendation?.crop && (
  recommendation.confidence === null || recommendation.confidence < 0.5
  || (recommendation.imputedValues ?? 0) > (recommendation.totalValues ?? 144) / 4
);
