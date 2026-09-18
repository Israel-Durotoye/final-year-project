const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1";

export type CropRecommendation = {
  crop: string | null;
  confidence: number | null;
};

export async function fetchCropRecommendations(nodeIds: string[]): Promise<Record<string, CropRecommendation>> {
  const entries = await Promise.all(nodeIds.map(async (nodeId) => {
    try {
      const response = await fetch(`${API_BASE}/ml/classify-suitability`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: nodeId }),
      });
      if (!response.ok) return [nodeId, { crop: null, confidence: null }] as const;
      const data = await response.json();
      return [nodeId, {
        crop: typeof data?.predicted_crop === "string" ? data.predicted_crop : null,
        confidence: typeof data?.crop_confidence === "number" ? data.crop_confidence : null,
      }] as const;
    } catch {
      return [nodeId, { crop: null, confidence: null }] as const;
    }
  }));

  return Object.fromEntries(entries);
}

export const formatCropRecommendation = (recommendation?: CropRecommendation) => {
  if (!recommendation?.crop) return "Recommendation unavailable";
  const confidence = recommendation.confidence === null ? "" : ` (${Math.round(recommendation.confidence * 100)}%)`;
  return `${recommendation.crop}${confidence}`;
};