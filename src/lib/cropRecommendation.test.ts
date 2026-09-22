import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchCropRecommendations, formatCropRecommendation, isTentativeCropRecommendation } from "./cropRecommendation";

afterEach(() => vi.unstubAllGlobals());

describe("crop recommendation API", () => {
  it("preserves model evidence and does not substitute the recorded crop", async () => {
    const request = vi.fn(async () => ({ ok: true, json: async () => ({
      crop: "Maize", predicted_crop: "Tomato", crop_confidence: 0.4, crop_status: "ready", readings_used: 24, readings_required: 24,
      crop_prediction: { window_start: "2026-09-01T00:00:00Z", window_end: "2026-09-17T00:00:00Z", imputed_values: 55, total_values: 144 },
    }) }));
    vi.stubGlobal("fetch", request);
    const result = (await fetchCropRecommendations(["NODE_01"]))["NODE_01"];
    expect(result).toMatchObject({ crop: "Tomato", readingsUsed: 24, imputedValues: 55, totalValues: 144 });
    expect(formatCropRecommendation(result)).toBe("Tomato · tentative (40% model score)");
    expect(request).toHaveBeenCalledWith(expect.stringContaining("/ml/classify-suitability"), expect.objectContaining({ body: JSON.stringify({ node_id: "NODE_01" }) }));
  });

  it("distinguishes collecting history from an unavailable model", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ predicted_crop: null, crop_status: "insufficient_data", readings_used: 8, readings_required: 24 }) })));
    expect(formatCropRecommendation((await fetchCropRecommendations(["NODE_02"]))["NODE_02"])).toBe("Collecting readings (8/24)");
    expect(formatCropRecommendation({ crop: null, confidence: null })).toBe("Recommendation unavailable");
  });

  it("returns a per-node unavailable state when a request fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    expect(await fetchCropRecommendations(["NODE_01"])).toEqual({ NODE_01: { crop: null, confidence: null } });
  });

  it("keeps a high-scoring prediction tentative when many inputs were estimated", () => {
    expect(isTentativeCropRecommendation({ crop: "Rice", confidence: 0.95, imputedValues: 80, totalValues: 144 })).toBe(true);
  });
});
