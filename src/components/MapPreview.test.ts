import { describe, expect, it } from "vitest";

import {
  OPENSTREETMAP_ATTRIBUTION,
  OPENSTREETMAP_TILE_URL,
} from "@/components/MapPreview";

describe("MapPreview basemap", () => {
  it("uses unauthenticated OpenStreetMap tiles", () => {
    expect(OPENSTREETMAP_TILE_URL).toBe(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    );
    expect(OPENSTREETMAP_TILE_URL).not.toMatch(/key|token/i);
    expect(OPENSTREETMAP_ATTRIBUTION).toContain("OpenStreetMap");
  });
});
