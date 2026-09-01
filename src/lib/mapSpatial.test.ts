import { describe, expect, it } from "vitest";

import {
  getMapCoordinate,
  getSpatialLayerColor,
  getSpatialLayerRadiusMeters,
  isMapNodeOnline,
  orderCoordinatesAroundCenter,
} from "@/lib/mapSpatial";

describe("map spatial helpers", () => {
  const now = new Date("2026-08-31T12:00:00.000Z");

  it("accepts valid coordinates and rejects invalid coordinates", () => {
    expect(getMapCoordinate({ Latitude: 8.48, Longitude: 4.54 })).toEqual([8.48, 4.54]);
    expect(getMapCoordinate({ Latitude: 98, Longitude: 4.54 })).toBeNull();
    expect(getMapCoordinate({ Latitude: null, Longitude: null })).toBeNull();
  });

  it("orders an unordered set of vertices into a closed-map perimeter order", () => {
    const vertices: Array<[number, number]> = [
      [1, 0],
      [-1, 0],
      [0.5, 0.866],
      [-0.5, -0.866],
      [0.5, -0.866],
      [-0.5, 0.866],
    ];
    const ordered = orderCoordinatesAroundCenter(vertices);

    expect(ordered).toHaveLength(6);
    expect(new Set(ordered.map((point) => point.join(","))).size).toBe(6);
    for (let index = 0; index < ordered.length; index += 1) {
      const current = ordered[index];
      const next = ordered[(index + 1) % ordered.length];
      expect(Math.hypot(current[0] - next[0], current[1] - next[1])).toBeCloseTo(1, 2);
    }
  });

  it("uses the 60-minute freshness rule for coverage", () => {
    const active = { Timestamp: "2026-08-31T11:30:00.000Z" };
    const stale = { Timestamp: "2026-08-31T10:59:00.000Z" };

    expect(isMapNodeOnline(active, now)).toBe(true);
    expect(isMapNodeOnline(stale, now)).toBe(false);
    expect(getSpatialLayerColor("coverage", active, [active, stale], now)).toBe("#10b981");
    expect(getSpatialLayerColor("coverage", stale, [active, stale], now)).toBe("#ef4444");
  });

  it("uses relative network colors and meter-based overlay radii", () => {
    const nodes = [
      { "Moisture_%": 30, Nitrogen_mg_k: 20 },
      { "Moisture_%": 60, Nitrogen_mg_k: 60 },
      { "Moisture_%": 90, Nitrogen_mg_k: 100 },
    ];

    expect(getSpatialLayerColor("moisture", nodes[0], nodes, now)).toBe("#f59e0b");
    expect(getSpatialLayerColor("moisture", nodes[2], nodes, now)).toBe("#2563eb");
    expect(getSpatialLayerColor("nitrogen", nodes[0], nodes, now)).toBe("#facc15");
    expect(getSpatialLayerColor("nitrogen", nodes[2], nodes, now)).toBe("#15803d");
    expect(getSpatialLayerRadiusMeters("coverage")).toBe(90);
    expect(getSpatialLayerRadiusMeters("moisture")).toBe(65);
  });
});
