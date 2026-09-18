import { describe, expect, it } from "vitest";

import {
  getMapCoordinate,
  getSpatialLayerColor,
  getSpatialLayerRadiusMeters,
  isMapNodeOnline,
  orderCoordinatesAroundCenter,
  getSpatialAssessment,
  getSpatialComparison,
  getNearestSensorReadings,
  SPATIAL_COLORS,
} from "@/lib/mapSpatial";
import { DEFAULT_THRESHOLDS } from "@/lib/alerting";

describe("map spatial helpers", () => {
  const now = new Date("2026-08-31T12:00:00.000Z");

  it("accepts valid coordinates and rejects invalid coordinates", () => {
    expect(getMapCoordinate({ Latitude: 8.48, Longitude: 4.54 })).toEqual([8.48, 4.54]);
    expect(getMapCoordinate({ Latitude: 98, Longitude: 4.54 })).toBeNull();
    expect(getMapCoordinate({ Latitude: null, Longitude: null })).toBeNull();
    expect(getMapCoordinate({ Latitude: " ", Longitude: "" })).toBeNull();
    expect(getMapCoordinate({ Latitude: false, Longitude: [] })).toBeNull();
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

  it("uses configured limits rather than relative colours that could hide a problem", () => {
    const nodes = [
      { "Moisture_%": 15, Nitrogen_mg_k: 10, Timestamp: now.toISOString() },
      { "Moisture_%": 60, Nitrogen_mg_k: 40, Timestamp: now.toISOString() },
      { "Moisture_%": 90, Nitrogen_mg_k: 100, Timestamp: now.toISOString() },
    ];

    expect(getSpatialLayerColor("moisture", nodes[0], nodes, now)).toBe(SPATIAL_COLORS.low);
    expect(getSpatialLayerColor("moisture", nodes[2], nodes, now)).toBe(SPATIAL_COLORS.high);
    expect(getSpatialLayerColor("nitrogen", nodes[0], nodes, now)).toBe(SPATIAL_COLORS.low);
    expect(getSpatialLayerColor("nitrogen", nodes[2], nodes, now)).toBe(SPATIAL_COLORS.high);
    expect(getSpatialLayerColor("moisture", nodes[2], [nodes[2]], now)).toBe(SPATIAL_COLORS.high);
    expect(getSpatialLayerRadiusMeters("coverage")).toBe(90);
    expect(getSpatialLayerRadiusMeters("moisture")).toBe(65);
  });

  const current = { Node_ID: "NODE_01", Timestamp: now.toISOString(), Latitude: 9.53, Longitude: 6.45,
    Data_Source: "hardware", Nitrogen_mg_k: 30, Phosphorus_m: 40, Potassium_mg_: 200,
    "Moisture_%": 40, Temperature_C: 25, "Humidity_%": 50 };

  it("checks all measurements and detects excessive wetness in the condition layer", () => {
    expect(getSpatialAssessment("health", current, DEFAULT_THRESHOLDS, now).status).toBe("within");
    for (const column of ["Moisture_%", "Temperature_C", "Phosphorus_m", "Potassium_mg_", "Humidity_%"]) {
      const value = column === "Humidity_%" || column === "Moisture_%" ? 95 : 500;
      expect(getSpatialAssessment("health", { ...current, [column]: value }, DEFAULT_THRESHOLDS, now).status).toBe("attention");
    }
  });

  it("does not treat null, blank or impossible readings as healthy or zero", () => {
    for (const value of [null, "", " ", false, [], 101, -1]) {
      expect(getSpatialAssessment("moisture", { ...current, "Moisture_%": value }, DEFAULT_THRESHOLDS, now).status).toBe("missing");
    }
    expect(getSpatialAssessment("moisture", { ...current, "Moisture_%": 0 }, DEFAULT_THRESHOLDS, now).status).toBe("low");
    expect(getSpatialAssessment("health", { ...current, Phosphorus_m: null }, DEFAULT_THRESHOLDS, now).status).toBe("missing");
  });

  it("excludes stale readings and unlocated nodes from numeric comparisons", () => {
    const rows = [current, { ...current, Node_ID: "OLD", Timestamp: "2020-01-01", "Moisture_%": 99 },
      { ...current, Node_ID: "NO_GPS", Latitude: null, Longitude: null, "Moisture_%": 5 }];
    const result = getSpatialComparison("moisture", rows, DEFAULT_THRESHOLDS, now);
    expect(result.current).toBe(1);
    expect(result.unmapped).toBe(1);
    expect(result.minimum).toBe(40);
    expect(result.maximum).toBe(40);
    expect(result.high).toBe(0);
    expect(getSpatialAssessment("moisture", rows[1], DEFAULT_THRESHOLDS, now).status).toBe("stale");
  });

  it("uses saved thresholds and includes their units in the explanation", () => {
    const thresholds = { ...DEFAULT_THRESHOLDS, moisture: { min: 10, max: 30 } };
    const result = getSpatialAssessment("moisture", current, thresholds, now);
    expect(result.status).toBe("high");
    expect(result.explanation).toContain("10%–30%");
  });

  it("orders actual nearby sensors by distance without mixing hardware and simulator data", () => {
    const rows = [current, { ...current, Node_ID: "FAR", Latitude: 9.54 },
      { ...current, Node_ID: "NEAR", Latitude: 9.5301 },
      { ...current, Node_ID: "SIM", Latitude: 9.53001, Data_Source: "simulator" }];
    const result = getNearestSensorReadings(current, rows, "moisture", DEFAULT_THRESHOLDS, now);
    expect(result.map(({ node }) => node.Node_ID)).toEqual(["NEAR", "FAR"]);
    expect(result[0].distance).toBeCloseTo(11.12, 1);
  });
});
