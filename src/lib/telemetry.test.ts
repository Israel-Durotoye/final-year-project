import { describe, expect, it, vi } from "vitest";
import { getMapCoordinate } from "@/lib/mapSpatial";

vi.mock("@/lib/supabase", () => ({
  supabase: {},
}));

import {
  firebasePushTimestamp,
  fetchTelemetry,
  latestTelemetryByNode,
  normalizeHardwareTelemetry,
  normalizeSimulatorTelemetry,
} from "@/lib/telemetry";

describe("mixed telemetry normalization", () => {
  it.each(["NODE_01", "NODE_02"])("passes %s coordinates to the map without replacing real or fallback locations", (node_id) => {
    for (const point of [
      { latitude: "9.531982", longitude: "6.451488", gps_source: "real" },
      { latitude: "9.056700", longitude: "7.496900", gps_source: "fallback" },
      { latitude: "8.123456", longitude: "4.654321", gps_source: "real" },
    ]) {
      const row = normalizeHardwareTelemetry({ node_id, ...point }, "-P0WR_Zl8PkYTF9bskHK");
      expect(getMapCoordinate(row)).toEqual([Number(point.latitude), Number(point.longitude)]);
      expect(row.GPS_Source).toBe(point.gps_source);
    }
  });

  it("does not turn missing GPS fields into a location at zero", () => {
    for (const invalid of [null, undefined, "", " ", false, [], "invalid", Infinity]) {
      const row = normalizeHardwareTelemetry({ node_id: "NODE_01", latitude: invalid, longitude: invalid }, "-P0WR_Zl8PkYTF9bskHK");
      expect(row.Latitude).toBeNull();
      expect(row.Longitude).toBeNull();
      expect(getMapCoordinate(row)).toBeNull();
    }
    const zero = normalizeHardwareTelemetry({ node_id: "NODE_01", latitude: "0", longitude: "0" }, "-P0WR_Zl8PkYTF9bskHK");
    expect(getMapCoordinate(zero)).toEqual([0, 0]);
  });

  it("loads the INO Firebase log for a physical node", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        "-P0WR_Zl8PkYTF9bskHK": {
          node_id: "NODE_01",
          nitrogen: 120,
          phosphorus: 40,
          potassium: 90,
          moisture: 32,
          temp: 27,
          humidity: 76,
          latitude: "9.532053",
          longitude: "6.451473",
        },
      }),
    } as Response);

    const rows = await fetchTelemetry({ nodeId: "NODE_01", limit: 10 });

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(String(fetchMock.mock.calls[0][0])).toContain("/readings/log.json?");
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      Node_ID: "NODE_01",
      Timestamp: "2026-09-02T09:27:59.025Z",
      Data_Source: "hardware",
    });
    fetchMock.mockRestore();
  });

  it("maps the physical sensor schema into the frontend schema", () => {
    const row = normalizeHardwareTelemetry({
      node_id: "NODE_01",
      nitrogen: 120,
      phosphorus: 40,
      potassium: 90,
      moisture: 32,
      temp: 27,
      humidity: 76.3,
      ph: 5.8,
      latitude: "9.532053",
      longitude: "6.451473",
      altitude: "235.7",
      satellites: "5",
      season: "Dry",
      gps_source: "real",
      timestamp: 1272,
    }, "-P0WRUo9W3XVjhrfpWa-");

    expect(row).toMatchObject({
      Node_ID: "NODE_01",
      Nitrogen_mg_k: 120,
      Phosphorus_m: 40,
      Potassium_mg_: 90,
      "Moisture_%": 32,
      Temperature_C: 27,
      "Humidity_%": 76.3,
      Soil_pH: 5.8,
      Latitude: 9.532053,
      Longitude: 6.451473,
      Altitude_m: 235.7,
      Satellites: "5",
      Season: "Dry",
      GPS_Source: "real",
      Device_Uptime_Seconds: 1272,
      Data_Source: "hardware",
    });
  });

  it("derives wall-clock time from the Firebase push ID", () => {
    expect(firebasePushTimestamp("-P0WR_Zl8PkYTF9bskHK"))
      .toBe("2026-09-02T09:27:59.025Z");
  });

  it("keeps only the newest normalized row for each node", () => {
    const rows = [
      normalizeSimulatorTelemetry({ Node_ID: "NODE_04", Timestamp: "2026-09-02T07:00:00Z" }),
      normalizeHardwareTelemetry({ node_id: "NODE_01" }, "-P0WR_Zl8PkYTF9bskHK"),
      normalizeSimulatorTelemetry({ Node_ID: "NODE_04", Timestamp: "2026-09-02T06:00:00Z" }),
    ];

    expect(latestTelemetryByNode(rows).map((row) => row.Node_ID)).toEqual(["NODE_01", "NODE_04"]);
    expect(latestTelemetryByNode(rows)[1].Timestamp).toBe("2026-09-02T07:00:00Z");
  });

  it("places simulator nodes in one area with distinct coordinates", () => {
    const rows = ["NODE_04", "NODE_05", "NODE_06", "NODE_07"].map((Node_ID) => (
      normalizeSimulatorTelemetry({ Node_ID, Timestamp: "2026-09-02T07:00:00Z" })
    ));

    expect(new Set(rows.map((row) => `${row.Latitude},${row.Longitude}`)).size).toBe(4);
    expect(rows.every((row) => Number(row.Latitude) > 9.52 && Number(row.Latitude) < 9.54)).toBe(true);
    expect(rows.every((row) => Number(row.Longitude) > 6.44 && Number(row.Longitude) < 6.46)).toBe(true);
  });
});
