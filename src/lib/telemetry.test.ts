import { describe, expect, it, vi } from "vitest";

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
      normalizeSimulatorTelemetry({ Node_ID: "NODE_03", Timestamp: "2026-09-02T07:00:00Z" }),
      normalizeHardwareTelemetry({ node_id: "NODE_01" }, "-P0WR_Zl8PkYTF9bskHK"),
      normalizeSimulatorTelemetry({ Node_ID: "NODE_03", Timestamp: "2026-09-02T06:00:00Z" }),
    ];

    expect(latestTelemetryByNode(rows).map((row) => row.Node_ID)).toEqual(["NODE_01", "NODE_03"]);
    expect(latestTelemetryByNode(rows)[1].Timestamp).toBe("2026-09-02T07:00:00Z");
  });
});
