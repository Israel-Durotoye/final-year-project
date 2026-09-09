import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { telemetryRow } = vi.hoisted(() => ({ telemetryRow: {
  Node_ID: "NODE_01",
  Timestamp: "2026-09-09T08:00:00Z",
  Data_Source: "hardware",
  Nitrogen_mg_k: 120,
  Phosphorus_m: 40,
  Potassium_mg_: 90,
  "Moisture_%": 32,
  Temperature_C: 27,
} }));

vi.mock("@/components/layout/PageHeader", () => ({ PageHeader: () => <div>Dashboard</div> }));
vi.mock("@/components/MetricCard", () => ({ MetricCard: ({ label }: { label: string }) => <div>{label}</div> }));
vi.mock("@/components/MapPreview", () => ({ MapPreview: () => <div>Map</div> }));
vi.mock("@/components/TrendCharts", () => ({ TrendCharts: () => <div>Trends</div> }));
vi.mock("@/lib/telemetry", () => ({
  fetchTelemetry: vi.fn(async () => [telemetryRow]),
  latestTelemetryByNode: vi.fn(() => [telemetryRow]),
}));

import Dashboard from "./Dashboard";

describe("Dashboard Field Report", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.unstubAllGlobals();
  });

  it("requests a report only when the user presses Get Field Report and restores it", async () => {
    const apiFetch = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ answer: "NODE_01 conditions are stable." }),
    } as Response));
    vi.stubGlobal("fetch", apiFetch);

    const firstRender = render(<Dashboard />);
    const reportButton = await screen.findByRole("button", { name: /get field report/i });
    await waitFor(() => expect(reportButton).toBeEnabled());
    expect(apiFetch).not.toHaveBeenCalled();

    fireEvent.click(reportButton);
    expect(await screen.findByText("NODE_01 conditions are stable.")).toBeInTheDocument();
    expect(apiFetch).toHaveBeenCalledTimes(1);

    firstRender.unmount();
    render(<Dashboard />);
    expect(await screen.findByText("NODE_01 conditions are stable.")).toBeInTheDocument();
    expect(apiFetch).toHaveBeenCalledTimes(1);
  });
});
