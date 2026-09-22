import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_THRESHOLDS, saveAlertThresholds } from "@/lib/alerting";
import type { ReactNode } from "react";

const { fetchRows, rows } = vi.hoisted(() => {
  const current = { Timestamp: new Date().toISOString(), Data_Source: "hardware", Latitude: 9.53, Longitude: 6.45,
    Target_Crop: "Maize", Nitrogen_mg_k: 10, Phosphorus_m: 40, Potassium_mg_: 200,
    "Moisture_%": 80, Temperature_C: 25, "Humidity_%": 50 };
  return { fetchRows: vi.fn(), rows: [
    { ...current, Node_ID: "NODE_01" },
    { ...current, Node_ID: "NODE_02", Latitude: 9.5301, "Moisture_%": 30, Nitrogen_mg_k: 40 },
    { ...current, Node_ID: "NODE_03", Latitude: 9.531, Timestamp: "2020-01-01T00:00:00Z", "Moisture_%": 99 },
  ] };
});

vi.mock("@/components/layout/PageHeader", () => ({ PageHeader: () => <h1>Map View</h1> }));
vi.mock("@/lib/telemetry", () => ({
  HARDWARE_NODE_IDS: ["NODE_01", "NODE_02"],
  SIMULATOR_NODE_IDS: ["NODE_03", "NODE_04", "NODE_05", "NODE_06"],
  fetchTelemetry: fetchRows, latestTelemetryByNode: (items: unknown[]) => items,
}));
vi.mock("@/lib/cropRecommendation", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/cropRecommendation")>(),
  fetchCropRecommendations: vi.fn(async () => ({})),
}));
vi.mock("react-leaflet", () => ({ Circle: () => <span data-testid="overlay-circle" /> }));
vi.mock("@/components/MapPreview", () => ({
  MapPreview: ({ nodes, onSelect, nodeLabels, children }: {
    nodes: Array<{ Node_ID: string }>; onSelect: (id: string) => void;
    nodeLabels?: Record<string, string>; children: ReactNode;
  }) => <section aria-label="Interactive sensor map">
    {nodes.map((node) => <button key={node.Node_ID} onClick={() => onSelect(node.Node_ID)} aria-label={`Select ${node.Node_ID} on map`}>
      {node.Node_ID}<span>{nodeLabels?.[node.Node_ID]}</span>
    </button>)}{children}
  </section>,
}));

import MapView from "./MapView";

describe("MapView spatial interpretation", () => {
  beforeEach(() => { window.localStorage.clear(); fetchRows.mockReset(); fetchRows.mockResolvedValue(rows); });
  afterEach(cleanup);

  it("shows values on the map and explains the selected measurement without an AI request", async () => {
    render(<MapView />);
    const map = await screen.findByRole("region", { name: "Interactive sensor map" });
    expect(within(map).getByText("80% · Above limit")).toBeInTheDocument();
    expect(screen.getByText("Measured range across current mapped nodes: 30%–80%.")).toBeInTheDocument();
    const details = screen.getByRole("complementary", { name: "Selected sensor interpretation" });
    expect(within(details).getByText(/Soil moisture is 80%, above your 25%–65% alert range/)).toBeInTheDocument();
    expect(within(details).getByText(/Check for standing water/)).toBeInTheDocument();
    expect(fetchRows).toHaveBeenCalledTimes(1);
  });

  it("updates values, explanation and selected sensor when a layer and marker are clicked", async () => {
    render(<MapView />);
    const map = await screen.findByRole("region", { name: "Interactive sensor map" });
    fireEvent.click(screen.getByRole("button", { name: "Nitrogen" }));
    expect(within(map).getByText("10 mg/kg · Below limit")).toBeInTheDocument();
    fireEvent.click(within(map).getByRole("button", { name: "Select NODE_02 on map" }));
    const details = screen.getByRole("complementary", { name: "Selected sensor interpretation" });
    expect(within(details).getByRole("heading", { name: "NODE_02" })).toBeInTheDocument();
    expect(within(details).getByText(/Nitrogen is 40 mg\/kg, within your 20 mg\/kg–50 mg\/kg alert range/)).toBeInTheDocument();
    expect(within(details).getByText("30 mg/kg lower than this location")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Clear overlays" }));
    expect(screen.queryByTestId("overlay-circle")).not.toBeInTheDocument();
    expect(within(map).queryByText("10 mg/kg · Below limit")).not.toBeInTheDocument();
  });

  it("marks stale readings and updates interpretation when Settings limits change", async () => {
    render(<MapView />);
    const map = await screen.findByRole("region", { name: "Interactive sensor map" });
    act(() => saveAlertThresholds({ ...DEFAULT_THRESHOLDS, moisture: { min: 25, max: 90 } }));
    expect(within(map).getByText("80% · Within limits")).toBeInTheDocument();
    fireEvent.click(within(map).getByRole("button", { name: "Select NODE_03 on map" }));
    const details = screen.getByRole("complementary", { name: "Selected sensor interpretation" });
    expect(within(details).getByText(/No current reading with a working connection/)).toBeInTheDocument();
    expect(within(details).queryByText(/standing water/)).not.toBeInTheDocument();
    expect(within(details).getByText(/No other current, located sensors/)).toBeInTheDocument();
  });

  it("shows an empty state and a refresh error when telemetry cannot be loaded", async () => {
    fetchRows.mockRejectedValue(new Error("Connection unavailable"));
    render(<MapView />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Connection unavailable");
    expect(screen.getByText("No sensor locations are available to plot.")).toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});
