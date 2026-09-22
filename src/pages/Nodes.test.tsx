import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_THRESHOLDS, saveAlertThresholds } from "@/lib/alerting";
import { fetchTelemetry, TelemetryRow } from "@/lib/telemetry";
import { fetchCropRecommendations } from "@/lib/cropRecommendation";
import Nodes from "./Nodes";

vi.mock("@/components/layout/PageHeader", () => ({ PageHeader: () => <h1>Nodes</h1> }));
vi.mock("@/lib/telemetry", () => ({ fetchTelemetry: vi.fn(), latestTelemetryByNode: (rows: TelemetryRow[]) => rows }));
vi.mock("@/lib/cropRecommendation", async (importOriginal) => ({ ...await importOriginal<typeof import("@/lib/cropRecommendation")>(), fetchCropRecommendations: vi.fn() }));

const row = (id: string, overrides = {}): TelemetryRow => ({
  Node_ID: id, Timestamp: new Date().toISOString(), Data_Source: "hardware",
  Nitrogen_mg_k: 35, Phosphorus_m: 50, Potassium_mg_: 200, "Moisture_%": 45, Temperature_C: 25, "Humidity_%": 60, ...overrides,
});
const renderPage = () => render(<MemoryRouter><Nodes /></MemoryRouter>);

beforeEach(() => {
  localStorage.clear();
  vi.mocked(fetchTelemetry).mockResolvedValue([row("NODE_01"), row("NODE_02", { "Moisture_%": 85 }), row("NODE_03", { Data_Source: "simulator", Target_Crop: "Cassava" })]);
  vi.mocked(fetchCropRecommendations).mockImplementation(async (ids) => Object.fromEntries(ids.map((id) => [id, { crop: "Maize", confidence: 0.82, readingsUsed: 24, status: "ready" }])));
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("Nodes overview", () => {
  it("keeps the original sensor row order and displays crop predictions separately", async () => {
    renderPage();
    const hardware = await screen.findByRole("article", { name: "NODE_01" });
    expect(await within(hardware).findByText("Maize")).toBeInTheDocument();
    expect(within(hardware).getByText(/24 recent readings · 82% model score/)).toBeInTheDocument();
    expect(within(hardware).queryByText(/Recorded planting/)).not.toBeInTheDocument();
    expect(within(screen.getByRole("article", { name: "NODE_03" })).getByText("Cassava")).toBeInTheDocument();
    expect(screen.getAllByRole("article")).toHaveLength(3);
    expect(within(hardware).getAllByRole("meter").map((meter) => meter.getAttribute("aria-label"))).toEqual([
      "Nitrogen at NODE_01", "Phosphorus at NODE_01", "Potassium at NODE_01", "Moisture at NODE_01", "Humidity at NODE_01", "Temperature at NODE_01",
    ]);
    const details = within(hardware).getByText("Prediction details").closest("details");
    expect(details).not.toHaveAttribute("open");
    fireEvent.click(within(hardware).getByText("Prediction details"));
    expect(details).toHaveAttribute("open");
  });

  it("uses Settings limits for attention and keeps missing values distinct from zero", async () => {
    vi.mocked(fetchTelemetry).mockResolvedValue([row("NODE_01", { "Humidity_%": null }), row("NODE_02", { "Moisture_%": 85 })]);
    renderPage();
    const first = await screen.findByRole("article", { name: "NODE_01" });
    expect(within(first).getByText("—")).toBeInTheDocument();
    expect(within(first).getByText("INCOMPLETE")).toBeInTheDocument();
    expect(within(first).getByRole("meter", { name: "Humidity at NODE_01" })).not.toHaveAttribute("aria-valuenow");
    const second = screen.getByRole("article", { name: "NODE_02" });
    expect(within(second).getByText("POOR")).toBeInTheDocument();
    act(() => saveAlertThresholds({ ...DEFAULT_THRESHOLDS, moisture: { min: 25, max: 90 } }));
    expect(within(second).getByText("GOOD")).toBeInTheDocument();
  });

  it("uses green, orange and red for good, warning and critical readings", async () => {
    vi.mocked(fetchTelemetry).mockResolvedValue([
      row("NODE_01"), row("NODE_02", { Nitrogen_mg_k: 18 }), row("NODE_03", { Nitrogen_mg_k: 16, "Moisture_%": 85 }),
    ]);
    renderPage();
    const good = await screen.findByRole("article", { name: "NODE_01" });
    const fair = screen.getByRole("article", { name: "NODE_02" });
    const poor = screen.getByRole("article", { name: "NODE_03" });
    expect(within(good).getByText("GOOD")).toHaveClass("text-green-700");
    expect(within(fair).getByText("FAIR")).toHaveClass("text-orange-700");
    expect(within(poor).getByText("POOR")).toHaveClass("text-red-700");
    expect(within(good).getByRole("meter", { name: "Nitrogen at NODE_01" }).firstChild).toHaveClass("bg-green-500");
    expect(within(fair).getByRole("meter", { name: "Nitrogen at NODE_02" }).firstChild).toHaveClass("bg-orange-500");
    expect(within(poor).getByRole("meter", { name: "Nitrogen at NODE_03" }).firstChild).toHaveClass("bg-red-500");
    expect(within(poor).getByRole("meter", { name: "Moisture at NODE_03" }).firstChild).toHaveClass("bg-red-500");
  });

  it("refreshes predictions and explains insufficient history", async () => {
    vi.mocked(fetchTelemetry).mockResolvedValue([row("NODE_01")]);
    vi.mocked(fetchCropRecommendations).mockResolvedValueOnce({ NODE_01: { crop: null, confidence: null, status: "insufficient_data", readingsUsed: 12, readingsRequired: 24 } });
    renderPage();
    expect(await screen.findByText("12 of 24 readings collected.")).toBeInTheDocument();
    const refresh = screen.getByRole("button", { name: "Refresh crop prediction for NODE_01" });
    await waitFor(() => expect(refresh).toBeEnabled());
    fireEvent.click(refresh);
    expect(await screen.findByText("Maize")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Diagnose Node" })).toHaveAttribute("href", "/ai-doctor");
  });

  it("labels low-confidence crop predictions as tentative", async () => {
    vi.mocked(fetchTelemetry).mockResolvedValue([row("NODE_01")]);
    vi.mocked(fetchCropRecommendations).mockResolvedValue({ NODE_01: { crop: "Tomato", confidence: 0.4, readingsUsed: 24, imputedValues: 55, totalValues: 144 } });
    renderPage();
    expect(await screen.findByText("Tomato")).toBeInTheDocument();
    expect(screen.getByText("Tentative match")).toBeInTheDocument();
    expect(screen.getByText(/Collect more valid readings/)).toBeInTheDocument();
  });

  it("retains readings on refresh failure and labels offline data without current limit scores", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      vi.mocked(fetchTelemetry).mockResolvedValueOnce([row("NODE_01", { Timestamp: "2020-01-01T00:00:00Z" })]).mockRejectedValue(new Error("Connection lost"));
      renderPage();
      expect(await screen.findByText("OFFLINE")).toBeInTheDocument();
      for (const meter of screen.getAllByRole("meter")) {
        expect(meter).toHaveAttribute("aria-valuetext", expect.stringContaining("Last reading; sensor offline"));
      }
      expect(screen.queryByText("Within limits")).not.toBeInTheDocument();
      await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
      expect(screen.getByRole("alert")).toHaveTextContent("Showing the last available readings");
      expect(screen.getByRole("article", { name: "NODE_01" })).toBeInTheDocument();
    } finally { vi.useRealTimers(); }
  });
});
