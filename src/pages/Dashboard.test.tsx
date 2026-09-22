import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
vi.mock("@/lib/cropRecommendation", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/cropRecommendation")>(),
  fetchCropRecommendations: vi.fn(async () => ({})),
}));
vi.mock("@/lib/telemetry", () => ({
  fetchTelemetry: vi.fn(async () => [telemetryRow, { ...telemetryRow, Node_ID: "NODE_02" }]),
  latestTelemetryByNode: vi.fn(() => [telemetryRow, { ...telemetryRow, Node_ID: "NODE_02" }]),
}));

import Dashboard from "./Dashboard";

describe("Dashboard Field Report", () => {
  beforeEach(() => {
    cleanup();
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
    const [, options] = apiFetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(options.body as string)).toMatchObject({
      response_mode: "field_summary", node_id: "NODE_01", top_k: 5,
    });
    expect(options.body).toContain("one short sentence");

    firstRender.unmount();
    render(<Dashboard />);
    expect(await screen.findByText("NODE_01 conditions are stable.")).toBeInTheDocument();
    expect(apiFetch).toHaveBeenCalledTimes(1);
  });

  it("displays a single plain-language summary", async () => {
    const report = "Soil moisture has fallen over the recorded period, so check the root zone before adjusting irrigation.";
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ answer: report }) })));
    render(<Dashboard />);
    const button = await screen.findByRole("button", { name: /get field report/i });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);
    expect(await screen.findByText(report)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Overall assessment" })).not.toBeInTheDocument();
    expect(screen.queryByRole("list")).not.toBeInTheDocument();
  });

  it("keeps the previous report visible when regeneration fails", async () => {
    const apiFetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ answer: "Previous assessment." }) })
      .mockResolvedValueOnce({ ok: false, status: 503 });
    vi.stubGlobal("fetch", apiFetch);
    render(<Dashboard />);
    const button = await screen.findByRole("button", { name: /get field report/i });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);
    await screen.findByText("Previous assessment.");
    fireEvent.click(button);
    expect(await screen.findByRole("alert")).toHaveTextContent("The previous report remains visible.");
    expect(screen.getByText("Previous assessment.")).toBeInTheDocument();
  });

  it("does not display a late report after switching nodes", async () => {
    let finish: (response: unknown) => void = () => undefined;
    const apiFetch = vi.fn(() => new Promise((resolve) => { finish = resolve; }));
    vi.stubGlobal("fetch", apiFetch);
    render(<Dashboard />);
    const button = await screen.findByRole("button", { name: /get field report/i });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);
    fireEvent.click(screen.getByRole("button", { name: "NODE_02" }));
    finish({ ok: true, json: async () => ({ answer: "Report for the previous node." }) });
    await waitFor(() => expect(button).toBeEnabled());
    expect(screen.queryByText("Report for the previous node.")).not.toBeInTheDocument();
    expect(window.localStorage.length).toBe(0);
  });
});
