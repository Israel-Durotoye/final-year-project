import { afterEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import {
  OPENSTREETMAP_ATTRIBUTION,
  OPENSTREETMAP_TILE_URL,
  MapPreview,
} from "@/components/MapPreview";

describe("MapPreview basemap", () => {
  afterEach(cleanup);
  it("uses unauthenticated OpenStreetMap tiles", () => {
    expect(OPENSTREETMAP_TILE_URL).toBe(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    );
    expect(OPENSTREETMAP_TILE_URL).not.toMatch(/key|token/i);
    expect(OPENSTREETMAP_ATTRIBUTION).toContain("OpenStreetMap");
  });

  it("renders a permanent reading label and selects its actual Leaflet marker", async () => {
    const onSelect = vi.fn();
    render(createElement(MapPreview, {
      nodes: [{ Node_ID: "NODE_01", Latitude: 9.53, Longitude: 6.45, Timestamp: new Date().toISOString() }],
      nodeLabels: { NODE_01: "80% · Above limit" }, nodeColors: { NODE_01: "#ef4444" }, onSelect,
    }));
    expect(await screen.findByText("80% · Above limit")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Select NODE_01" }));
    expect(onSelect).toHaveBeenCalledWith("NODE_01");
  });
});
