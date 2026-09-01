import { describe, expect, it } from "vitest";

import {
  ACTIVE_NODE_WINDOW_MINUTES,
  activityCutoffIso,
  countActiveNodes,
  isNodeActive,
} from "@/lib/nodeActivity";

describe("node activity", () => {
  const now = new Date("2026-08-31T12:00:00.000Z");

  it("uses a 60-minute activity cutoff", () => {
    expect(ACTIVE_NODE_WINDOW_MINUTES).toBe(60);
    expect(activityCutoffIso(now)).toBe("2026-08-31T11:00:00.000Z");
    expect(isNodeActive({ Timestamp: "2026-08-31T11:00:00.000Z" }, now)).toBe(true);
    expect(isNodeActive({ Timestamp: "2026-08-31T10:59:59.999Z" }, now)).toBe(false);
  });

  it("counts unique recently reporting nodes", () => {
    const rows = [
      { Node_ID: "NODE_01", Timestamp: "2026-08-31T11:15:00.000Z" },
      { Node_ID: "NODE_01", Timestamp: "2026-08-31T11:45:00.000Z" },
      { Node_ID: "NODE_02", Timestamp: "2026-08-31T11:05:00.000Z" },
      { Node_ID: "NODE_03", Timestamp: "2026-08-31T10:59:00.000Z" },
    ];

    expect(countActiveNodes(rows, now)).toBe(2);
  });

  it("rejects missing, invalid, and future timestamps", () => {
    expect(isNodeActive({ Timestamp: null }, now)).toBe(false);
    expect(isNodeActive({ Timestamp: "not-a-date" }, now)).toBe(false);
    expect(isNodeActive({ Timestamp: "2026-08-31T12:01:00.000Z" }, now)).toBe(false);
    expect(countActiveNodes([{ Node_ID: "", Timestamp: now.toISOString() }], now)).toBe(0);
  });
});
