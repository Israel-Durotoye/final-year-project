import { beforeEach, describe, expect, it } from "vitest";

import {
  FIELD_REPORT_TTL_MS,
  readFieldReport,
  saveFieldReport,
} from "@/lib/fieldReportCache";

describe("field report cache", () => {
  beforeEach(() => window.localStorage.clear());

  it("keeps a node report unchanged for 30 minutes", () => {
    const generatedAt = Date.UTC(2026, 8, 9, 8, 0, 0);
    saveFieldReport("node_01", "Field conditions are stable.", generatedAt);

    expect(readFieldReport("NODE_01", generatedAt + FIELD_REPORT_TTL_MS - 1)?.report)
      .toBe("Field conditions are stable.");
  });

  it("expires and removes a report after 30 minutes", () => {
    const generatedAt = Date.UTC(2026, 8, 9, 8, 0, 0);
    saveFieldReport("NODE_02", "Monitor moisture.", generatedAt);

    expect(readFieldReport("NODE_02", generatedAt + FIELD_REPORT_TTL_MS)).toBeNull();
    expect(window.localStorage.length).toBe(0);
  });
});
