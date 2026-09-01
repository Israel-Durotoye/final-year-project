export const ACTIVE_NODE_WINDOW_MINUTES = 60;

type TelemetryActivityRow = {
  Node_ID?: unknown;
  Timestamp?: unknown;
};

const timestampMillis = (value: unknown): number | null => {
  if (typeof value !== "string" && !(value instanceof Date)) return null;

  const parsed = value instanceof Date ? value.getTime() : Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
};

export const activityCutoffIso = (
  now: Date = new Date(),
  windowMinutes: number = ACTIVE_NODE_WINDOW_MINUTES,
): string => new Date(now.getTime() - windowMinutes * 60_000).toISOString();

export const isNodeActive = (
  row: TelemetryActivityRow,
  now: Date = new Date(),
  windowMinutes: number = ACTIVE_NODE_WINDOW_MINUTES,
): boolean => {
  const readingTime = timestampMillis(row.Timestamp);
  if (readingTime === null) return false;

  const ageMillis = now.getTime() - readingTime;
  return ageMillis >= 0 && ageMillis <= windowMinutes * 60_000;
};

export const countActiveNodes = (
  rows: TelemetryActivityRow[],
  now: Date = new Date(),
  windowMinutes: number = ACTIVE_NODE_WINDOW_MINUTES,
): number => {
  const activeNodeIds = new Set<string>();

  for (const row of rows) {
    const nodeId = typeof row.Node_ID === "string" ? row.Node_ID.trim() : "";
    if (nodeId && isNodeActive(row, now, windowMinutes)) {
      activeNodeIds.add(nodeId);
    }
  }

  return activeNodeIds.size;
};
