export const FIELD_REPORT_TTL_MS = 30 * 60 * 1000;

// Versioned so reports generated under an older prompt are not reused.
const FIELD_REPORT_STORAGE_PREFIX = "soil-doctor-field-report:v2:";

export type CachedFieldReport = {
  nodeId: string;
  report: string;
  generatedAt: number;
  expiresAt: number;
};

const storageKey = (nodeId: string) => (
  `${FIELD_REPORT_STORAGE_PREFIX}${nodeId.trim().toUpperCase()}`
);

export function readFieldReport(
  nodeId: string,
  now: number = Date.now(),
): CachedFieldReport | null {
  if (typeof window === "undefined") return null;

  const key = storageKey(nodeId);
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return null;
    const cached = JSON.parse(raw) as Partial<CachedFieldReport>;
    const valid = (
      cached.nodeId === nodeId.trim().toUpperCase()
      && typeof cached.report === "string"
      && cached.report.trim().length > 0
      && typeof cached.generatedAt === "number"
      && typeof cached.expiresAt === "number"
      && cached.expiresAt > now
    );
    if (!valid) {
      window.localStorage.removeItem(key);
      return null;
    }
    return cached as CachedFieldReport;
  } catch {
    window.localStorage.removeItem(key);
    return null;
  }
}

export function saveFieldReport(
  nodeId: string,
  report: string,
  generatedAt: number = Date.now(),
): CachedFieldReport {
  const cached: CachedFieldReport = {
    nodeId: nodeId.trim().toUpperCase(),
    report: report.trim(),
    generatedAt,
    expiresAt: generatedAt + FIELD_REPORT_TTL_MS,
  };
  if (typeof window !== "undefined") {
    window.localStorage.setItem(storageKey(nodeId), JSON.stringify(cached));
  }
  return cached;
}
