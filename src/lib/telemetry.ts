import { supabase } from "@/lib/supabase";

export const HARDWARE_NODE_IDS = ["NODE_01", "NODE_02"] as const;
export const SIMULATOR_NODE_IDS = ["NODE_03", "NODE_04", "NODE_05", "NODE_06"] as const;

const hardwareNodeIds = new Set<string>(HARDWARE_NODE_IDS);
const simulatorNodeIds = new Set<string>(SIMULATOR_NODE_IDS);
const FIREBASE_PUSH_ALPHABET = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
const FIREBASE_LOG_LIMIT = 1000;
const HARDWARE_FIREBASE_URL = (
  import.meta.env.VITE_HARDWARE_FIREBASE_URL
  || (process.env.VITE_HARDWARE_FIREBASE_URL as string)
  || "https://capstone-2e26e-default-rtdb.firebaseio.com"
).replace(/\/$/, "");

export type TelemetrySource = "hardware" | "simulator";

export type TelemetryRow = Record<string, unknown> & {
  Node_ID: string;
  Timestamp: string;
  Data_Source: TelemetrySource;
};

export type TelemetryQueryOptions = {
  start?: string | null;
  end?: string | null;
  nodeId?: string | null;
  limit?: number;
  ascending?: boolean;
};

const normalizeNodeId = (value: unknown) => String(value ?? "").trim().toUpperCase();

const timestampValue = (row: Pick<TelemetryRow, "Timestamp">) => {
  const value = Date.parse(row.Timestamp);
  return Number.isFinite(value) ? value : 0;
};

export const normalizeSimulatorTelemetry = (row: Record<string, unknown>): TelemetryRow => ({
  ...row,
  Node_ID: normalizeNodeId(row.Node_ID),
  Timestamp: String(row.Timestamp ?? ""),
  Data_Source: "simulator",
});

export const firebasePushTimestamp = (pushId: string): string => {
  if (pushId.length < 8) return "";

  let timestamp = 0;
  for (let index = 0; index < 8; index += 1) {
    const value = FIREBASE_PUSH_ALPHABET.indexOf(pushId[index]);
    if (value < 0) return "";
    timestamp = timestamp * 64 + value;
  }

  const parsed = new Date(timestamp);
  return Number.isNaN(parsed.valueOf()) ? "" : parsed.toISOString();
};

export const normalizeHardwareTelemetry = (
  row: Record<string, unknown>,
  pushId: string,
): TelemetryRow => ({
  Node_ID: normalizeNodeId(row.node_id),
  // The INO sends millis()/1000, which is device uptime. Firebase push IDs
  // carry the server-side creation time needed by activity and trend views.
  Timestamp: firebasePushTimestamp(pushId),
  Nitrogen_mg_k: row.nitrogen,
  Phosphorus_m: row.phosphorus,
  Potassium_mg_: row.potassium,
  "Moisture_%": row.moisture,
  Temperature_C: row.temp,
  "Humidity_%": row.humidity,
  Soil_pH: row.ph,
  Latitude: Number(row.latitude),
  Longitude: Number(row.longitude),
  Altitude_m: Number(row.altitude),
  Satellites: row.satellites,
  Season: row.season,
  GPS_Source: row.gps_source,
  Device_Uptime_Seconds: row.timestamp,
  Data_Source: "hardware",
});

const fetchSimulatorTelemetry = async (options: TelemetryQueryOptions) => {
  const requestedNode = normalizeNodeId(options.nodeId);
  if (requestedNode && !simulatorNodeIds.has(requestedNode)) return [];

  let query = supabase
    .from("capstone_dataset")
    .select("*")
    .in("Node_ID", [...SIMULATOR_NODE_IDS]);

  if (requestedNode) query = query.eq("Node_ID", requestedNode);
  if (options.start) query = query.gte("Timestamp", options.start);
  if (options.end) query = query.lte("Timestamp", options.end);
  query = query.order("Timestamp", { ascending: options.ascending ?? false });
  if (options.limit) query = query.limit(options.limit);

  const { data, error } = await query;
  if (error) throw error;
  return (Array.isArray(data) ? data : []).map(normalizeSimulatorTelemetry);
};

const fetchHardwareTelemetry = async (options: TelemetryQueryOptions) => {
  const requestedNode = normalizeNodeId(options.nodeId);
  if (requestedNode && !hardwareNodeIds.has(requestedNode)) return [];

  // Firebase filters this shared log before the browser can separate node IDs.
  // Pull a broad window for mixed-node views, and oversample focused history
  // requests so readings from the other physical node do not crowd them out.
  const requestedLimit = requestedNode
    ? Math.max((options.limit ?? 100) * 2, 100)
    : FIREBASE_LOG_LIMIT;
  const params = new URLSearchParams({
    orderBy: '"$key"',
    limitToLast: String(Math.min(requestedLimit, FIREBASE_LOG_LIMIT)),
  });
  const response = await fetch(
    `${HARDWARE_FIREBASE_URL}/readings/log.json?${params.toString()}`,
  );
  if (!response.ok) {
    throw new Error(`Hardware Firebase request failed with HTTP ${response.status}.`);
  }

  const payload: unknown = await response.json();
  if (payload == null) return [];
  if (typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("Hardware Firebase returned an unexpected telemetry payload.");
  }

  const startTime = options.start ? Date.parse(options.start) : null;
  const endTime = options.end ? Date.parse(options.end) : null;
  return Object.entries(payload as Record<string, unknown>)
    .filter((entry): entry is [string, Record<string, unknown>] => (
      typeof entry[1] === "object" && entry[1] !== null && !Array.isArray(entry[1])
    ))
    .map(([pushId, row]) => normalizeHardwareTelemetry(row, pushId))
    .filter((row) => !requestedNode || row.Node_ID === requestedNode)
    .filter((row) => hardwareNodeIds.has(row.Node_ID))
    .filter((row) => {
      const timestamp = Date.parse(row.Timestamp);
      if (!Number.isFinite(timestamp)) return false;
      if (startTime !== null && Number.isFinite(startTime) && timestamp < startTime) return false;
      if (endTime !== null && Number.isFinite(endTime) && timestamp > endTime) return false;
      return true;
    });
};

export async function fetchTelemetry(options: TelemetryQueryOptions = {}): Promise<TelemetryRow[]> {
  const requestedNode = normalizeNodeId(options.nodeId);
  const requests: Array<Promise<TelemetryRow[]>> = [];

  if (!requestedNode || hardwareNodeIds.has(requestedNode)) {
    requests.push(fetchHardwareTelemetry(options));
  }
  if (!requestedNode || simulatorNodeIds.has(requestedNode)) {
    requests.push(fetchSimulatorTelemetry(options));
  }
  if (requests.length === 0) return [];

  const results = await Promise.allSettled(requests);
  const rows = results.flatMap((result) => result.status === "fulfilled" ? result.value : []);

  if (rows.length === 0) {
    const failures = results.filter((result) => result.status === "rejected");
    if (failures.length === results.length) {
      const reason = failures[0]?.status === "rejected" ? failures[0].reason : null;
      throw reason instanceof Error ? reason : new Error("Unable to load telemetry data.");
    }
  }

  rows.sort((left, right) => {
    const delta = timestampValue(left) - timestampValue(right);
    return options.ascending ? delta : -delta;
  });

  // `limit` is applied independently to each source. Do not slice the merged
  // result again: a busy hardware feed would otherwise hide all simulator
  // nodes simply because its records are newer.
  return rows;
}

export function latestTelemetryByNode(rows: TelemetryRow[]): TelemetryRow[] {
  const latest = new Map<string, TelemetryRow>();
  for (const row of rows) {
    if (row.Node_ID && !latest.has(row.Node_ID)) latest.set(row.Node_ID, row);
  }
  return [...latest.values()].sort((left, right) => left.Node_ID.localeCompare(right.Node_ID));
}
