export type NodeStatus = "online" | "offline";

export interface SensorNode {
  id: string;
  farm: string;
  location: string;
  lat: number;
  lng: number;
  status: NodeStatus;
  metrics: {
    nitrogen: number;
    phosphorus: number;
    potassium: number;
    moisture: number;
    humidity: number;
    temperature: number;
  };
  lastReading: string;
}

export const nodes: SensorNode[] = [
  { id: "NODE_01", farm: "Green Valley Farm", location: "North Field A", lat: 36.7783, lng: -119.4179, status: "online",
    metrics: { nitrogen: 18, phosphorus: 42, potassium: 180, moisture: 32, humidity: 64, temperature: 24.5 }, lastReading: "2 min ago" },
  { id: "NODE_02", farm: "Green Valley Farm", location: "North Field B", lat: 36.7790, lng: -119.4150, status: "online",
    metrics: { nitrogen: 35, phosphorus: 55, potassium: 210, moisture: 48, humidity: 70, temperature: 23.8 }, lastReading: "1 min ago" },
  { id: "NODE_03", farm: "Sunrise Acres", location: "East Plot", lat: 36.7820, lng: -119.4100, status: "offline",
    metrics: { nitrogen: 22, phosphorus: 38, potassium: 165, moisture: 28, humidity: 58, temperature: 26.1 }, lastReading: "3 hr ago" },
  { id: "NODE_04", farm: "Sunrise Acres", location: "South Plot", lat: 36.7750, lng: -119.4200, status: "online",
    metrics: { nitrogen: 41, phosphorus: 60, potassium: 225, moisture: 52, humidity: 72, temperature: 22.9 }, lastReading: "30 sec ago" },
  { id: "NODE_05", farm: "Orchard Hills", location: "Greenhouse 1", lat: 36.7800, lng: -119.4250, status: "online",
    metrics: { nitrogen: 28, phosphorus: 48, potassium: 195, moisture: 40, humidity: 68, temperature: 25.3 }, lastReading: "1 min ago" },
  { id: "NODE_06", farm: "Orchard Hills", location: "Greenhouse 2", lat: 36.7810, lng: -119.4220, status: "online",
    metrics: { nitrogen: 12, phosphorus: 30, potassium: 140, moisture: 22, humidity: 55, temperature: 27.4 }, lastReading: "4 min ago" },
];

export interface TelemetryLog {
  nodeId: string;
  n: number; p: number; k: number;
  moisture: number; humidity: number; temp: number;
  lat: number; lng: number;
  timestamp: string;
}

export const telemetryLogs: TelemetryLog[] = Array.from({ length: 48 }, (_, i) => {
  const node = nodes[i % nodes.length];
  const d = new Date(Date.now() - i * 1000 * 60 * 30);
  return {
    nodeId: node.id,
    n: Math.round(node.metrics.nitrogen + (Math.random() - 0.5) * 6),
    p: Math.round(node.metrics.phosphorus + (Math.random() - 0.5) * 8),
    k: Math.round(node.metrics.potassium + (Math.random() - 0.5) * 20),
    moisture: Math.round(node.metrics.moisture + (Math.random() - 0.5) * 8),
    humidity: Math.round(node.metrics.humidity + (Math.random() - 0.5) * 6),
    temp: +(node.metrics.temperature + (Math.random() - 0.5) * 2).toFixed(1),
    lat: node.lat, lng: node.lng,
    timestamp: d.toISOString(),
  };
});

export interface Alert {
  id: string;
  nodeId: string;
  severity: "critical" | "warning" | "info";
  title: string;
  message: string;
  timestamp: string;
  prescription: string;
}

export const alerts: Alert[] = [
  {
    id: "a1", nodeId: "NODE_01", severity: "warning",
    title: "Nitrogen level low on NODE_01",
    message: "Current N reading is 18 ppm — below the recommended 25 ppm threshold for leafy crops.",
    timestamp: "2 min ago",
    prescription: `### Recommended Action
**Apply Urea (46-0-0)** at a rate of **40 kg/hectare**.

#### Application Plan
- **Method:** Side-dressing along plant rows
- **Timing:** Apply within next 48 hours, ideally early morning
- **Irrigation:** Water immediately after application (15–20 mm)

#### Expected Outcome
Nitrogen levels should rise to **30–35 ppm** within 5–7 days. Re-test after one week.

> ⚠️ Avoid over-application — excess N causes leaf burn and groundwater contamination.`
  },
  {
    id: "a2", nodeId: "NODE_06", severity: "critical",
    title: "Critical moisture deficit on NODE_06",
    message: "Soil moisture at 22% — crops at risk of wilting within 12 hours.",
    timestamp: "8 min ago",
    prescription: `### Immediate Irrigation Required
**Activate drip irrigation** for **45 minutes** at zone 6.

#### Action Steps
1. Run irrigation cycle now (target: **30 mm** depth)
2. Re-check moisture after 2 hours — should reach **40–45%**
3. Schedule recurring 20-min cycles every 6 hours for next 48 hours

#### Crop Protection
- Apply mulch to retain moisture
- Inspect for early wilting signs on leaves`
  },
  {
    id: "a3", nodeId: "NODE_03", severity: "critical",
    title: "NODE_03 offline — last seen 3 hours ago",
    message: "No telemetry received. Possible power, connectivity, or hardware failure.",
    timestamp: "3 hr ago",
    prescription: `### Diagnostic Checklist
1. **Verify power source** — check solar panel & battery voltage (>3.6V)
2. **Inspect LoRa antenna** for physical damage
3. **Check gateway logs** for last heartbeat
4. **Field visit** if remote restart fails

#### Backup Plan
Use **NODE_04** readings as a proxy for South Plot until restored.`
  },
  {
    id: "a4", nodeId: "NODE_02", severity: "info",
    title: "Optimal growth conditions on NODE_02",
    message: "All NPK and environmental metrics within ideal range. Continue current regime.",
    timestamp: "1 hr ago",
    prescription: `### Maintain Current Practices
All metrics nominal:
- **N:** 35 ppm ✓
- **P:** 55 ppm ✓
- **K:** 210 ppm ✓
- **Moisture:** 48% ✓

No action needed. Next scheduled review in **7 days**.`
  },
];
