export type EvaluateResponse = {
  summary: {
    crop_profile: string;
    field_health_score: number;
    health_band: string;
    requires_intervention: boolean;
    critical_count: number;
    warning_count: number;
  };
  flags: Array<Record<string, any>>;
  llm_alert_block?: string;
};

export async function evaluateTelemetry(telemetry: Record<string, any>, crop = "maize_corn") {
  const res = await fetch("http://localhost:8000/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ telemetry, crop }),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Evaluator API error: ${res.status} ${txt}`);
  }
  const data: EvaluateResponse = await res.json();
  return data;
}
