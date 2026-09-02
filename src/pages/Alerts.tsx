import { useCallback, useEffect, useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { AlertTriangle, AlertCircle, CheckCircle2, Clock, RefreshCw, Settings2 } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ThresholdAlert, evaluateNodeThresholds, latestReadingsByNode, loadAlertThresholds } from "@/lib/alerting";
import { fetchTelemetry } from "@/lib/telemetry";

const severityMap = {
  critical: { icon: AlertCircle, cls: "bg-destructive/10 text-destructive border-destructive/20", label: "Critical" },
  warning: { icon: AlertTriangle, cls: "bg-warning/10 text-warning border-warning/20", label: "Warning" },
};

const formatTime = (timestamp: string | null) => {
  if (!timestamp) return "Latest reading";
  const date = new Date(timestamp);
  return Number.isNaN(date.valueOf()) ? "Latest reading" : date.toLocaleString();
};

const Alerts = () => {
  const [alerts, setAlerts] = useState<ThresholdAlert[]>([]);
  const [nodeCount, setNodeCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshAlerts = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchTelemetry({ limit: 500 });
      const latestReadings = latestReadingsByNode(data);
      const thresholds = loadAlertThresholds();
      const generatedAlerts = latestReadings
        .flatMap((row) => evaluateNodeThresholds(row, thresholds))
        .sort((a, b) => (a.severity === b.severity ? 0 : a.severity === "critical" ? -1 : 1));

      setNodeCount(latestReadings.length);
      setAlerts(generatedAlerts);
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Unable to load current node readings.");
      setAlerts([]);
      setNodeCount(0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshAlerts();
    const refreshForThresholdChange = () => void refreshAlerts();
    window.addEventListener("soilnet:thresholds-updated", refreshForThresholdChange);
    const timer = window.setInterval(() => void refreshAlerts(), 30_000);
    return () => {
      window.removeEventListener("soilnet:thresholds-updated", refreshForThresholdChange);
      window.clearInterval(timer);
    };
  }, [refreshAlerts]);

  return (
    <>
      <PageHeader title="Alerts" subtitle={`${alerts.length} item${alerts.length === 1 ? "" : "s"} need your attention`} />
      <div className="space-y-4 p-6">
        <div className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4 shadow-card sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm font-semibold">Monitoring</p>
            <p className="text-xs text-muted-foreground">We check your readings every 30 seconds and flag anything unusual.</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => void refreshAlerts()} disabled={loading}><RefreshCw className={cn("mr-2 h-4 w-4", loading && "animate-spin")} />Refresh</Button>
            <Button asChild size="sm"><Link to="/settings"><Settings2 className="mr-2 h-4 w-4" />Thresholds</Link></Button>
          </div>
        </div>

        {error && <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">Could not evaluate alerts: {error}</div>}

        {loading ? (
          <div className="rounded-xl border border-border bg-card p-12 text-center text-sm text-muted-foreground">Checking your farm…</div>
        ) : alerts.length === 0 && !error ? (
          <div className="rounded-xl border border-border bg-card p-12 text-center">
            <CheckCircle2 className="mx-auto h-10 w-10 text-primary" />
            <h2 className="mt-4 font-semibold">Everything looks good</h2>
            <p className="mt-1 text-sm text-muted-foreground">The latest readings from all {nodeCount} sensors are within normal limits.</p>
          </div>
        ) : (
          <Accordion type="multiple" className="space-y-3">
            {alerts.map((alert) => {
              const severity = severityMap[alert.severity];
              const Icon = severity.icon;
              return (
                <AccordionItem key={alert.id} value={alert.id} className="overflow-hidden rounded-xl border border-border bg-card shadow-card transition-shadow hover:shadow-elevated">
                  <AccordionTrigger className="group px-5 py-4 hover:no-underline">
                    <div className="flex flex-1 items-center gap-4 text-left">
                      <div className={cn("flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border", severity.cls)}><Icon className="h-5 w-5" /></div>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className={cn("rounded border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider", severity.cls)}>{severity.label}</span>
                          <span className="text-xs font-semibold text-primary">{alert.nodeId}</span>
                          <span className="inline-flex items-center gap-1 text-xs text-muted-foreground"><Clock className="h-3 w-3" />{formatTime(alert.timestamp)}</span>
                        </div>
                        <h3 className="mt-1 font-semibold transition-colors group-hover:text-primary">{alert.title}</h3>
                        <p className="mt-0.5 truncate text-sm text-muted-foreground">{alert.message}</p>
                      </div>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="px-5 pb-5">
                    <div className="rounded-xl border border-primary/20 bg-primary-soft/40 p-4">
                      <p className="text-sm font-semibold text-primary">Recommended next step</p>
                      <p className="mt-1 text-sm leading-6 text-foreground/90">{alert.recommendation}</p>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              );
            })}
          </Accordion>
        )}
      </div>
    </>
  );
};

export default Alerts;
