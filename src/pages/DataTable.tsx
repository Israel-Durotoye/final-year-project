import { useEffect, useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Calendar } from "lucide-react";
import { toast } from "sonner";
import { fetchTelemetry as loadTelemetry, TelemetryRow } from "@/lib/telemetry";

const DataTable = () => {
  const [telemetry, setTelemetry] = useState<TelemetryRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startDate, setStartDate] = useState<string | null>(null);
  const [endDate, setEndDate] = useState<string | null>(null);

  const fetchTelemetry = async (opts?: { start?: string | null; end?: string | null }) => {
    setLoading(true);
    setError(null);
    try {
      const data = await loadTelemetry({
        start: opts?.start ? new Date(opts.start).toISOString() : null,
        end: opts?.end ? new Date(opts.end).toISOString() : null,
      });
      setTelemetry(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
      setTelemetry([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Load unfiltered on mount
    fetchTelemetry();
  }, []);

  const handleFilterApply = () => {
    fetchTelemetry({ start: startDate, end: endDate });
  };

  const handleClear = () => {
    setStartDate(null);
    setEndDate(null);
    fetchTelemetry();
  };

  const [training, setTraining] = useState(false);

  const handleTrainModel = async () => {
    setTraining(true);
    toast.info("LSTM anomaly model training started in background.");
    try {
      const response = await fetch("http://localhost:8000/api/v1/ml/train-anomaly-model", {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      toast.success("Training job submitted successfully.");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      toast.error(`Training failed to start: ${message}`);
    } finally {
      // We set training to false even if it runs in background just to unlock the button,
      // or we can keep it true for a few seconds.
      setTimeout(() => setTraining(false), 5000);
    }
  };

  return (
    <>
      <PageHeader title="Data Table" subtitle="Sensor Readings Log" />
      <div className="p-6 space-y-4">
        <div className="bg-card border border-border rounded-xl shadow-card p-4 flex flex-col sm:flex-row gap-3 items-stretch sm:items-center">
          <div className="flex items-center gap-2 flex-1">
            <div className="relative flex-1">
              <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input type="date" className="pl-9" value={startDate ?? ""} onChange={(e) => setStartDate(e.target.value || null)} />
            </div>
            <span className="text-sm text-muted-foreground">to</span>
            <div className="relative flex-1">
              <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input type="date" className="pl-9" value={endDate ?? ""} onChange={(e) => setEndDate(e.target.value || null)} />
            </div>
            <Button variant="outline" onClick={handleClear} className="ml-2">Clear</Button>
            <Button onClick={handleFilterApply} className="ml-2">Apply</Button>
            <Button variant="secondary" onClick={handleTrainModel} disabled={training} className="ml-auto">
              {training ? "Training..." : "Train Anomaly Model"}
            </Button>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl shadow-card overflow-hidden">
          <div className="overflow-x-auto">
            {loading ? (
              <div className="p-8 text-center">Loading records…</div>
            ) : telemetry.length === 0 ? (
              <div className="p-12 text-center text-muted-foreground">No records found for this period.</div>
            ) : (
              <table className="w-full text-sm">
                <thead className="bg-secondary/60">
                  <tr className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                    {[
                      "SENSOR","SOURCE","N","P","K","MOISTURE","TEMP","HUMIDITY","LAT","LNG","ALTITUDE","SATS","TIMESTAMP"
                    ].map(h => (
                      <th key={h} className="px-4 py-3 font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {telemetry.map((r, i) => (
                    <tr key={i} className="border-t border-border hover:bg-secondary/40 transition-colors">
                      <td className="px-4 py-3 font-semibold text-primary">{r.Node_ID}</td>
                      <td className="px-4 py-3 text-xs font-semibold uppercase text-muted-foreground">{r.Data_Source}</td>
                      <td className="px-4 py-3">{r.Nitrogen_mg_k ?? "-"}</td>
                      <td className="px-4 py-3">{r.Phosphorus_m ?? "-"}</td>
                      <td className="px-4 py-3">{r.Potassium_mg_ ?? "-"}</td>
                      <td className="px-4 py-3">{typeof r["Moisture_%"] === 'number' ? `${r["Moisture_%"]}%` : (r["Moisture_%"] ?? "-")}</td>
                      <td className="px-4 py-3">{typeof r.Temperature_C === 'number' ? `${r.Temperature_C}°C` : (r.Temperature_C ?? "-")}</td>
                      <td className="px-4 py-3">{typeof r["Humidity_%"] === 'number' ? `${r["Humidity_%"]}%` : (r["Humidity_%"] ?? "-")}</td>
                      <td className="px-4 py-3 text-muted-foreground">{typeof r.Latitude === 'number' ? r.Latitude.toFixed(4) : (r.Latitude ?? "-")}</td>
                      <td className="px-4 py-3 text-muted-foreground">{typeof r.Longitude === 'number' ? r.Longitude.toFixed(4) : (r.Longitude ?? "-")}</td>
                      <td className="px-4 py-3">{r.Altitude_m ?? "-"}</td>
                      <td className="px-4 py-3">{r.Satellites ?? "-"}</td>
                      <td className="px-4 py-3 text-muted-foreground whitespace-nowrap">{r.Timestamp ? new Date(r.Timestamp).toLocaleString() : "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default DataTable;
