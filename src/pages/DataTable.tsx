import { useEffect, useState } from "react";
import { createClient } from "@supabase/supabase-js";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Download, Calendar, ChevronLeft, ChevronRight } from "lucide-react";
import { toast } from "sonner";
// Initialize Supabase client using Vite env vars (or fall back to process.env)
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || (process.env.VITE_SUPABASE_URL as string) || "";
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || (process.env.VITE_SUPABASE_ANON_KEY as string) || "";
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

const DataTable = () => {
  const [telemetry, setTelemetry] = useState<Array<any>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startDate, setStartDate] = useState<string | null>(null);
  const [endDate, setEndDate] = useState<string | null>(null);

  const fetchTelemetry = async (opts?: { start?: string | null; end?: string | null }) => {
    setLoading(true);
    setError(null);
    try {
      let query: any = supabase
        .from("sensor_telemetry")
        .select("*")
        .order("timestamp_utc", { ascending: false });

      if (opts?.start) {
        const startISO = new Date(opts.start).toISOString();
        query = query.gte("timestamp_utc", startISO);
      }
      if (opts?.end) {
        // include end of day by default if only date provided
        const endISO = new Date(opts.end).toISOString();
        query = query.lte("timestamp_utc", endISO);
      }

      const { data, error: sbError } = await query;
      if (sbError) throw sbError;

      setTelemetry(Array.isArray(data) ? data : []);
    } catch (err: any) {
      setError(err.message || String(err));
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

  return (
    <>
      <PageHeader title="Data Table" subtitle="Historical telemetry logs across all nodes" />
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
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl shadow-card overflow-hidden">
          <div className="overflow-x-auto">
            {loading ? (
              <div className="p-8 text-center">Loading telemetry…</div>
            ) : telemetry.length === 0 ? (
              <div className="p-12 text-center text-muted-foreground">No historical telemetry records found.</div>
            ) : (
              <table className="w-full text-sm">
                <thead className="bg-secondary/60">
                  <tr className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                    {[
                      "NODE ID","N","P","K","MOISTURE","HUMIDITY","TEMP","pH","BATTERY","LAT","LNG","STATUS","TIMESTAMP"
                    ].map(h => (
                      <th key={h} className="px-4 py-3 font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {telemetry.map((r, i) => (
                    <tr key={i} className="border-t border-border hover:bg-secondary/40 transition-colors">
                      <td className="px-4 py-3 font-semibold text-primary">{r.node_id}</td>
                      <td className="px-4 py-3">{r.nitrogen_ppm ?? "-"}</td>
                      <td className="px-4 py-3">{r.phosphorus_ppm ?? "-"}</td>
                      <td className="px-4 py-3">{r.potassium_ppm ?? "-"}</td>
                      <td className="px-4 py-3">{typeof r.soil_moisture === 'number' ? `${r.soil_moisture}%` : (r.soil_moisture ?? "-")}</td>
                      <td className="px-4 py-3">{typeof r.humidity === 'number' ? `${r.humidity}%` : (r.humidity ?? "-")}</td>
                      <td className="px-4 py-3">{typeof r.soil_temperature_c === 'number' ? `${r.soil_temperature_c}°C` : (r.soil_temperature_c ?? "-")}</td>
                      <td className="px-4 py-3">{r.ph ?? "-"}</td>
                      <td className="px-4 py-3">{typeof r.battery_voltage === 'number' ? `${r.battery_voltage}V` : (r.battery_voltage ?? "-")}</td>
                      <td className="px-4 py-3 text-muted-foreground">{typeof r.latitude === 'number' ? r.latitude.toFixed(4) : (r.latitude ?? "-")}</td>
                      <td className="px-4 py-3 text-muted-foreground">{typeof r.longitude === 'number' ? r.longitude.toFixed(4) : (r.longitude ?? "-")}</td>
                      <td className="px-4 py-3">
                        {r.communication_ok ? (
                          <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-green-100 text-green-800 text-xs">Online</span>
                        ) : (
                          <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-red-100 text-red-800 text-xs">Offline</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-muted-foreground whitespace-nowrap">{r.timestamp_utc ? new Date(r.timestamp_utc).toLocaleString() : "-"}</td>
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
