import { useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { telemetryLogs } from "@/lib/mockData";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Download, Calendar, ChevronLeft, ChevronRight } from "lucide-react";
import { toast } from "sonner";

const PAGE_SIZE = 12;

const DataTable = () => {
  const [page, setPage] = useState(0);
  const total = Math.ceil(telemetryLogs.length / PAGE_SIZE);
  const rows = telemetryLogs.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const downloadCSV = () => {
    const header = "Node ID,N,P,K,Moisture,Humidity,Temp,Lat,Lng,Timestamp\n";
    const body = telemetryLogs.map(r =>
      `${r.nodeId},${r.n},${r.p},${r.k},${r.moisture},${r.humidity},${r.temp},${r.lat},${r.lng},${r.timestamp}`
    ).join("\n");
    const blob = new Blob([header + body], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "telemetry.csv"; a.click();
    URL.revokeObjectURL(url);
    toast.success("CSV downloaded");
  };

  return (
    <>
      <PageHeader title="Data Table" subtitle="Historical telemetry logs across all nodes" />
      <div className="p-6 space-y-4">
        <div className="bg-card border border-border rounded-xl shadow-card p-4 flex flex-col sm:flex-row gap-3 items-stretch sm:items-center">
          <div className="flex items-center gap-2 flex-1">
            <div className="relative flex-1">
              <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input type="date" className="pl-9" defaultValue="2026-04-23" />
            </div>
            <span className="text-sm text-muted-foreground">to</span>
            <div className="relative flex-1">
              <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input type="date" className="pl-9" defaultValue="2026-04-30" />
            </div>
          </div>
          <Button variant="outline" onClick={downloadCSV} className="gap-2">
            <Download className="h-4 w-4" /> Download CSV
          </Button>
        </div>

        <div className="bg-card border border-border rounded-xl shadow-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-secondary/60">
                <tr className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                  {["Node ID","N","P","K","Moisture","Humidity","Temp","Lat","Lng","Timestamp"].map(h => (
                    <th key={h} className="px-4 py-3 font-semibold">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i} className="border-t border-border hover:bg-secondary/40 transition-colors">
                    <td className="px-4 py-3 font-semibold text-primary">{r.nodeId}</td>
                    <td className="px-4 py-3">{r.n}</td>
                    <td className="px-4 py-3">{r.p}</td>
                    <td className="px-4 py-3">{r.k}</td>
                    <td className="px-4 py-3">{r.moisture}%</td>
                    <td className="px-4 py-3">{r.humidity}%</td>
                    <td className="px-4 py-3">{r.temp}°C</td>
                    <td className="px-4 py-3 text-muted-foreground">{r.lat.toFixed(4)}</td>
                    <td className="px-4 py-3 text-muted-foreground">{r.lng.toFixed(4)}</td>
                    <td className="px-4 py-3 text-muted-foreground whitespace-nowrap">{new Date(r.timestamp).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between p-4 border-t border-border">
            <p className="text-sm text-muted-foreground">
              Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, telemetryLogs.length)} of {telemetryLogs.length}
            </p>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" disabled={page === 0} onClick={() => setPage(p => p - 1)}>
                <ChevronLeft className="h-4 w-4" /> Prev
              </Button>
              <Button variant="outline" size="sm" disabled={page >= total - 1} onClick={() => setPage(p => p + 1)}>
                Next <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default DataTable;
