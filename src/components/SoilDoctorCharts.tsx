import { useState, useEffect, useMemo } from "react";
import { format, subDays } from "date-fns";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend
} from "recharts";
import { Activity, Droplets, Thermometer, Leaf, RefreshCw } from "lucide-react";
import { fetchTelemetry } from "@/lib/telemetry";

type TimeFilter = "7d" | "30d" | "3m";

interface Props {
  nodeId: string;
}

export const SoilDoctorCharts = ({ nodeId }: Props) => {
  const [filter, setFilter] = useState<TimeFilter>("3m");
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const now = new Date();
        let startTime = new Date();
        
        if (filter === "7d") startTime = subDays(now, 7);
        else if (filter === "30d") startTime = subDays(now, 30);
        else if (filter === "3m") startTime = subDays(now, 90);

        const rows = await fetchTelemetry({
          nodeId,
          start: startTime.toISOString(),
          ascending: true,
          limit: 1000,
        });
        setData(rows);
      } catch (err: any) {
        setError(err.message || "Failed to load sensor data.");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [nodeId, filter]);

  const chartData = useMemo(() => {
    return data.map((r) => ({
      ...r,
      formattedTime: format(new Date(r.Timestamp), filter === "7d" ? "MMM dd" : "MMM dd"),
      n: Number(r.Nitrogen_mg_k ?? 0),
      p: Number(r.Phosphorus_m ?? 0),
      k: Number(r.Potassium_mg_ ?? 0),
      moisture: Number(r["Moisture_%"] ?? 0),
      temp: Number(r.Temperature_C ?? 0)
    }));
  }, [data, filter]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground bg-card border border-border rounded-xl">
        <RefreshCw className="h-8 w-8 animate-spin mb-4 text-primary" />
        <p>Loading historical data for {nodeId}...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64 text-destructive border border-destructive/20 bg-destructive/10 rounded-xl">
        <p>{error}</p>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground border border-border bg-card rounded-xl">
        <p>No data found for {nodeId} in the selected time period.</p>
      </div>
    );
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-popover border border-border shadow-lg rounded-lg p-3 text-sm font-mono">
          <p className="text-muted-foreground mb-2 pb-2 border-b border-white/10">{label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center gap-2 mb-1">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
              <span className="capitalize">{entry.name}:</span>
              <span className="font-semibold text-foreground">
                {typeof entry.value === 'number' ? entry.value.toFixed(1) : entry.value}
              </span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Time Filter Controls */}
      <div className="flex items-center justify-between border-b border-border pb-4">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            Sensor Readings: {nodeId}
          </h2>
          <p className="text-sm text-muted-foreground mt-1">Visualizing behavioral changes over time</p>
        </div>
        <div className="flex bg-secondary/50 rounded-lg p-1">
          {(["7d", "30d", "3m"] as TimeFilter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
                filter === f
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {f === "7d" ? "7 Days" : f === "30d" ? "30 Days" : "3 Months"}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* NPK Individual Area Charts */}
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Leaf className="h-4 w-4 text-emerald-500" />
            Nitrogen (N)
          </h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorN" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="formattedTime" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="n" name="Nitrogen" stroke="#10b981" fillOpacity={1} fill="url(#colorN)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Leaf className="h-4 w-4 text-indigo-500" />
            Phosphorus (P)
          </h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorP" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="formattedTime" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="p" name="Phosphorus" stroke="#6366f1" fillOpacity={1} fill="url(#colorP)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Leaf className="h-4 w-4 text-purple-500" />
            Potassium (K)
          </h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorK" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#a855f7" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="formattedTime" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="k" name="Potassium" stroke="#a855f7" fillOpacity={1} fill="url(#colorK)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Droplets className="h-4 w-4 text-blue-500" />
            Soil Moisture
          </h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorM" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="formattedTime" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} domain={[0, 100]} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="moisture" name="Moisture %" stroke="#3b82f6" fillOpacity={1} fill="url(#colorM)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Combined Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        
        {/* NPK Combined Bar Chart */}
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Leaf className="h-4 w-4 text-primary" />
            NPK Combined Analysis
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                <XAxis dataKey="formattedTime" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '12px' }} />
                <Bar dataKey="n" name="Nitrogen" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="p" name="Phosphorus" fill="#6366f1" radius={[4, 4, 0, 0]} />
                <Bar dataKey="k" name="Potassium" fill="#a855f7" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Moisture vs Temperature Area/Line Hybrid */}
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Thermometer className="h-4 w-4 text-orange-500" />
            Climate Analysis (Temp vs Moisture)
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f97316" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                <XAxis dataKey="formattedTime" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis yAxisId="left" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis yAxisId="right" orientation="right" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '12px' }} />
                <Area yAxisId="left" type="monotone" dataKey="temp" name="Temp °C" stroke="#f97316" fillOpacity={1} fill="url(#colorTemp)" />
                <Area yAxisId="right" type="monotone" dataKey="moisture" name="Moisture %" stroke="#3b82f6" fillOpacity={0} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
};
