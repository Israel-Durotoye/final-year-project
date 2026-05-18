import { useMemo } from "react";
import { format } from "date-fns";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  Legend
} from "recharts";

interface Props {
  rows: any[];
  selectedId: string | undefined;
}

export const TrendCharts = ({ rows, selectedId }: Props) => {
  const chartData = useMemo(() => {
    if (!selectedId) return [];
    
    // Filter rows by selected node and sort chronologically (oldest to newest)
    const nodeRows = rows.filter((r) => r.node_id === selectedId);
    const sorted = [...nodeRows].sort(
      (a, b) => new Date(a.timestamp_utc).getTime() - new Date(b.timestamp_utc).getTime()
    );

    return sorted.map((r) => ({
      ...r,
      formattedTime: format(new Date(r.timestamp_utc), "HH:mm"),
      formattedDate: format(new Date(r.timestamp_utc), "MMM dd"),
      n: Number(r.nitrogen_ppm ?? r.nitrogen ?? 0),
      p: Number(r.phosphorus_ppm ?? r.phosphorus ?? 0),
      k: Number(r.potassium_ppm ?? r.potassium ?? 0),
      moisture: Number(r.soil_moisture ?? r.moisture ?? 0),
      humidity: Number(r.humidity ?? 0),
      temp: Number(r.soil_temperature_c ?? r.temperature ?? 0)
    }));
  }, [rows, selectedId]);

  if (!selectedId || chartData.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center border border-dashed border-white/10 rounded-xl bg-black/10">
        <p className="text-sm font-mono text-muted-foreground italic">No trend data available for selected node.</p>
      </div>
    );
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-popover border border-border shadow-lg rounded-lg p-3 text-sm font-mono">
          <p className="text-muted-foreground mb-2 pb-2 border-b border-white/10">{payload[0].payload.formattedDate} - {label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
              <span className="text-muted-foreground uppercase">{entry.name}:</span>
              <span className="font-bold text-foreground">{entry.value}</span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* NPK Area Chart */}
      <div className="glass-card border border-border rounded-xl shadow-card p-5">
        <h3 className="text-[11px] uppercase tracking-widest text-muted-foreground font-medium mb-4">Macronutrients (NPK)</h3>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorN" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(158 64% 52%)" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="hsl(158 64% 52%)" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorP" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(210 90% 55%)" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="hsl(210 90% 55%)" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorK" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(38 92% 50%)" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="hsl(38 92% 50%)" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
              <XAxis dataKey="formattedTime" stroke="hsl(var(--muted-foreground))" fontSize={10} tickLine={false} axisLine={false} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={10} tickLine={false} axisLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
              <Area type="monotone" dataKey="n" name="Nitrogen" stroke="hsl(158 64% 52%)" fillOpacity={1} fill="url(#colorN)" />
              <Area type="monotone" dataKey="p" name="Phosphorus" stroke="hsl(210 90% 55%)" fillOpacity={1} fill="url(#colorP)" />
              <Area type="monotone" dataKey="k" name="Potassium" stroke="hsl(38 92% 50%)" fillOpacity={1} fill="url(#colorK)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Moisture & Humidity Line Chart */}
      <div className="glass-card border border-border rounded-xl shadow-card p-5">
        <h3 className="text-[11px] uppercase tracking-widest text-muted-foreground font-medium mb-4">Hydration Trends</h3>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
              <XAxis dataKey="formattedTime" stroke="hsl(var(--muted-foreground))" fontSize={10} tickLine={false} axisLine={false} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={10} tickLine={false} axisLine={false} domain={[0, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
              <Line type="monotone" dataKey="moisture" name="Soil Moisture (%)" stroke="hsl(210 90% 55%)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
              <Line type="monotone" dataKey="humidity" name="Humidity (%)" stroke="hsl(120 10% 60%)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Temperature Chart */}
      <div className="glass-card border border-border rounded-xl shadow-card p-5">
        <h3 className="text-[11px] uppercase tracking-widest text-muted-foreground font-medium mb-4">Soil Temperature</h3>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="hsl(15 63% 60%)" stopOpacity={0.8}/>
                  <stop offset="100%" stopColor="hsl(15 63% 60%)" stopOpacity={0.2}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
              <XAxis dataKey="formattedTime" stroke="hsl(var(--muted-foreground))" fontSize={10} tickLine={false} axisLine={false} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={10} tickLine={false} axisLine={false} domain={['dataMin - 5', 'dataMax + 5']} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
              <Bar dataKey="temp" name="Temperature (°C)" fill="url(#colorTemp)" radius={[4, 4, 0, 0]} barSize={8} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
