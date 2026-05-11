import { useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { MapPreview } from "@/components/MapPreview";
import { nodes } from "@/lib/mockData";
import {
  Radio, Wifi, WifiOff, Leaf, FlaskConical, Droplets, CloudRain, Thermometer, Sprout,
} from "lucide-react";
import { cn } from "@/lib/utils";

const Dashboard = () => {
  const [selected, setSelected] = useState(nodes[0].id);
  const node = nodes.find((n) => n.id === selected) ?? nodes[0];
  const online = nodes.filter((n) => n.status === "online").length;

  return (
    <>
      <PageHeader title="Dashboard" subtitle="Real-time field intelligence across all nodes" />
      <div className="p-6 space-y-6">
        {/* Top metrics */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <MetricCard icon={Radio} label="Total Nodes" value={nodes.length} trend="Across 3 farms" tone="info" />
          <MetricCard icon={Wifi} label="Online" value={online} trend="Streaming live" tone="success" />
          <MetricCard icon={WifiOff} label="Offline" value={nodes.length - online} trend="Needs attention" tone="destructive" />
        </div>

        {/* Node parameters */}
        <section className="bg-card border border-border rounded-xl shadow-card overflow-hidden">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-5 border-b border-border">
            <div>
              <h2 className="text-lg font-semibold">Node Parameters</h2>
              <p className="text-sm text-muted-foreground">Live readings · {node.farm} — {node.location}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {nodes.map((n) => (
                <button
                  key={n.id}
                  onClick={() => setSelected(n.id)}
                  className={cn(
                    "px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all",
                    selected === n.id
                      ? "bg-primary text-primary-foreground border-primary shadow-md"
                      : "bg-secondary text-secondary-foreground border-border hover:border-primary/50"
                  )}
                >
                  {n.id}
                </button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4 p-5">
            <MetricCard icon={Leaf} label="Nitrogen" value={node.metrics.nitrogen} unit="ppm" tone="success" />
            <MetricCard icon={FlaskConical} label="Phosphorus" value={node.metrics.phosphorus} unit="ppm" tone="info" />
            <MetricCard icon={Sprout} label="Potassium" value={node.metrics.potassium} unit="ppm" tone="success" />
            <MetricCard icon={Droplets} label="Soil Moisture" value={node.metrics.moisture} unit="%" tone="info" />
            <MetricCard icon={CloudRain} label="Humidity" value={node.metrics.humidity} unit="%" tone="default" />
            <MetricCard icon={Thermometer} label="Temperature" value={node.metrics.temperature} unit="°C" tone="warning" />
          </div>
        </section>

        {/* Map */}
        <section className="bg-card border border-border rounded-xl shadow-card p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold">Field Map</h2>
              <p className="text-sm text-muted-foreground">Live node locations — click a marker to inspect</p>
            </div>
          </div>
          <MapPreview nodes={nodes} selectedId={selected} onSelect={setSelected} height="h-96" />
        </section>
      </div>
    </>
  );
};

export default Dashboard;
