import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  icon: LucideIcon;
  label: string;
  value: string | number;
  unit?: string;
  trend?: string;
  tone?: "default" | "success" | "warning" | "destructive" | "info";
}

const toneMap = {
  default: "bg-secondary text-secondary-foreground",
  success: "bg-primary-soft text-primary",
  warning: "bg-warning/15 text-warning",
  destructive: "bg-destructive/15 text-destructive",
  info: "bg-info/15 text-info",
};

export const MetricCard = ({ icon: Icon, label, value, unit, trend, tone = "default" }: Props) => (
  <div className="glass-card border border-border rounded-xl p-4 shadow-card hover:shadow-elevated hover:-translate-y-0.5 transition-all min-w-0 overflow-hidden">
    <div className="flex items-start justify-between mb-2 gap-2">
      <span className="text-[11px] font-sans uppercase tracking-wider text-muted-foreground font-medium truncate min-w-0">{label}</span>
      <div className={cn("h-8 w-8 rounded-lg flex items-center justify-center shrink-0", toneMap[tone])}>
        <Icon className="h-4 w-4" />
      </div>
    </div>
    <div className="flex items-baseline gap-1 min-w-0">
      <span className="text-2xl font-mono font-bold tracking-tight text-foreground truncate">{value}</span>
      {unit && <span className="text-xs font-mono text-muted-foreground font-medium shrink-0">{unit}</span>}
    </div>
    {trend && <p className="text-xs font-sans text-muted-foreground mt-1 truncate">{trend}</p>}
  </div>
);
