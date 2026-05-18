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
  <div className="glass-card border border-border rounded-xl p-5 shadow-card hover:shadow-elevated hover:-translate-y-0.5 transition-all">
    <div className="flex items-start justify-between mb-3">
      <span className="text-[10px] font-sans uppercase tracking-widest text-muted-foreground font-medium">{label}</span>
      <div className={cn("h-9 w-9 rounded-lg flex items-center justify-center", toneMap[tone])}>
        <Icon className="h-[18px] w-[18px]" />
      </div>
    </div>
    <div className="flex items-baseline gap-1.5">
      <span className="text-3xl font-mono font-bold tracking-tight text-foreground">{value}</span>
      {unit && <span className="text-sm font-mono text-muted-foreground font-medium">{unit}</span>}
    </div>
    {trend && <p className="text-xs font-sans text-muted-foreground mt-1.5">{trend}</p>}
  </div>
);
