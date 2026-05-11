import { NavLink, useLocation } from "react-router-dom";
import {
  LayoutDashboard, Radio, PlusCircle, Map, Table2, AlertTriangle,
  Bot, Settings, Sprout, LogOut
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/nodes", label: "Nodes", icon: Radio },
  { to: "/add-node", label: "Add Node", icon: PlusCircle },
  { to: "/map", label: "Map View", icon: Map },
  { to: "/data", label: "Data Table", icon: Table2 },
  { to: "/alerts", label: "Alerts", icon: AlertTriangle },
  { to: "/ai-doctor", label: "AI Soil Doctor", icon: Bot },
  { to: "/settings", label: "Settings", icon: Settings },
];

export const AppSidebar = () => {
  const { pathname } = useLocation();
  return (
    <aside className="hidden md:flex w-64 shrink-0 flex-col gradient-sidebar text-sidebar-foreground border-r border-sidebar-border">
      <div className="px-6 py-5 flex items-center gap-3 border-b border-sidebar-border">
        <div className="h-10 w-10 rounded-xl gradient-primary flex items-center justify-center shadow-glow">
          <Sprout className="h-5 w-5 text-primary-foreground" />
        </div>
        <div>
          <h1 className="text-base font-bold tracking-tight text-sidebar-accent-foreground">Soil Doctor</h1>
          <p className="text-[11px] uppercase tracking-wider text-sidebar-foreground/60">Pro · Edge Plot</p>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const active = pathname === item.to;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all",
                active
                  ? "bg-sidebar-primary text-sidebar-primary-foreground shadow-md"
                  : "text-sidebar-foreground/80 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              )}
            >
              <item.icon className="h-[18px] w-[18px]" />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      <div className="p-3 border-t border-sidebar-border">
        <div className="flex items-center gap-3 p-2 rounded-lg hover:bg-sidebar-accent transition-colors cursor-pointer group">
          <div className="h-9 w-9 rounded-full bg-sidebar-primary/20 flex items-center justify-center text-sm font-semibold text-sidebar-primary">
            AK
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-sidebar-accent-foreground truncate">Aanya Kumar</p>
            <p className="text-xs text-sidebar-foreground/60 truncate">Farm Manager</p>
          </div>
          <LogOut className="h-4 w-4 text-sidebar-foreground/60 group-hover:text-sidebar-accent-foreground" />
        </div>
      </div>
    </aside>
  );
};
