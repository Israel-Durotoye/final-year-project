import { useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import {
  LayoutDashboard, Radio, PlusCircle, Map, Table2, AlertTriangle,
  Bot, Settings, Sprout, LogOut, Sun, Moon, Menu, X
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/components/theme-provider";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/nodes", label: "Nodes", icon: Radio },
  { to: "/add-node", label: "Add Node", icon: PlusCircle },
  { to: "/map", label: "Map View", icon: Map },
  { to: "/data", label: "Data Table", icon: Table2 },
  { to: "/alerts", label: "Alerts", icon: AlertTriangle },
  { to: "/ai-doctor", label: "AI Soil Doctor", icon: Bot },
];

export const TopNavbar = () => {
  const { pathname } = useLocation();
  const { theme, setTheme } = useTheme();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="absolute top-0 left-0 right-0 z-40 w-full bg-transparent">
      <div className="container flex h-20 items-center justify-between px-6 lg:px-12">
        <NavLink to="/" className="flex items-center gap-4 z-50">
          <div className="h-10 w-10 rounded-xl bg-transparent border-2 border-primary flex items-center justify-center shadow-glow">
            <Sprout className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h1 className="text-lg font-bold font-display tracking-widest text-foreground uppercase">Soil<span className="text-primary">Net</span></h1>
            <p className="text-[9px] font-mono uppercase tracking-[0.2em] text-primary/70">Smart Farming System</p>
          </div>
        </NavLink>

        <nav className="hidden md:flex items-center space-x-2 bg-background/40 backdrop-blur-md border border-white/5 rounded-full px-2 py-1.5 shadow-lg">
          {navItems.map((item) => {
            const active = pathname === item.to;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-full text-xs font-mono tracking-widest uppercase transition-all duration-300",
                  active
                    ? "bg-primary text-primary-foreground shadow-glow"
                    : "text-foreground/60 hover:text-primary hover:bg-black/5 dark:hover:bg-white/5"
                )}
              >
                <item.icon className="h-4 w-4" />
                <span className="hidden lg:inline-block">{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Right side controls */}
        <div className="flex items-center justify-end md:w-32 z-50 gap-2">
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="hidden md:flex items-center justify-center h-10 w-10 rounded-full border border-border bg-background/40 backdrop-blur-md text-foreground hover:bg-black/5 dark:hover:bg-white/5 transition-all shadow-sm"
            title="Toggle theme"
          >
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>

          {/* Mobile menu toggle */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden flex items-center justify-center h-10 w-10 rounded-full border border-border bg-background/40 backdrop-blur-md text-foreground hover:bg-black/5 dark:hover:bg-white/5 transition-all shadow-sm"
          >
            {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <div className="md:hidden absolute top-20 left-0 right-0 bg-background/95 backdrop-blur-xl border-b border-border shadow-lg flex flex-col p-4 space-y-2 z-40 animate-in slide-in-from-top-2">
          {navItems.map((item) => {
            const active = pathname === item.to;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                onClick={() => setMobileMenuOpen(false)}
                className={cn(
                  "flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-mono tracking-widest uppercase transition-all duration-300",
                  active
                    ? "bg-primary text-primary-foreground shadow-glow"
                    : "text-foreground hover:bg-black/5 dark:hover:bg-white/5"
                )}
              >
                <item.icon className="h-5 w-5" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
          
          <div className="h-px w-full bg-border my-2" />
          
          <button
            onClick={() => {
              setTheme(theme === "dark" ? "light" : "dark");
              setMobileMenuOpen(false);
            }}
            className="flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-mono tracking-widest uppercase text-foreground hover:bg-black/5 dark:hover:bg-white/5 transition-all w-full text-left"
          >
            {theme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            <span>{theme === "dark" ? "Light Mode" : "Dark Mode"}</span>
          </button>
        </div>
      )}
    </header>
  );
};
