import { NavLink } from "react-router-dom";
import { Settings, LogOut } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { toast } from "sonner";

export const FloatingControls = () => {
  return (
    <div className="fixed bottom-6 left-6 z-50">
      <div className="glass-card bg-background/60 backdrop-blur-md border border-white/10 rounded-full py-2 px-3 flex items-center gap-3 shadow-lg hover:shadow-glow transition-all duration-300">
        <NavLink
          to="/settings"
          className="h-8 w-8 flex items-center justify-center rounded-full bg-transparent hover:bg-white/10 text-foreground/60 hover:text-primary transition-all group"
          title="Settings"
        >
          <Settings className="h-4 w-4 group-hover:rotate-45 transition-transform duration-300" />
        </NavLink>

        <div className="w-[1px] h-6 bg-white/10"></div>

        <div className="flex items-center gap-2 px-1">
          <div className="h-7 w-7 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center text-[10px] font-mono font-bold text-primary">
            ID
          </div>
          <div className="pr-1">
            <p className="text-[10px] font-bold font-sans text-foreground leading-tight uppercase tracking-wider">Israel Durotoye</p>
            <p className="text-[9px] font-mono text-muted-foreground uppercase tracking-widest">Road manager</p>
          </div>
        </div>

        <div className="w-[1px] h-6 bg-white/10"></div>

        <button
          onClick={async () => {
            await supabase.auth.signOut();
            toast.success("Logged out successfully");
          }}
          className="h-8 w-8 flex items-center justify-center rounded-full bg-transparent hover:bg-white/10 text-foreground/60 hover:text-destructive transition-all group"
          title="Log Out"
        >
          <LogOut className="h-4 w-4 group-hover:translate-x-0.5 transition-transform duration-300" />
        </button>
      </div>
    </div>
  );
};
