import { NavLink } from "react-router-dom";
import { ChevronRight, LogOut, Settings } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { toast } from "sonner";
import { useEffect, useState } from "react";
import { loadUserProfile, profileInitials } from "@/lib/profile";

export const FloatingControls = () => {
  const [expanded, setExpanded] = useState(false);
  const [profile, setProfile] = useState(loadUserProfile);

  useEffect(() => {
    const updateProfile = () => setProfile(loadUserProfile());
    window.addEventListener("soilnet:profile-updated", updateProfile);
    return () => window.removeEventListener("soilnet:profile-updated", updateProfile);
  }, []);

  return (
    <div className="fixed bottom-6 left-6 z-50">
      <div className="flex items-center rounded-full border border-border/80 bg-background/90 p-1.5 shadow-lg backdrop-blur-md transition-all duration-300">
        {expanded && (
          <>
            <NavLink to="/settings" className="flex h-9 w-9 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-secondary hover:text-primary" title="Settings" aria-label="Settings">
              <Settings className="h-4 w-4" />
            </NavLink>
            <div className="mx-1 h-6 w-px bg-border" />
          </>
        )}

        <button onClick={() => setExpanded((value) => !value)} className="flex items-center gap-2 rounded-full px-1.5 py-1 text-left" aria-expanded={expanded} aria-label={expanded ? "Collapse account controls" : "Expand account controls"}>
          <div className="flex h-9 w-9 items-center justify-center rounded-full border border-primary/30 bg-primary/15 text-[10px] font-mono font-bold text-primary">
            {profileInitials(profile.name)}
          </div>
          {expanded && <div className="max-w-44 pr-1"><p className="truncate text-[10px] font-bold uppercase tracking-wider text-foreground">{profile.name}</p><p className="truncate text-[9px] font-mono uppercase tracking-widest text-muted-foreground">{profile.role}</p></div>}
          <ChevronRight className={`h-4 w-4 text-muted-foreground transition-transform ${expanded ? "rotate-180" : ""}`} />
        </button>

        {expanded && (
          <>
            <div className="mx-1 h-6 w-px bg-border" />
            <button onClick={async () => { await supabase.auth.signOut(); toast.success("Logged out successfully"); }} className="flex h-9 w-9 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive" title="Log out" aria-label="Log out">
              <LogOut className="h-4 w-4" />
            </button>
          </>
        )}
      </div>
    </div>
  );
};
