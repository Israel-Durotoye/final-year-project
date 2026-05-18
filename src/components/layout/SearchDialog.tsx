import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { LayoutDashboard, Radio, Map, AlertTriangle, Bot } from "lucide-react";

export function SearchDialog({ open, onOpenChange }: { open: boolean, onOpenChange: (open: boolean) => void }) {
  const navigate = useNavigate();

  const handleSelect = (route: string) => {
    onOpenChange(false);
    navigate(route);
  };

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput placeholder="Type a command or search..." />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>
        <CommandGroup heading="Navigation">
          <CommandItem onSelect={() => handleSelect("/")}>
            <LayoutDashboard className="mr-2 h-4 w-4" />
            <span>Dashboard</span>
          </CommandItem>
          <CommandItem onSelect={() => handleSelect("/nodes")}>
            <Radio className="mr-2 h-4 w-4" />
            <span>Nodes Overview</span>
          </CommandItem>
          <CommandItem onSelect={() => handleSelect("/map")}>
            <Map className="mr-2 h-4 w-4" />
            <span>Map View</span>
          </CommandItem>
          <CommandItem onSelect={() => handleSelect("/alerts")}>
            <AlertTriangle className="mr-2 h-4 w-4" />
            <span>Alerts</span>
          </CommandItem>
          <CommandItem onSelect={() => handleSelect("/ai-doctor")}>
            <Bot className="mr-2 h-4 w-4" />
            <span>AI Soil Doctor</span>
          </CommandItem>
        </CommandGroup>
        
        {/* We can expand this with mock nodes or real nodes fetched from Supabase later */}
        <CommandGroup heading="Quick Actions">
          <CommandItem onSelect={() => handleSelect("/add-node")}>
            <Radio className="mr-2 h-4 w-4" />
            <span>Deploy New Node</span>
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
