import { Bell, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useState, useEffect } from "react";
import { SearchDialog } from "./SearchDialog";

interface Props {
  title: string;
  subtitle?: string;
}

export const PageHeader = ({ title, subtitle }: Props) => {
  const [searchOpen, setSearchOpen] = useState(false);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setSearchOpen((open) => !open);
      }
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  return (
    <>
      <header className="sticky top-0 z-20 bg-background/80 backdrop-blur border-b border-border">
        <div className="flex items-center justify-between px-6 py-4 gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-semibold tracking-tight truncate">{title}</h1>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-primary-soft text-primary text-[11px] font-bold tracking-wider">
                <span className="h-1.5 w-1.5 rounded-full bg-primary live-dot" />
                LIVE
              </span>
            </div>
            {subtitle && <p className="text-sm text-muted-foreground mt-0.5">{subtitle}</p>}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setSearchOpen(true)}
              className="relative hidden sm:flex items-center w-64 h-9 px-3 rounded-md bg-secondary/50 hover:bg-secondary/80 border border-border/50 text-sm text-muted-foreground transition-colors"
            >
              <Search className="mr-2 h-4 w-4" />
              <span>Search nodes, alerts...</span>
              <kbd className="pointer-events-none absolute right-1.5 top-1.5 hidden h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100 sm:flex">
                <span className="text-xs">⌘</span>K
              </kbd>
            </button>
            <Button variant="outline" size="icon" className="relative">
              <Bell className="h-4 w-4" />
              <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-destructive" />
            </Button>
          </div>
        </div>
      </header>

      <SearchDialog open={searchOpen} onOpenChange={setSearchOpen} />
    </>
  );
};
