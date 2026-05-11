import { Bell, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

interface Props {
  title: string;
  subtitle?: string;
}

export const PageHeader = ({ title, subtitle }: Props) => (
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
        <div className="relative hidden sm:block">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input placeholder="Search nodes, alerts..." className="pl-9 w-64 bg-secondary/50" />
        </div>
        <Button variant="outline" size="icon" className="relative">
          <Bell className="h-4 w-4" />
          <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-destructive" />
        </Button>
      </div>
    </div>
  </header>
);
