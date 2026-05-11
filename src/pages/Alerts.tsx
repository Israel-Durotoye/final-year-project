import { PageHeader } from "@/components/layout/PageHeader";
import { alerts } from "@/lib/mockData";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { AlertTriangle, AlertCircle, Info, Bot, Clock } from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";

const severityMap = {
  critical: { icon: AlertCircle, cls: "bg-destructive/10 text-destructive border-destructive/20", label: "Critical" },
  warning: { icon: AlertTriangle, cls: "bg-warning/10 text-warning border-warning/20", label: "Warning" },
  info: { icon: Info, cls: "bg-info/10 text-info border-info/20", label: "Info" },
};

const Alerts = () => (
  <>
    <PageHeader title="Alerts" subtitle={`${alerts.length} active alerts across your network`} />
    <div className="p-6">
      <Accordion type="multiple" className="space-y-3">
        {alerts.map((a) => {
          const sev = severityMap[a.severity];
          const Icon = sev.icon;
          return (
            <AccordionItem
              key={a.id}
              value={a.id}
              className="bg-card border border-border rounded-xl shadow-card overflow-hidden hover:shadow-elevated transition-shadow"
            >
              <AccordionTrigger className="px-5 py-4 hover:no-underline group">
                <div className="flex items-center gap-4 flex-1 text-left">
                  <div className={cn("h-11 w-11 rounded-lg flex items-center justify-center border shrink-0", sev.cls)}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={cn("text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border", sev.cls)}>
                        {sev.label}
                      </span>
                      <span className="text-xs font-semibold text-primary">{a.nodeId}</span>
                      <span className="text-xs text-muted-foreground inline-flex items-center gap-1">
                        <Clock className="h-3 w-3" />{a.timestamp}
                      </span>
                    </div>
                    <h3 className="font-semibold mt-1 group-hover:text-primary transition-colors">{a.title}</h3>
                    <p className="text-sm text-muted-foreground mt-0.5 truncate">{a.message}</p>
                  </div>
                </div>
              </AccordionTrigger>
              <AccordionContent className="px-5 pb-5">
                <div className="rounded-xl border border-primary/20 bg-primary-soft/40 p-5">
                  <div className="flex items-center gap-2.5 mb-3">
                    <div className="h-8 w-8 rounded-lg gradient-primary flex items-center justify-center shadow-glow">
                      <Bot className="h-4 w-4 text-primary-foreground" />
                    </div>
                    <h4 className="font-semibold text-primary">🤖 AI Prescription</h4>
                  </div>
                  <div className="prose prose-sm max-w-none prose-headings:text-foreground prose-p:text-foreground/90 prose-strong:text-foreground prose-li:text-foreground/90 prose-blockquote:text-muted-foreground prose-blockquote:border-l-primary">
                    <ReactMarkdown>{a.prescription}</ReactMarkdown>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          );
        })}
      </Accordion>
    </div>
  </>
);

export default Alerts;
