import { useState, useEffect } from "react";
import { Sparkles, X, Send, Bot } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export const AiWidget = () => {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<{ role: "user" | "ai"; text: string }[]>([
    { role: "ai", text: "Hi! I'm your Soil Doctor AI. Ask me anything about your fields." },
  ]);
  const [isLoading, setIsLoading] = useState(false);

  // listen for diagnose events
  useEffect(() => {
    const handler = (e: Event) => {
      const ev = e as CustomEvent;
      const nodeId = ev?.detail?.nodeId;
      if (!nodeId) return;
      setOpen(true);
      const prompt = `Diagnose ${nodeId}: please analyze latest sensor telemetry and recommend interventions.`;
      setMessages((m) => [...m, { role: "user", text: prompt }]);
      callChat(prompt);
    };

    window.addEventListener("ai:diagnose", handler as EventListener);
    return () => window.removeEventListener("ai:diagnose", handler as EventListener);
  }, []);
    async function callChat(query: string) {
      try {
        setIsLoading(true);
        const res = await fetch("http://localhost:8000/api/v1/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query }),
        });
        if (!res.ok) {
          const txt = await res.text();
          throw new Error(`Chat API error: ${res.status} ${txt}`);
        }
        const data = await res.json();
        const answer = data.answer ?? data.result ?? data.reply ?? JSON.stringify(data);
        setMessages((m) => [...m, { role: "ai", text: String(answer) }]);
      } catch (err: any) {
        console.error("Chat error", err);
        setMessages((m) => [...m, { role: "ai", text: `Error: ${err?.message ?? String(err)}` }]);
      } finally {
        setIsLoading(false);
      }
    }

    const send = () => {
      if (!input.trim()) return;
      const q = input.trim();
      setMessages((m) => [...m, { role: "user", text: q }]);
      setInput("");
      callChat(q);
    };

  return (
    <>
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "fixed bottom-6 right-6 z-40 h-14 w-14 rounded-full gradient-primary shadow-glow flex items-center justify-center text-primary-foreground transition-transform hover:scale-110",
          open && "scale-90"
        )}
        aria-label="Ask AI"
      >
        {open ? <X className="h-6 w-6" /> : <Sparkles className="h-6 w-6" />}
      </button>

      {open && (
        <div className="fixed bottom-24 right-6 z-40 w-[360px] max-w-[calc(100vw-2rem)] h-[480px] bg-card border border-border rounded-2xl shadow-elevated flex flex-col animate-float-in overflow-hidden">
          <div className="flex items-center gap-3 p-4 border-b border-border gradient-primary text-primary-foreground">
            <div className="h-9 w-9 rounded-full bg-white/20 flex items-center justify-center">
              <Bot className="h-5 w-5" />
            </div>
            <div>
              <p className="font-semibold text-sm">AI Soil Doctor</p>
              <p className="text-xs opacity-90">Contextual farming assistant</p>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.map((m, i) => (
              <div key={i} className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}>
                <div className={cn(
                  "max-w-[80%] px-3.5 py-2.5 rounded-2xl text-sm",
                  m.role === "user"
                    ? "bg-primary text-primary-foreground rounded-br-sm"
                    : "bg-secondary text-secondary-foreground rounded-bl-sm"
                )}>
                  {m.text}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className={cn("flex justify-start")}> 
                <div className={cn("max-w-[80%] px-3.5 py-2.5 rounded-2xl text-sm bg-secondary text-secondary-foreground rounded-bl-sm")}>Thinking...</div>
              </div>
            )}
          </div>
          <div className="p-3 border-t border-border flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
              placeholder="Ask about your soil..."
              className="flex-1 px-3 py-2 text-sm bg-secondary/50 border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <Button size="icon" onClick={send}><Send className="h-4 w-4" /></Button>
          </div>
        </div>
      )}
    </>
  );
};
