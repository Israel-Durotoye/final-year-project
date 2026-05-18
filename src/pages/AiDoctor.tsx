import { useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Send, Bot, Sparkles, Leaf, Droplets, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";

interface Msg { role: "user" | "assistant"; content: string; }

const suggestions = [
  { icon: Leaf, text: "Which crops suit my current N-P-K levels?" },
  { icon: Droplets, text: "How should I adjust irrigation for NODE_04?" },
  { icon: AlertTriangle, text: "Diagnose recent alerts and suggest priorities" },
];

const sampleResponse = `Based on telemetry across your **6 nodes**, here's my assessment:

### 🌱 Soil Health Summary
- **Average N:** 26 ppm — slightly below optimal (30+ ppm)
- **Average P:** 46 ppm — within healthy range
- **Average K:** 186 ppm — adequate

### 🎯 Recommendations
1. **Apply a balanced 20-10-10 NPK** to NODE_01 and NODE_06
2. **Increase irrigation** on NODE_06 (moisture at 22%)
3. **Inspect NODE_03** — offline for 3+ hours

> Reach optimal yields by addressing nitrogen deficiency this week.`;

const AiDoctor = () => {
  const [messages, setMessages] = useState<Msg[]>([
    { role: 'assistant', content: "👋 Hello! I'm your AI Soil Doctor. Ask me anything about your fields, soil chemistry, or crop health." },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const location = useLocation();
  const hasAutoFired = useRef(false);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, isLoading]);

  const handleSendMessage = async (e?: any) => {
    if (e) e.preventDefault();
    const user_query = inputValue.trim();
    if (!user_query) return;

    // append user message immediately
    setMessages((m) => [...m, { role: 'user', content: user_query }]);
    setInputValue("");
    setIsLoading(true);

    try {
      const res = await fetch("http://localhost:8000/api/v1/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: user_query }),
      });

      if (!res.ok) {
        throw new Error(`Server responded ${res.status}`);
      }

      const data = await res.json();
      const answer = data?.answer ?? "";
      setMessages((m) => [...m, { role: 'assistant', content: answer }]);
    } catch (err) {
      setMessages((m) => [...m, { role: 'assistant', content: "Sorry — the server is unreachable. Please try again later." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const sendText = async (text: string) => {
    const user_query = text.trim();
    if (!user_query) return;

    setMessages((m) => [...m, { role: 'user', content: user_query }]);
    setIsLoading(true);

    try {
      const res = await fetch("http://localhost:8000/api/v1/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: user_query }),
      });

      if (!res.ok) {
        throw new Error(`Server responded ${res.status}`);
      }

      const data = await res.json();
      const answer = data?.answer ?? "";
      setMessages((m) => [...m, { role: 'assistant', content: answer }]);
    } catch (err) {
      setMessages((m) => [...m, { role: 'assistant', content: "Sorry — the server is unreachable. Please try again later." }]);
    } finally {
      setIsLoading(false);
    }
  };

  // If this page was opened with an `autoQuery` in router state, send it once on mount.
  useEffect(() => {
    const auto = (location as any)?.state?.autoQuery;
    if (auto && !hasAutoFired.current) {
      hasAutoFired.current = true;
      // call sendText but do not await to avoid blocking render
      void sendText(auto);
    }
  }, [location]);

  return (
    <>
      <PageHeader title="AI Soil Doctor" subtitle="Your intelligent farming companion" />
      <div className="flex-1 flex flex-col max-w-4xl w-full mx-auto px-4 sm:px-6 py-4 min-h-0">
        <div className="flex-1 overflow-y-auto space-y-5 py-4">
          {messages.map((m, i) => (
            <div key={i} className={cn("flex gap-3 animate-float-in", m.role === "user" ? "justify-end" : "justify-start")}>
              {m.role === "assistant" && (
                <div className="h-9 w-9 rounded-full gradient-primary flex items-center justify-center shrink-0 shadow-glow">
                  <Bot className="h-4 w-4 text-primary-foreground" />
                </div>
              )}
              <div className={cn(
                "max-w-[78%] rounded-3xl px-5 py-4 shadow-sm backdrop-blur-md transition-all duration-300",
                m.role === "user"
                  ? "bg-primary/90 text-primary-foreground rounded-br-md shadow-glow border border-primary/50"
                  : "glass-card rounded-bl-md"
              )}>
                <div className={cn(
                  "prose prose-sm max-w-none",
                  m.role === "user"
                    ? "prose-invert prose-p:text-primary-foreground"
                    : "prose-headings:text-foreground prose-p:text-foreground/90 prose-strong:text-foreground prose-li:text-foreground/90 prose-blockquote:text-muted-foreground prose-blockquote:border-l-primary"
                )}>
                  {m.role === 'assistant' ? (
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  ) : (
                    <div>{m.content}</div>
                  )}
                </div>
              </div>
              {m.role === "user" && (
                <div className="h-9 w-9 rounded-full bg-primary/20 border border-primary/50 flex items-center justify-center shrink-0 text-xs font-semibold text-primary">
                  AK
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex gap-3 animate-float-in">
              <div className="h-9 w-9 rounded-full gradient-primary flex items-center justify-center shrink-0 shadow-glow">
                <Bot className="h-4 w-4 text-primary-foreground" />
              </div>
              <div className="glass-card rounded-3xl rounded-bl-md px-5 py-4 flex gap-1.5 items-center">
                <span className="h-2 w-2 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "0ms" }} />
                <span className="h-2 w-2 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "150ms" }} />
                <span className="h-2 w-2 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          )}
          {isLoading && (
            <div className="text-sm text-muted-foreground text-center mt-2">The Soil Doctor is thinking...</div>
          )}
          <div ref={endRef} />
        </div>

        {messages.length <= 1 && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mb-3">
              {suggestions.map((s, i) => (
                <button key={i} onClick={() => sendText(s.text)}
                  className="text-left p-4 glass-card rounded-2xl hover:border-primary/50 hover:shadow-glow hover:-translate-y-1 transition-all duration-300 flex items-start gap-3 group">
                <s.icon className="h-5 w-5 text-primary mt-0.5 shrink-0 group-hover:scale-110 transition-transform duration-300" />
                <span className="text-sm font-medium text-foreground/90 group-hover:text-primary transition-colors">{s.text}</span>
              </button>
            ))}
          </div>
        )}

        <form onSubmit={handleSendMessage} className="glass-panel rounded-3xl p-3 flex items-end gap-3 mt-2">
          <Sparkles className="h-6 w-6 text-primary ml-3 mb-3 animate-pulse" />
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } }}
            placeholder="Ask the Soil Doctor anything..."
            rows={1}
            disabled={isLoading}
            className="flex-1 resize-none bg-transparent px-3 py-3 text-[15px] focus:outline-none max-h-32 placeholder:text-muted-foreground/70"
          />
          <Button type="submit" size="icon" disabled={!inputValue.trim() || isLoading} className="shrink-0 h-12 w-12 rounded-2xl bg-primary hover:bg-primary/90 text-primary-foreground transition-all duration-300 hover:shadow-glow hover:-translate-y-0.5">
            <Send className="h-5 w-5" />
          </Button>
        </form>
        <p className="text-[11px] text-muted-foreground text-center mt-4 tracking-wide uppercase">
          AI guidance is advisory. Always verify with agronomic best practices.
        </p>
      </div>
    </>
  );
};

export default AiDoctor;
