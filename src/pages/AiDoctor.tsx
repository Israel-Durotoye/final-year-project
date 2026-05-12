import { useState, useRef, useEffect } from "react";
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
      const res = await fetch("http://localhost:8000/api/chat", {
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
      const res = await fetch("http://localhost:8000/api/chat", {
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
                "max-w-[78%] rounded-2xl px-4 py-3 shadow-sm",
                m.role === "user"
                  ? "bg-primary text-primary-foreground rounded-br-md"
                  : "bg-card border border-border rounded-bl-md"
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
                <div className="h-9 w-9 rounded-full bg-secondary border border-border flex items-center justify-center shrink-0 text-xs font-semibold">
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
              <div className="bg-card border border-border rounded-2xl rounded-bl-md px-4 py-3 flex gap-1.5">
                <span className="h-2 w-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="h-2 w-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="h-2 w-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: "300ms" }} />
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
                  className="text-left p-3 bg-card border border-border rounded-xl hover:border-primary/40 hover:shadow-md transition-all flex items-start gap-2.5">
                <s.icon className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                <span className="text-sm">{s.text}</span>
              </button>
            ))}
          </div>
        )}

        <form onSubmit={handleSendMessage} className="bg-card border border-border rounded-2xl shadow-card p-2 flex items-end gap-2">
          <Sparkles className="h-5 w-5 text-primary ml-2 mb-2.5" />
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } }}
            placeholder="Ask the Soil Doctor anything..."
            rows={1}
            disabled={isLoading}
            className="flex-1 resize-none bg-transparent px-2 py-2.5 text-sm focus:outline-none max-h-32"
          />
          <Button type="submit" size="icon" disabled={!inputValue.trim() || isLoading} className="shrink-0">
            <Send className="h-4 w-4" />
          </Button>
        </form>
        <p className="text-[11px] text-muted-foreground text-center mt-2">
          AI guidance is advisory. Always verify with agronomic best practices.
        </p>
      </div>
    </>
  );
};

export default AiDoctor;
