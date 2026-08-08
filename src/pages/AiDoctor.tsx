import { useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Send, Bot, Sparkles, Leaf, Droplets, AlertTriangle, Activity } from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";

interface Msg { role: "user" | "assistant"; content: string; }

const LOCAL_STORAGE_KEY = "soil-doctor-conversation";
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1";

const defaultAssistantMessage: Msg = {
  role: "assistant",
  content:
    "👋 Hello! I'm your AI Soil Doctor. Ask me anything about your fields, soil chemistry, or crop health.",
};

const suggestions = [
  { icon: Leaf, text: "Which crops suit my current N-P-K levels?" },
  { icon: Droplets, text: "How should I adjust irrigation for NODE_04?" },
  { icon: AlertTriangle, text: "Diagnose recent alerts and suggest priorities" },
];

const AiDoctor = () => {
  const [conversationId, setConversationId] = useState<string>(() => {
    if (typeof window === "undefined") {
      return `conv-${Date.now()}`;
    }

    try {
      const stored = window.localStorage.getItem(LOCAL_STORAGE_KEY);
      if (!stored) {
        return window.crypto?.randomUUID?.() ?? `conv-${Date.now()}`;
      }

      const parsed = JSON.parse(stored);
      return parsed?.conversationId ?? window.crypto?.randomUUID?.() ?? `conv-${Date.now()}`;
    } catch {
      return window.crypto?.randomUUID?.() ?? `conv-${Date.now()}`;
    }
  });

  const [messages, setMessages] = useState<Msg[]>(() => {
    if (typeof window === "undefined") {
      return [defaultAssistantMessage];
    }

    try {
      const stored = window.localStorage.getItem(LOCAL_STORAGE_KEY);
      if (!stored) return [defaultAssistantMessage];

      const parsed = JSON.parse(stored);
      return Array.isArray(parsed?.messages) && parsed.messages.length > 0
        ? parsed.messages
        : [defaultAssistantMessage];
    } catch {
      return [defaultAssistantMessage];
    }
  });

  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const location = useLocation();
  const hasAutoFired = useRef(false);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, isLoading]);

  useEffect(() => {
    if (!conversationId) return;

    const syncHistory = async () => {
      try {
        const res = await fetch(`${API_BASE}/chat/history/${conversationId}`);
        if (!res.ok) {
          return;
        }

        const data = await res.json();
        if (Array.isArray(data?.messages) && data.messages.length > messages.length) {
          setMessages(data.messages);
        }
      } catch {
        // Server history is best-effort; continue with local cache.
      }
    };

    void syncHistory();
  }, [conversationId]);

  useEffect(() => {
    if (!conversationId) return;
    window.localStorage.setItem(
      LOCAL_STORAGE_KEY,
      JSON.stringify({ conversationId, messages }),
    );
  }, [conversationId, messages]);

  const handleSendMessage = async (e?: any) => {
    if (e) e.preventDefault();
    const user_query = inputValue.trim();
    if (!user_query) return;

    const nextMessages = [...messages, { role: 'user', content: user_query }];

    // append user message immediately
    setMessages(nextMessages);
    setInputValue("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // The backend owns persisted context for this conversation. Sending the
        // full UI transcript eventually exceeds its 50-message validation cap.
        body: JSON.stringify({ query: user_query, conversation_id: conversationId }),
      });

      if (!res.ok) {
        const errorBody = await res.json().catch(() => null);
        const detail = typeof errorBody?.detail === "string"
          ? errorBody.detail
          : `Server responded ${res.status}`;
        throw new Error(detail);
      }

      const data = await res.json();
      const answer = data?.answer ?? "";
      setMessages((m) => [...m, { role: 'assistant', content: answer }]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to reach the server.";
      setMessages((m) => [...m, { role: 'assistant', content: `Sorry — ${message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const sendText = async (text: string) => {
    const user_query = text.trim();
    if (!user_query) return;

    const nextMessages = [...messages, { role: 'user', content: user_query }];

    setMessages(nextMessages);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: user_query, conversation_id: conversationId }),
      });

      if (!res.ok) {
        const errorBody = await res.json().catch(() => null);
        const detail = typeof errorBody?.detail === "string"
          ? errorBody.detail
          : `Server responded ${res.status}`;
        throw new Error(detail);
      }

      const data = await res.json();
      const answer = data?.answer ?? "";
      setMessages((m) => [...m, { role: 'assistant', content: answer }]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to reach the server.";
      setMessages((m) => [...m, { role: 'assistant', content: `Sorry — ${message}` }]);
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

  const isWelcomeState = !messages.some((message) => message.role === "user");
  const visibleMessages = messages.filter(
    (message, index) => !(isWelcomeState && index === 0 && message.role === "assistant"),
  );

  return (
    <div className="flex h-[calc(100vh-6rem)] min-h-[540px] flex-col">
      <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 sm:px-6 min-h-0">
        <div className="flex items-center justify-between border-b border-border/60 py-3">
          <div className="flex items-center gap-2 text-sm font-medium">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Bot className="h-4 w-4" />
            </div>
            Soil Doctor
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="h-2 w-2 rounded-full bg-primary" />
            Farm context active
          </div>
        </div>

        <div className="flex-1 overflow-y-auto py-8 sm:py-10">
          {isWelcomeState && (
            <div className="mx-auto flex max-w-2xl flex-col items-center pb-10 pt-4 text-center animate-float-in">
              <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-glow">
                <Sparkles className="h-7 w-7" />
              </div>
              <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">How can I help with your farm?</h1>
              <p className="mt-3 max-w-lg text-sm leading-6 text-muted-foreground sm:text-base">
                I use your latest node readings to turn farm data into clear, practical guidance.
              </p>
            </div>
          )}

          <div className="mx-auto max-w-3xl space-y-8">
          {visibleMessages.map((m, i) => (
            <div key={i} className={cn("flex gap-3 sm:gap-4 animate-float-in", m.role === "user" ? "justify-end" : "justify-start")}>
              {m.role === "assistant" && (
                <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                  <Bot className="h-4 w-4 text-primary-foreground" />
                </div>
              )}
              <div className={cn(
                "max-w-[82%] text-[15px] leading-7",
                m.role === "user"
                  ? "rounded-3xl rounded-br-lg bg-primary px-5 py-3 text-primary-foreground shadow-sm"
                  : "px-1 py-0.5 text-foreground"
              )}>
                <div className={cn(
                  "prose prose-sm max-w-none",
                  m.role === "user"
                    ? "prose-invert prose-p:text-primary-foreground"
                    : "prose-headings:mt-0 prose-headings:text-foreground prose-p:text-foreground/90 prose-strong:text-foreground prose-li:text-foreground/90 prose-blockquote:text-muted-foreground prose-blockquote:border-l-primary"
                )}>
                  {m.role === 'assistant' ? (
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  ) : (
                    <div>{m.content}</div>
                  )}
                </div>
              </div>
              {m.role === "user" && (
                <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-secondary text-xs font-semibold text-secondary-foreground">
                  You
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex gap-3 animate-float-in">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <Bot className="h-4 w-4 text-primary-foreground" />
              </div>
              <div className="flex h-9 items-center gap-1.5 px-1">
                <span className="h-2 w-2 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "0ms" }} />
                <span className="h-2 w-2 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "150ms" }} />
                <span className="h-2 w-2 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          )}
          {isLoading && (
            <div className="ml-12 text-xs text-muted-foreground">Checking the farm context…</div>
          )}
          <div ref={endRef} />
          </div>
        </div>

        {isWelcomeState && (
          <div className="mx-auto grid w-full max-w-3xl grid-cols-1 gap-2 pb-4 sm:grid-cols-3">
              {suggestions.map((s, i) => (
                <button key={i} onClick={() => sendText(s.text)}
                  className="group flex min-h-24 items-start gap-3 rounded-xl border border-border bg-card p-4 text-left transition-colors hover:bg-secondary/60">
                <s.icon className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
                <span className="text-sm font-medium leading-5 text-foreground/90">{s.text}</span>
              </button>
            ))}
          </div>
        )}

        <div className="mx-auto w-full max-w-3xl pb-3">
        <form onSubmit={handleSendMessage} className="flex items-end gap-2 rounded-3xl border border-border bg-card p-2 shadow-elevated focus-within:border-primary/60">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } }}
            placeholder="Message Soil Doctor…"
            rows={1}
            disabled={isLoading}
            className="max-h-32 flex-1 resize-none bg-transparent px-3 py-2.5 text-[15px] leading-6 focus:outline-none placeholder:text-muted-foreground"
          />
          <Button type="submit" size="icon" disabled={!inputValue.trim() || isLoading} className="h-10 w-10 shrink-0 rounded-2xl bg-primary text-primary-foreground hover:bg-primary/90">
            <Send className="h-4 w-4" />
          </Button>
        </form>
        <p className="mt-3 flex items-center justify-center gap-1.5 text-[11px] text-muted-foreground">
          <Activity className="h-3 w-3" /> Live node context is used when available. Verify important decisions in the field.
        </p>
        </div>
      </div>
    </div>
  );
};

export default AiDoctor;
