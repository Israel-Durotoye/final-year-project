import { useState, useEffect, useRef } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Leaf, Droplets, Sparkles, MessageCircleQuestion, ChevronRight, Activity, RotateCcw, Bot, Send, UserRound, Plus } from "lucide-react";
import { cn } from "@/lib/utils";
import { SoilDoctorCharts } from "@/components/SoilDoctorCharts";
import { AssistantMarkdown } from "@/components/chat/AssistantMarkdown";
import { fetchTelemetry } from "@/lib/telemetry";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1";

type ActionType = "view" | "analyse" | "recommend" | "ask" | null;
type ChatRole = "user" | "assistant";

type ChatMessage = {
  role: ChatRole;
  content: string;
};

const CHAT_SESSION_STORAGE_KEY = "soil-doctor-conversation-id";
const CHAT_GREETING: ChatMessage = {
  role: "assistant",
  content: "Hello! I'm your Soil Doctor. Ask me anything about your farm, crops, soil, or sensor readings. You can keep asking follow-up questions and I'll remember our conversation.",
};

const createConversationId = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `soil-chat-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
};

const getInitialConversationId = () => {
  if (typeof window === "undefined") return createConversationId();
  return sessionStorage.getItem(CHAT_SESSION_STORAGE_KEY) || createConversationId();
};

const ACTIONS = [
  { id: "view", title: "View Sensor Readings", description: "Get a summary of the current conditions on a specific farm section.", icon: Activity },
  { id: "analyse", title: "Analyse Conditions", description: "Understand what the current soil health means for your crops.", icon: Leaf },
  { id: "recommend", title: "Get Recommendations", description: "Receive practical steps to improve soil and crop health.", icon: Droplets },
  { id: "ask", title: "Ask a Question", description: "Start a conversation and ask follow-up questions about your farm or sensor data.", icon: MessageCircleQuestion },
] as const;

const SoilDoctor = () => {
  const [nodes, setNodes] = useState<string[]>([]);
  const [loadingNodes, setLoadingNodes] = useState(true);

  const [step, setStep] = useState<"action" | "node" | "question" | "result">("action");
  const [selectedAction, setSelectedAction] = useState<ActionType>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [customQuestion, setCustomQuestion] = useState("");
  const [conversationId, setConversationId] = useState(getInitialConversationId);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([CHAT_GREETING]);
  const [loadingChat, setLoadingChat] = useState(false);
  const [chatHistoryLoaded, setChatHistoryLoaded] = useState(false);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  
  const [loadingResult, setLoadingResult] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [resultCanContinue, setResultCanContinue] = useState(false);

  // Load nodes for selection
  useEffect(() => {
    const fetchNodes = async () => {
      try {
        const data = await fetchTelemetry({ limit: 200 });
        const uniqueNodes = Array.from(new Set(data.map((row) => row.Node_ID))).sort();
        setNodes(uniqueNodes);
      } catch (err) {
        console.error("Failed to load nodes", err);
      } finally {
        setLoadingNodes(false);
      }
    };
    fetchNodes();
  }, []);

  useEffect(() => {
    sessionStorage.setItem(CHAT_SESSION_STORAGE_KEY, conversationId);
  }, [conversationId]);

  useEffect(() => {
    if (selectedAction !== "ask" || chatHistoryLoaded) return;
    let cancelled = false;

    const loadChatHistory = async () => {
      try {
        const res = await fetch(`${API_BASE}/chat/history/${conversationId}`);
        if (!res.ok) return;

        const data = await res.json();
        const savedMessages = Array.isArray(data?.messages)
          ? data.messages.filter(
              (message: ChatMessage) =>
                (message.role === "user" || message.role === "assistant") &&
                typeof message.content === "string",
            )
          : [];

        if (!cancelled && savedMessages.length > 0) {
          setChatMessages([CHAT_GREETING, ...savedMessages]);
        }
      } catch (err) {
        if (!cancelled) console.error("Failed to restore chat history", err);
      } finally {
        if (!cancelled) setChatHistoryLoaded(true);
      }
    };

    loadChatHistory();
    return () => {
      cancelled = true;
    };
  }, [chatHistoryLoaded, conversationId, selectedAction]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, loadingChat]);

  const handleActionSelect = (actionId: ActionType) => {
    setSelectedAction(actionId);
    if (actionId === "ask") {
      setStep("question");
    } else {
      setStep("node");
    }
  };

  const executeAnalysis = async (query: string, nodeId?: string) => {
    setStep("result");
    setLoadingResult(true);
    setResult(null);
    setResultCanContinue(false);

    try {
      const res = await fetch(`${API_BASE}/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          node_id: nodeId,
          conversation_id: conversationId,
        }),
      });

      if (!res.ok) throw new Error("Failed to get analysis");

      const data = await res.json();
      setResult(data?.answer ?? "No analysis returned.");
      setResultCanContinue(Boolean(data?.answer));
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Connection error";
      setResult(`Sorry, we couldn't complete the analysis right now. (${message})`);
    } finally {
      setLoadingResult(false);
    }
  };

  const handleNodeSelect = (nodeId: string) => {
    setSelectedNode(nodeId);
    
    if (selectedAction === "view") {
      setStep("result");
      return;
    }
    
    let query = "";
    if (selectedAction === "analyse") {
      query = `Please analyse the current soil conditions for ${nodeId}. Explain what the levels mean for the health of the field in simple terms.`;
    } else if (selectedAction === "recommend") {
      query = `Based on the latest data for ${nodeId}, what practical recommendations and steps should be taken to improve conditions?`;
    }
    executeAnalysis(query, nodeId);
  };

  const handleQuestionSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = customQuestion.trim();
    if (!query || loadingChat || !chatHistoryLoaded) return;

    setChatMessages((messages) => [...messages, { role: "user", content: query }]);
    setCustomQuestion("");
    setLoadingChat(true);

    try {
      const res = await fetch(`${API_BASE}/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          conversation_id: conversationId,
          node_id: selectedNode,
        }),
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`Chat API error: ${res.status}${detail ? ` ${detail}` : ""}`);
      }

      const data = await res.json();
      setChatMessages((messages) => [
        ...messages,
        {
          role: "assistant",
          content: data?.answer ?? "I couldn't generate an answer. Please try again.",
        },
      ]);
    } catch (err: unknown) {
      const message = err instanceof Error
        ? err.message
        : "Please check the backend connection and try again.";
      setChatMessages((messages) => [
        ...messages,
        {
          role: "assistant",
          content: `Sorry, I couldn't answer that right now. ${message}`,
        },
      ]);
    } finally {
      setLoadingChat(false);
    }
  };

  const startNewChat = () => {
    const nextConversationId = createConversationId();
    setConversationId(nextConversationId);
    setChatMessages([CHAT_GREETING]);
    setCustomQuestion("");
    setChatHistoryLoaded(true);
  };

  const continueInChat = () => {
    // The report request used this conversation_id, so reloading history brings
    // both the original analysis question and its answer into the chat thread.
    setChatMessages([CHAT_GREETING]);
    setCustomQuestion("");
    setChatHistoryLoaded(false);
    setSelectedAction("ask");
    setStep("question");
  };

  const reset = () => {
    setStep("action");
    setSelectedAction(null);
    setSelectedNode(null);
    setResult(null);
    setResultCanContinue(false);
  };

  return (
    <>
      <PageHeader title="Soil Doctor" subtitle="Your farm's analysis room" />
      
      <div className="mx-auto w-full p-4 sm:p-6 pb-20 space-y-6 animate-in slide-in-from-bottom-4 duration-500">
        
        {/* Progress / Breadcrumbs */}
        {step !== "action" && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground bg-card border border-border p-3 rounded-xl shadow-sm">
            <button onClick={reset} className="hover:text-primary transition-colors flex items-center gap-1 font-medium">
              <Sparkles className="h-4 w-4" /> Start
            </button>
            <ChevronRight className="h-4 w-4 opacity-50" />
            <span className={step !== "action" ? "text-foreground font-medium" : ""}>
              {ACTIONS.find(a => a.id === selectedAction)?.title}
            </span>
            
            {selectedNode && (
              <>
                <ChevronRight className="h-4 w-4 opacity-50" />
                <span className="text-foreground font-medium">{selectedNode}</span>
              </>
            )}
            
            {(step === "result") && (
              <>
                <ChevronRight className="h-4 w-4 opacity-50" />
                <span className="text-foreground font-medium text-primary">Report</span>
              </>
            )}
          </div>
        )}

        {/* STEP 1: Select Action */}
        {step === "action" && (
          <div className="space-y-4">
            <div className="text-center py-6 mb-2">
              <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-glow">
                <Sparkles className="h-7 w-7" />
              </div>
              <h2 className="text-2xl font-bold tracking-tight">What would you like to do?</h2>
              <p className="text-muted-foreground mt-2">Select an option below to analyse your farm data.</p>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {ACTIONS.map((action) => (
                <button
                  key={action.id}
                  onClick={() => handleActionSelect(action.id)}
                  className="group flex flex-col text-left items-start gap-3 rounded-xl border border-border bg-card p-6 shadow-sm transition-all hover:border-primary/50 hover:shadow-md hover:bg-primary/5"
                >
                  <div className="p-3 bg-primary/10 text-primary rounded-lg group-hover:scale-110 transition-transform">
                    <action.icon className="h-6 w-6" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg">{action.title}</h3>
                    <p className="text-sm text-muted-foreground mt-1 leading-relaxed">{action.description}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* STEP 2: Select Node */}
        {step === "node" && (
          <div className="space-y-6">
            <div className="bg-card border border-border p-6 rounded-xl shadow-sm text-center">
              <h2 className="text-xl font-bold">Which sensor location?</h2>
              <p className="text-muted-foreground mt-1">Select the area you want to {selectedAction}.</p>
            </div>
            
            {loadingNodes ? (
              <div className="p-12 text-center text-muted-foreground border border-border rounded-xl bg-card">
                Finding your sensors...
              </div>
            ) : nodes.length === 0 ? (
              <div className="p-12 text-center text-muted-foreground border border-border rounded-xl bg-card">
                No active sensors found.
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                {nodes.map(nodeId => (
                  <button
                    key={nodeId}
                    onClick={() => handleNodeSelect(nodeId)}
                    className="flex flex-col items-center justify-center p-6 border border-border bg-card rounded-xl shadow-sm hover:border-primary hover:bg-primary/5 hover:shadow-md transition-all group"
                  >
                    <Activity className="h-8 w-8 text-muted-foreground group-hover:text-primary mb-3 transition-colors" />
                    <span className="font-bold font-mono text-lg">{nodeId}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* STEP 2 (Alternative): Ask Question */}
        {step === "question" && (
          <div className="bg-card border border-border rounded-xl shadow-sm max-w-4xl mx-auto min-h-[600px] overflow-hidden flex flex-col">
            <div className="bg-secondary/50 border-b border-border p-4 sm:px-6 flex items-center justify-between gap-4">
              <div className="flex items-center gap-3 min-w-0">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <Bot className="h-5 w-5" />
                </div>
                <div className="min-w-0">
                  <h2 className="font-semibold">Chat with Soil Doctor</h2>
                  <p className="text-xs text-muted-foreground">Ask follow-up questions in the same conversation</p>
                </div>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={startNewChat}
                disabled={loadingChat || !chatHistoryLoaded}
                className="shrink-0"
              >
                <Plus className="h-4 w-4 mr-2" /> New Chat
              </Button>
            </div>

            <div
              className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-5 min-h-[420px] max-h-[60vh]"
              aria-live="polite"
            >
              {chatMessages.map((message, index) => {
                const isUser = message.role === "user";
                return (
                  <div
                    key={`${message.role}-${index}`}
                    className={cn("flex items-start gap-3", isUser && "flex-row-reverse")}
                  >
                    <div
                      className={cn(
                        "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
                        isUser ? "bg-primary text-primary-foreground" : "bg-primary/10 text-primary",
                      )}
                    >
                      {isUser ? <UserRound className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                    </div>
                    <div
                      className={cn(
                        "max-w-[85%] rounded-2xl px-4 py-3 text-sm sm:text-base",
                        isUser
                          ? "bg-primary text-primary-foreground rounded-tr-sm whitespace-pre-wrap"
                          : "bg-secondary text-foreground rounded-tl-sm",
                      )}
                    >
                      {isUser ? (
                        message.content
                      ) : (
                        <AssistantMarkdown content={message.content} />
                      )}
                    </div>
                  </div>
                );
              })}

              {!chatHistoryLoaded && (
                <div className="flex items-start gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                    <Bot className="h-4 w-4" />
                  </div>
                  <div className="rounded-2xl rounded-tl-sm bg-secondary px-4 py-3 text-sm text-muted-foreground animate-pulse">
                    Loading your conversation...
                  </div>
                </div>
              )}

              {chatHistoryLoaded && loadingChat && (
                <div className="flex items-start gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                    <Bot className="h-4 w-4" />
                  </div>
                  <div className="rounded-2xl rounded-tl-sm bg-secondary px-4 py-3 text-sm text-muted-foreground animate-pulse">
                    Soil Doctor is thinking...
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <form onSubmit={handleQuestionSubmit} className="border-t border-border bg-card p-4 sm:p-5">
              <div className="flex items-end gap-3">
                <textarea
                  value={customQuestion}
                  onChange={(e) => setCustomQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      e.currentTarget.form?.requestSubmit();
                    }
                  }}
                  placeholder="Ask a question or continue the conversation..."
                  aria-label="Message Soil Doctor"
                  rows={2}
                  disabled={loadingChat}
                  className="flex-1 max-h-36 min-h-[52px] resize-none rounded-xl border border-border bg-background px-4 py-3 text-sm sm:text-base focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-60"
                  autoFocus
                />
                <Button
                  type="submit"
                  size="icon"
                  disabled={!customQuestion.trim() || loadingChat || !chatHistoryLoaded}
                  aria-label="Send message"
                  className="h-[52px] w-[52px] shrink-0 rounded-xl"
                >
                  <Send className="h-5 w-5" />
                </Button>
              </div>
              <p className="mt-2 text-xs text-muted-foreground">Press Enter to send. Use Shift + Enter for a new line.</p>
            </form>
          </div>
        )}

        {/* STEP 3: Results */}
        {step === "result" && (
          selectedAction === "view" && selectedNode ? (
            <div className="bg-card border border-border rounded-xl shadow-card overflow-hidden flex flex-col p-2 sm:p-6">
              <div className="flex justify-end mb-4">
                <Button variant="outline" size="sm" onClick={reset} className="text-muted-foreground hover:text-foreground">
                  <RotateCcw className="h-4 w-4 mr-2" /> Start Over
                </Button>
              </div>
              <SoilDoctorCharts nodeId={selectedNode} />
            </div>
          ) : (
            <div className="bg-card border border-border rounded-xl shadow-card overflow-hidden flex flex-col min-h-[400px]">
              <div className="bg-secondary/50 border-b border-border p-4 flex justify-between items-center">
                <div className="font-medium flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-primary" />
                  Soil Doctor Analysis
                </div>
                <Button variant="ghost" size="sm" onClick={reset} className="text-muted-foreground hover:text-foreground">
                  <RotateCcw className="h-4 w-4 mr-2" /> Start Over
                </Button>
              </div>
              
              <div className="p-6 sm:p-8 flex-1">
                {loadingResult ? (
                  <div className="flex flex-col items-center justify-center h-full space-y-4 text-muted-foreground min-h-[200px]">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/20 text-primary animate-pulse">
                      <Sparkles className="h-6 w-6" />
                    </div>
                    <p>Analyzing farm data...</p>
                  </div>
                ) : (
                  <AssistantMarkdown
                    content={result || ""}
                    className="sm:prose-base prose-headings:font-bold prose-headings:tracking-tight"
                  />
                )}
              </div>

              {!loadingResult && resultCanContinue && (
                <div className="border-t border-border bg-secondary/20 p-4 sm:px-8 sm:py-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <div>
                    <p className="font-medium">Need more clarity?</p>
                    <p className="text-sm text-muted-foreground">
                      Continue with this report in a chat and ask as many follow-up questions as you need.
                    </p>
                  </div>
                  <Button type="button" onClick={continueInChat} className="shrink-0">
                    <MessageCircleQuestion className="h-4 w-4 mr-2" />
                    Ask a follow-up
                  </Button>
                </div>
              )}
            </div>
          )
        )}
      </div>
    </>
  );
};

export default SoilDoctor;
