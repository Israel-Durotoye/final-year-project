import { useState, useEffect } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Leaf, Droplets, Sparkles, MessageCircleQuestion, ChevronRight, Activity, RotateCcw } from "lucide-react";
import { createClient } from "@supabase/supabase-js";
import ReactMarkdown from "react-markdown";
import { cn } from "@/lib/utils";
import { SoilDoctorCharts } from "@/components/SoilDoctorCharts";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1";
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || (process.env.VITE_SUPABASE_URL as string) || "";
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || (process.env.VITE_SUPABASE_ANON_KEY as string) || "";
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

type ActionType = "view" | "analyse" | "recommend" | "ask" | null;

const ACTIONS = [
  { id: "view", title: "View Sensor Readings", description: "Get a summary of the current conditions on a specific farm section.", icon: Activity },
  { id: "analyse", title: "Analyse Conditions", description: "Understand what the current soil health means for your crops.", icon: Leaf },
  { id: "recommend", title: "Get Recommendations", description: "Receive practical steps to improve soil and crop health.", icon: Droplets },
  { id: "ask", title: "Ask a Question", description: "Ask anything else about your farm or the sensor data.", icon: MessageCircleQuestion },
] as const;

const SoilDoctor = () => {
  const [nodes, setNodes] = useState<string[]>([]);
  const [loadingNodes, setLoadingNodes] = useState(true);

  const [step, setStep] = useState<"action" | "node" | "question" | "result">("action");
  const [selectedAction, setSelectedAction] = useState<ActionType>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [customQuestion, setCustomQuestion] = useState("");
  
  const [loadingResult, setLoadingResult] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  // Load nodes for selection
  useEffect(() => {
    const fetchNodes = async () => {
      try {
        const { data, error } = await supabase
          .from("capstone_dataset")
          .select("Node_ID")
          .order("Timestamp", { ascending: false })
          .limit(200);
        
        if (error) throw error;
        
        const uniqueNodes = Array.from(new Set((data || []).map(r => r.Node_ID))).sort();
        setNodes(uniqueNodes);
      } catch (err) {
        console.error("Failed to load nodes", err);
      } finally {
        setLoadingNodes(false);
      }
    };
    fetchNodes();
  }, []);

  const handleActionSelect = (actionId: ActionType) => {
    setSelectedAction(actionId);
    if (actionId === "ask") {
      setStep("question");
    } else {
      setStep("node");
    }
  };

  const executeAnalysis = async (query: string) => {
    setStep("result");
    setLoadingResult(true);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) throw new Error("Failed to get analysis");

      const data = await res.json();
      setResult(data?.answer ?? "No analysis returned.");
    } catch (err: any) {
      setResult(`Sorry, we couldn't complete the analysis right now. (${err.message || "Connection error"})`);
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
    executeAnalysis(query);
  };

  const handleQuestionSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!customQuestion.trim()) return;
    executeAnalysis(customQuestion.trim());
  };

  const reset = () => {
    setStep("action");
    setSelectedAction(null);
    setSelectedNode(null);
    setCustomQuestion("");
    setResult(null);
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
          <div className="bg-card border border-border p-6 sm:p-8 rounded-xl shadow-sm max-w-2xl mx-auto">
            <h2 className="text-xl font-bold mb-4">What's your question?</h2>
            <form onSubmit={handleQuestionSubmit} className="space-y-4">
              <textarea
                value={customQuestion}
                onChange={(e) => setCustomQuestion(e.target.value)}
                placeholder="E.g., Which of my fields needs the most water right now?"
                className="w-full min-h-[120px] rounded-lg border border-border bg-background p-4 text-base focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary resize-none"
                autoFocus
              />
              <div className="flex justify-end gap-3">
                <Button type="button" variant="outline" onClick={reset}>Cancel</Button>
                <Button type="submit" disabled={!customQuestion.trim()}>Get Answer</Button>
              </div>
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
                  <div className="prose prose-sm sm:prose-base max-w-none prose-headings:font-bold prose-headings:tracking-tight prose-a:text-primary prose-p:leading-relaxed">
                    <ReactMarkdown>{result || ""}</ReactMarkdown>
                  </div>
                )}
              </div>
            </div>
          )
        )}
      </div>
    </>
  );
};

export default SoilDoctor;
