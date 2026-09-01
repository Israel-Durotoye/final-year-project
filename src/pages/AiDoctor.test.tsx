import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import AiDoctor from "./AiDoctor";

vi.mock("@/components/layout/PageHeader", () => ({
  PageHeader: () => <div>Soil Doctor header</div>,
}));

vi.mock("@/components/SoilDoctorCharts", () => ({
  SoilDoctorCharts: ({ nodeId }: { nodeId: string }) => <div>Charts for {nodeId}</div>,
}));

vi.mock("@/components/chat/AssistantMarkdown", () => ({
  AssistantMarkdown: ({ content }: { content: string }) => <div>{content}</div>,
}));

vi.mock("@supabase/supabase-js", () => ({
  createClient: () => ({
    from: () => ({
      select: () => ({
        order: () => ({
          limit: async () => ({ data: [{ Node_ID: "NODE_01" }], error: null }),
        }),
      }),
    }),
  }),
}));

describe("AiDoctor report follow-up", () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.restoreAllMocks();
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: vi.fn(),
    });
  });

  it("opens chat with the generated analysis preserved in conversation history", async () => {
    const analysisAnswer = "NODE_01 is generally healthy, with moisture worth monitoring.";
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (init?.method === "POST") {
        const body = JSON.parse(String(init.body));
        expect(body.node_id).toBe("NODE_01");
        expect(body.conversation_id).toBeTruthy();
        return {
          ok: true,
          json: async () => ({ answer: analysisAnswer }),
        } as Response;
      }
      if (url.includes("/chat/history/")) {
        return {
          ok: true,
          json: async () => ({
            messages: [
              { role: "user", content: "Please analyse NODE_01" },
              { role: "assistant", content: analysisAnswer },
            ],
          }),
        } as Response;
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<AiDoctor />);

    fireEvent.click(screen.getByRole("button", { name: /analyse conditions/i }));
    const nodeButton = await screen.findByRole("button", { name: /node_01/i });
    fireEvent.click(nodeButton);

    expect(await screen.findByText(analysisAnswer)).toBeInTheDocument();
    const followUpButton = screen.getByRole("button", { name: /ask a follow-up/i });
    fireEvent.click(followUpButton);

    expect(await screen.findByText("Chat with Soil Doctor")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("Please analyse NODE_01")).toBeInTheDocument();
      expect(screen.getByText(analysisAnswer)).toBeInTheDocument();
    });
    expect(
      screen.getByPlaceholderText(/ask a question or continue the conversation/i),
    ).toBeEnabled();
  });
});
