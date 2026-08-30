import { createElement } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  AssistantMarkdown,
  extractLinkPreviews,
  linkifyBareUrls,
  normalizeHttpUrl,
} from "./AssistantMarkdown";

describe("AssistantMarkdown URL handling", () => {
  it("turns a bare HTTP URL into a clickable Markdown link", () => {
    const content = "Read https://www.fao.org/3/y1453e/y1453e00.htm for guidance.";

    expect(linkifyBareUrls(content)).toContain(
      "[https://www.fao.org/3/y1453e/y1453e00.htm](https://www.fao.org/3/y1453e/y1453e00.htm)",
    );
  });

  it("does not alter existing Markdown links or inline code", () => {
    const linked = "[FAO guide](https://www.fao.org/guide)";
    const urlLabel = "[https://www.fao.org/guide](https://www.fao.org/guide)";
    const code = "Run `curl https://example.com` locally.";

    expect(linkifyBareUrls(linked)).toBe(linked);
    expect(linkifyBareUrls(urlLabel)).toBe(urlLabel);
    expect(linkifyBareUrls(code)).toBe(code);
  });

  it("builds a preview title from the text immediately before the URL", () => {
    const previews = extractLinkPreviews(
      "**FAO – Fertilizer and Plant Nutrition Guide:**\nhttps://www.fao.org/3/y1453e/y1453e00.htm",
    );

    expect(previews).toEqual([
      {
        url: "https://www.fao.org/3/y1453e/y1453e00.htm",
        hostname: "fao.org",
        title: "FAO – Fertilizer and Plant Nutrition Guide",
        resourceType: "webpage",
      },
    ]);
  });

  it("deduplicates links, strips sentence punctuation, and identifies PDFs", () => {
    const previews = extractLinkPreviews(
      "Guide: https://example.org/report.pdf. Duplicate: https://example.org/report.pdf",
    );

    expect(previews).toHaveLength(1);
    expect(previews[0].url).toBe("https://example.org/report.pdf");
    expect(previews[0].resourceType).toBe("pdf");
  });

  it("accepts only HTTP and HTTPS destinations", () => {
    expect(normalizeHttpUrl("https://fao.org/resource")).toBe("https://fao.org/resource");
    expect(normalizeHttpUrl("javascript:alert(1)")).toBeNull();
    expect(normalizeHttpUrl("data:text/html,test")).toBeNull();
    expect(normalizeHttpUrl("not-a-url")).toBeNull();
  });

  it("renders a safe clickable link and a titled preview card", () => {
    render(createElement(AssistantMarkdown, {
      content: "**FAO Fertilizer Guide:**\nhttps://www.fao.org/3/y1453e/y1453e00.htm",
    }));

    const preview = screen.getByRole("link", {
      name: "Open FAO Fertilizer Guide in a new tab",
    });
    expect(preview).toHaveAttribute("href", "https://www.fao.org/3/y1453e/y1453e00.htm");
    expect(preview).toHaveAttribute("target", "_blank");
    expect(preview).toHaveAttribute("rel", "noopener noreferrer");
    expect(screen.getByText("fao.org")).toBeInTheDocument();
  });
});
