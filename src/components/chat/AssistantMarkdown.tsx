import { useMemo, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import { ExternalLink, FileText, Globe2 } from "lucide-react";
import { cn } from "@/lib/utils";

const URL_PATTERN = /https?:\/\/[^\s<>"'`]+/gi;
const MARKDOWN_LINK_PATTERN = /(?<!!)\[([^\]]+)]\((https?:\/\/[^)\s]+)(?:\s+"[^"]*")?\)/gi;
const PROTECTED_SEGMENT_PATTERN = /(```[\s\S]*?```|`[^`\n]+`|!?\[[^\]\n]*]\(https?:\/\/[^)\s]+(?:\s+"[^"]*")?\)|<https?:\/\/[^>\s]+>)/gi;
const MAX_LINK_PREVIEWS = 3;

export type LinkPreview = {
  url: string;
  hostname: string;
  title: string;
  resourceType: "webpage" | "pdf";
};

const stripTrailingPunctuation = (candidate: string) => {
  let url = candidate;
  while (/[.,;:!?\]}]$/.test(url)) url = url.slice(0, -1);

  while (url.endsWith(")")) {
    const openingCount = (url.match(/\(/g) || []).length;
    const closingCount = (url.match(/\)/g) || []).length;
    if (closingCount <= openingCount) break;
    url = url.slice(0, -1);
  }

  return url;
};

export const normalizeHttpUrl = (candidate: string): string | null => {
  const cleaned = stripTrailingPunctuation(candidate.trim());

  try {
    const parsed = new URL(cleaned);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
    return parsed.toString();
  } catch {
    return null;
  }
};

const cleanTitle = (value: string) => value
  .replace(/!\[([^\]]*)]\([^)]*\)/g, "$1")
  .replace(/\[([^\]]+)]\([^)]*\)/g, "$1")
  .replace(/^[\s>*#\-\d.)]+/, "")
  .replace(/[*_~`]/g, "")
  .replace(/\s*[:\-–—]\s*$/, "")
  .replace(/\s+/g, " ")
  .trim()
  .slice(0, 120);

const titleFromHostname = (hostname: string) => {
  const primaryName = hostname.replace(/^www\./, "").split(".")[0] || hostname;
  const readableName = primaryName
    .split(/[-_]/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
  return `${readableName || hostname} resource`;
};

const findContextTitle = (
  markdown: string,
  rawUrl: string,
  normalizedUrl: string,
  markdownTitles: Map<string, string>,
) => {
  const linkedTitle = markdownTitles.get(normalizedUrl);
  if (linkedTitle) return linkedTitle;

  const lines = markdown.split(/\r?\n/);
  const lineIndex = lines.findIndex((line) => line.includes(rawUrl));
  if (lineIndex >= 0) {
    for (let index = lineIndex - 1; index >= 0; index -= 1) {
      const candidate = cleanTitle(lines[index]);
      if (candidate) return candidate;
    }
  }

  return "";
};

export const extractLinkPreviews = (markdown: string): LinkPreview[] => {
  const markdownTitles = new Map<string, string>();
  for (const match of markdown.matchAll(MARKDOWN_LINK_PATTERN)) {
    const normalized = normalizeHttpUrl(match[2]);
    const title = cleanTitle(match[1]);
    if (normalized && title) markdownTitles.set(normalized, title);
  }

  const previews: LinkPreview[] = [];
  const seen = new Set<string>();

  for (const match of markdown.matchAll(URL_PATTERN)) {
    const rawUrl = stripTrailingPunctuation(match[0]);
    const normalized = normalizeHttpUrl(rawUrl);
    if (!normalized || seen.has(normalized)) continue;

    const parsed = new URL(normalized);
    const hostname = parsed.hostname.replace(/^www\./, "");
    const contextTitle = findContextTitle(markdown, rawUrl, normalized, markdownTitles);

    previews.push({
      url: normalized,
      hostname,
      title: contextTitle || titleFromHostname(hostname),
      resourceType: parsed.pathname.toLowerCase().endsWith(".pdf") ? "pdf" : "webpage",
    });
    seen.add(normalized);

    if (previews.length >= MAX_LINK_PREVIEWS) break;
  }

  return previews;
};

export const linkifyBareUrls = (markdown: string) => markdown
  .split(PROTECTED_SEGMENT_PATTERN)
  .map((segment) => {
    if (
      segment.startsWith("`")
      || /^!?\[[^\]\n]*]\(https?:\/\//i.test(segment)
      || /^<https?:\/\//i.test(segment)
    ) {
      return segment;
    }

    return segment.replace(URL_PATTERN, (candidate, offset: number, source: string) => {
      const rawUrl = stripTrailingPunctuation(candidate);
      const normalized = normalizeHttpUrl(rawUrl);
      if (!normalized) return candidate;

      const prefix = source.slice(Math.max(0, offset - 2), offset);
      const previousCharacter = source[offset - 1];
      if (prefix === "](" || previousCharacter === "<") return candidate;

      return `[${rawUrl}](${normalized})${candidate.slice(rawUrl.length)}`;
    });
  })
  .join("");

const SafeExternalLink = ({ href, children }: { href?: string; children?: ReactNode }) => {
  const normalized = href ? normalizeHttpUrl(href) : null;
  if (!normalized) return <span>{children}</span>;

  return (
    <a
      href={normalized}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-baseline gap-1 break-all font-medium text-primary underline decoration-primary/40 underline-offset-4 hover:decoration-primary"
    >
      <span>{children}</span>
      <ExternalLink className="inline h-3.5 w-3.5 shrink-0" aria-hidden="true" />
    </a>
  );
};

const LinkPreviewCard = ({ preview, compact = false }: { preview: LinkPreview; compact?: boolean }) => {
  const Icon = preview.resourceType === "pdf" ? FileText : Globe2;

  return (
    <a
      href={preview.url}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={`Open ${preview.title} in a new tab`}
      className={cn(
        "group flex items-center gap-3 rounded-xl border border-border/80 bg-background/90 text-foreground no-underline shadow-sm transition-all hover:border-primary/50 hover:shadow-md",
        compact ? "p-2.5" : "p-3.5",
      )}
    >
      <div className={cn(
        "flex shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary",
        compact ? "h-8 w-8" : "h-10 w-10",
      )}>
        <Icon className={compact ? "h-4 w-4" : "h-5 w-5"} aria-hidden="true" />
      </div>
      <div className="min-w-0 flex-1">
        <p className={cn("truncate font-semibold", compact ? "text-xs" : "text-sm")}>
          {preview.title}
        </p>
        <p className="mt-0.5 flex items-center gap-1.5 truncate text-xs text-muted-foreground">
          <span className="truncate">{preview.hostname}</span>
          {preview.resourceType === "pdf" && (
            <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-primary">
              PDF
            </span>
          )}
        </p>
      </div>
      <ExternalLink
        className="h-4 w-4 shrink-0 text-muted-foreground transition-colors group-hover:text-primary"
        aria-hidden="true"
      />
    </a>
  );
};

type AssistantMarkdownProps = {
  content: string;
  className?: string;
  compact?: boolean;
};

export const AssistantMarkdown = ({ content, className, compact = false }: AssistantMarkdownProps) => {
  const linkedContent = useMemo(() => linkifyBareUrls(content), [content]);
  const previews = useMemo(() => extractLinkPreviews(content), [content]);

  return (
    <>
      <div
        className={cn(
          "prose prose-sm max-w-none prose-headings:font-semibold prose-a:text-primary prose-p:leading-relaxed dark:prose-invert",
          compact && "text-sm prose-p:my-2",
          className,
        )}
      >
        <ReactMarkdown components={{ a: SafeExternalLink }}>{linkedContent}</ReactMarkdown>
      </div>

      {previews.length > 0 && (
        <div className={cn("not-prose grid gap-2", compact ? "mt-2" : "mt-3")}>
          {previews.map((preview) => (
            <LinkPreviewCard key={preview.url} preview={preview} compact={compact} />
          ))}
        </div>
      )}
    </>
  );
};
