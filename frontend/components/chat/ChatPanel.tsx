"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { askQuestion } from "@/lib/api";
import type { Citation } from "@/lib/api";
import { useVideoStore } from "@/lib/store";
import { Button } from "@/components/ui/button";

// ── Types ────────────────────────────────────────────────────────────────────

interface CompletedMessage {
  id: number;
  question: string;
  answer: string;
  citations: Citation[];
}

interface Props {
  /**
   * Called when a citation chip is clicked with the chunk's start time (seconds).
   * Uses the same signature as TranscriptViewer's onTimestampClick so prompt 5
   * can wire both to the same player seek handler.
   */
  onCitationClick: (seconds: number) => void;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// ── Citation chip ────────────────────────────────────────────────────────────

/**
 * Displays speaker (if present) + timestamp + short text excerpt.
 *
 * Backend Citation fields (qa.py lines 26-31):
 *   chunk_id, speaker (nullable), start, end, text
 *
 * NOTE: chunk_type is NOT in the Citation response — no slide/speech
 * distinction is made here. Confirmed by reading qa.py directly.
 */
function CitationChip({
  citation,
  onClick,
}: {
  citation: Citation;
  onClick: () => void;
}) {
  const timestamp = formatTime(citation.start);

  // Build the label: "Speaker · mm:ss" for meeting, "mm:ss" for lecture/null.
  // Never show "undefined" or "null" — guard explicitly.
  const label =
    citation.speaker && citation.speaker.trim()
      ? `${citation.speaker} · ${timestamp}`
      : timestamp;

  return (
    <button
      type="button"
      onClick={onClick}
      title={citation.text}
      className={[
        "inline-flex max-w-xs items-start gap-1.5 rounded-full border border-border",
        "bg-muted px-2.5 py-1 text-left text-xs transition-colors",
        "hover:border-primary/50 hover:bg-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
      ].join(" ")}
    >
      <span className="shrink-0 font-medium tabular-nums">{label}</span>
      <span className="max-w-[140px] truncate text-muted-foreground">
        {citation.text}
      </span>
    </button>
  );
}

// ── Loading indicator ────────────────────────────────────────────────────────

function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-2 text-sm text-muted-foreground">
      <span
        className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
        aria-label="Loading"
      />
      Thinking… (Ollama may take a few seconds)
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function ChatPanel({ onCitationClick }: Props) {
  const { videoId } = useVideoStore();

  // Chat history — session-only, not persisted (no backend chat storage).
  const [messages, setMessages] = useState<CompletedMessage[]>([]);
  const [input, setInput] = useState("");
  // Holds the question currently in-flight (for display + retry).
  const [pendingQuestion, setPendingQuestion] = useState("");

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const msgCounter = useRef(0);

  // Backend /ask is non-streaming (ollama.Client().chat() is synchronous).
  // useMutation blocks until the full JSON response arrives.
  const mutation = useMutation({
    mutationFn: (question: string) => askQuestion(videoId!, question),
    onSuccess: (data, question) => {
      msgCounter.current += 1;
      setMessages((prev) => [
        ...prev,
        {
          id: msgCounter.current,
          question,
          answer: data.answer,
          citations: data.citations,
        },
      ]);
      setPendingQuestion("");
      // Re-focus input for quick follow-up questions
      inputRef.current?.focus();
    },
    // On error: pendingQuestion stays set so the retry button works.
  });

  // Scroll to bottom whenever a new message arrives or loading state changes.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, mutation.isPending]);

  // ── Guard: no video in store (reload / direct navigation) ──────────────────
  if (!videoId) {
    return (
      <div className="flex h-[60vh] items-center justify-center rounded-lg border border-border bg-muted/40 p-6 text-center text-sm text-muted-foreground">
        Không có video nào đang được chọn — vui lòng upload video mới.
      </div>
    );
  }

  // ── Handlers ────────────────────────────────────────────────────────────────

  const handleSubmit = () => {
    const q = input.trim();
    if (!q || mutation.isPending) return;
    setPendingQuestion(q);
    setInput("");
    mutation.mutate(q);
  };

  const handleRetry = () => {
    if (!pendingQuestion || mutation.isPending) return;
    // mutate() auto-resets error state from the previous attempt.
    mutation.mutate(pendingQuestion);
  };

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-[60vh] flex-col rounded-lg border border-border bg-card">
      {/* Header */}
      <div className="shrink-0 border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold">Ask about this video</h2>
        <p className="text-xs text-muted-foreground">
          Answers are grounded in the video content only
        </p>
      </div>

      {/* Message list */}
      <div className="flex-1 space-y-5 overflow-y-auto px-4 py-3">
        {messages.length === 0 && !mutation.isPending && (
          <p className="mt-8 text-center text-sm text-muted-foreground">
            Ask a question — the answer will cite the relevant moments.
          </p>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className="space-y-2">
            {/* User question */}
            <div className="flex justify-end">
              <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-primary px-3 py-2 text-sm text-primary-foreground">
                {msg.question}
              </div>
            </div>

            {/* Assistant answer */}
            <div className="space-y-2">
              <div className="max-w-[90%] whitespace-pre-wrap rounded-2xl rounded-bl-sm bg-muted px-3 py-2 text-sm leading-relaxed">
                {msg.answer}
              </div>

              {/* Citation chips — show only when there are citations */}
              {msg.citations.length > 0 && (
                <div className="flex flex-wrap gap-1.5 pl-1">
                  {msg.citations.map((c) => (
                    <CitationChip
                      key={c.chunk_id}
                      citation={c}
                      onClick={() => onCitationClick(c.start)}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* In-flight question + thinking indicator */}
        {mutation.isPending && (
          <div className="space-y-2">
            <div className="flex justify-end">
              <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-primary px-3 py-2 text-sm text-primary-foreground">
                {pendingQuestion}
              </div>
            </div>
            <ThinkingIndicator />
          </div>
        )}

        {/* Error state with retry */}
        {mutation.isError && (
          <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm">
            <p className="text-destructive">
              {(mutation.error as Error).message}
            </p>
            <Button
              size="sm"
              variant="outline"
              className="mt-2"
              onClick={handleRetry}
            >
              Retry
            </Button>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input row */}
      <div className="shrink-0 border-t border-border p-3">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
            placeholder="Ask a question… (Enter to send)"
            disabled={mutation.isPending}
            className={[
              "flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm",
              "placeholder:text-muted-foreground",
              "focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              "disabled:cursor-not-allowed disabled:opacity-50",
            ].join(" ")}
          />
          <Button
            onClick={handleSubmit}
            disabled={!input.trim() || mutation.isPending}
            size="sm"
          >
            Send
          </Button>
        </div>
      </div>
    </div>
  );
}
