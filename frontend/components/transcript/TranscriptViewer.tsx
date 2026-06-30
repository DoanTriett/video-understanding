"use client";

import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";

import { getTranscript } from "@/lib/api";
import type { TranscriptSegment } from "@/lib/api";
import { useVideoStore } from "@/lib/store";

interface Props {
  /** Called when a timestamp button is clicked (seconds). */
  onTimestampClick: (seconds: number) => void;
  /**
   * Current playback position from the video player (seconds).
   * Used to highlight the active segment and auto-scroll to it.
   * Optional — when absent (e.g. before player is ready) highlighting is skipped.
   */
  currentTime?: number;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** Returns the index of the last segment whose start <= currentTime. */
function findActiveIndex(
  segments: TranscriptSegment[],
  currentTime: number,
): number {
  let active = -1;
  for (let i = 0; i < segments.length; i++) {
    if (segments[i].start <= currentTime) active = i;
    else break;
  }
  return active;
}

function SkeletonRow() {
  return (
    <div className="flex gap-3 py-3 animate-pulse">
      <div className="h-5 w-10 shrink-0 rounded bg-muted" />
      <div className="flex-1 space-y-1.5">
        <div className="h-4 w-full rounded bg-muted" />
        <div className="h-4 w-3/4 rounded bg-muted" />
      </div>
    </div>
  );
}

function TranscriptSkeleton() {
  return (
    <div className="divide-y divide-border">
      {Array.from({ length: 6 }).map((_, i) => (
        <SkeletonRow key={i} />
      ))}
    </div>
  );
}

function SegmentRow({
  segment,
  isActive,
  onTimestampClick,
}: {
  segment: TranscriptSegment;
  isActive: boolean;
  onTimestampClick: (seconds: number) => void;
}) {
  return (
    <div
      className={[
        "flex gap-3 py-2.5 px-2 -mx-2 rounded-md text-sm leading-relaxed transition-colors",
        isActive ? "bg-primary/10 dark:bg-primary/15" : "",
      ].join(" ")}
    >
      <button
        type="button"
        onClick={() => onTimestampClick(segment.start)}
        className={[
          "shrink-0 font-mono text-xs hover:underline focus:outline-none",
          "focus-visible:ring-2 focus-visible:ring-ring rounded",
          isActive ? "text-primary font-semibold" : "text-muted-foreground",
        ].join(" ")}
        title={`Jump to ${formatTime(segment.start)}`}
      >
        {formatTime(segment.start)}
      </button>

      <p className={isActive ? "text-foreground" : "text-foreground/80"}>
        {segment.text}
      </p>
    </div>
  );
}

export function TranscriptViewer({ onTimestampClick, currentTime = 0 }: Props) {
  const { videoId, step } = useVideoStore();

  const { data, isLoading, error } = useQuery({
    queryKey: ["transcript", videoId],
    queryFn: () => getTranscript(videoId!),
    enabled: !!videoId && step === "done",
    staleTime: Infinity,
  });

  // Refs for each segment row — used to auto-scroll the active one into view.
  const rowRefs = useRef<(HTMLDivElement | null)[]>([]);

  const activeIndex =
    data && currentTime > 0 ? findActiveIndex(data.segments, currentTime) : -1;

  // Auto-scroll active segment into view whenever it changes.
  useEffect(() => {
    if (activeIndex >= 0 && rowRefs.current[activeIndex]) {
      rowRefs.current[activeIndex]!.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [activeIndex]);

  // ── No video in store (reload) ───────────────────────────────────────────
  if (!videoId) {
    return (
      <div className="rounded-lg border border-border bg-muted/40 p-6 text-center text-sm text-muted-foreground">
        Không có video nào đang được chọn — vui lòng upload video mới.
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="w-full rounded-lg border border-border bg-card p-4">
        <p className="mb-3 text-sm font-medium text-muted-foreground">
          Loading transcript…
        </p>
        <TranscriptSkeleton />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
        Failed to load transcript: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.segments.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-muted/40 p-6 text-center text-sm text-muted-foreground">
        Transcript is empty or unavailable for this video.
      </div>
    );
  }

  return (
    <div className="w-full rounded-lg border border-border bg-card">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold">Transcript</h2>
        <span className="text-xs text-muted-foreground">
          {data.segments.length} segments · {formatTime(data.duration)} total
        </span>
      </div>

      {/*
        /transcript returns raw Whisper segments (text, start, end, words).
        No speaker field — speaker info is in /chunks only.
      */}
      <div className="max-h-[40vh] overflow-y-auto px-4 py-1">
        {data.segments.map((seg, i) => (
          <div
            key={i}
            ref={(el) => {
              rowRefs.current[i] = el;
            }}
          >
            <SegmentRow
              segment={seg}
              isActive={i === activeIndex}
              onTimestampClick={onTimestampClick}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
