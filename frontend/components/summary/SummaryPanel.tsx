"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { getSummary, regenerateSummary } from "@/lib/api";
import type { MeetingContent, LectureContent, SummaryResponse } from "@/lib/api";
import { useVideoStore } from "@/lib/store";
import { Button } from "@/components/ui/button";

// ── Skeleton ──────────────────────────────────────────────────────────────────

function SkeletonSection() {
  return (
    <div className="space-y-1.5 animate-pulse">
      <div className="h-3.5 w-24 rounded bg-muted" />
      <div className="space-y-1">
        <div className="h-3 w-full rounded bg-muted" />
        <div className="h-3 w-5/6 rounded bg-muted" />
        <div className="h-3 w-4/6 rounded bg-muted" />
      </div>
    </div>
  );
}

function SummarySkeleton({ numSections }: { numSections: number }) {
  return (
    <div className="space-y-4 px-4 py-3">
      {Array.from({ length: numSections }).map((_, i) => (
        <SkeletonSection key={i} />
      ))}
    </div>
  );
}

// ── Section ───────────────────────────────────────────────────────────────────

function SummarySection({
  title,
  items,
}: {
  title: string;
  items: string[];
}) {
  return (
    <div className="space-y-1">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </h3>
      {items.length === 0 ? (
        <p className="text-xs text-muted-foreground italic">None noted</p>
      ) : (
        <ul className="space-y-0.5">
          {items.map((item, i) => (
            <li key={i} className="flex gap-2 text-sm leading-snug">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-muted-foreground/60" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ── Content renderers ─────────────────────────────────────────────────────────

function MeetingSummary({ content }: { content: MeetingContent }) {
  return (
    <div className="space-y-4 px-4 py-3">
      <SummarySection title="Agenda" items={content.agenda} />
      <SummarySection title="Decisions" items={content.decisions} />
      <SummarySection title="Action Items" items={content.action_items} />
      <SummarySection title="Participants" items={content.participants} />
    </div>
  );
}

function LectureSummary({ content }: { content: LectureContent }) {
  return (
    <div className="space-y-4 px-4 py-3">
      <SummarySection title="Topic Outline" items={content.topic_outline} />
      <SummarySection title="Key Concepts" items={content.key_concepts} />
      <SummarySection title="Examples" items={content.examples} />
    </div>
  );
}

function SummaryContent({ data }: { data: SummaryResponse }) {
  if (data.video_type === "meeting") {
    return <MeetingSummary content={data.content as MeetingContent} />;
  }
  if (data.video_type === "lecture") {
    return <LectureSummary content={data.content as LectureContent} />;
  }
  // Fallback for "unknown" video_type — render raw keys generically
  return (
    <div className="px-4 py-3 text-sm text-muted-foreground">
      Summary is available but the video type is unrecognised ({data.video_type}).
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function SummaryPanel() {
  const { videoId, videoType, step } = useVideoStore();
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ["summary", videoId],
    queryFn: () => getSummary(videoId!),
    // Only fetch once the pipeline is done; guard against empty videoId.
    enabled: !!videoId && step === "done",
    // 5-minute stale window — prevents spurious refetches on window focus
    // while still allowing invalidateQueries from regenerate to trigger a refetch.
    staleTime: 5 * 60 * 1000,
  });

  const regenerateMutation = useMutation({
    mutationFn: () => regenerateSummary(videoId!),
    onSuccess: () => {
      // Invalidate so the GET query re-fetches the fresh record from DB.
      queryClient.invalidateQueries({ queryKey: ["summary", videoId] });
    },
  });

  // ── Guard: no video in store ────────────────────────────────────────────────
  if (!videoId) {
    return (
      <div className="rounded-lg border border-border bg-muted/40 p-6 text-center text-sm text-muted-foreground">
        No video selected — upload a new one.
      </div>
    );
  }

  // Number of skeleton sections to show (matches the expected schema shape).
  const skeletonSections = videoType === "lecture" ? 3 : 4;

  return (
    <div className="w-full rounded-lg border border-border bg-card">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold">Summary</h2>
          {isLoading && (
            <p className="text-xs text-muted-foreground">
              Generating… (LLM may take up to a few minutes)
            </p>
          )}
        </div>
        <Button
          size="sm"
          variant="outline"
          disabled={isLoading || regenerateMutation.isPending}
          onClick={() => regenerateMutation.mutate()}
          title="Force regenerate summary via LLM"
        >
          {regenerateMutation.isPending ? (
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
              Regenerating…
            </span>
          ) : (
            "Regenerate"
          )}
        </Button>
      </div>

      {/* Body */}
      {isLoading ? (
        <SummarySkeleton numSections={skeletonSections} />
      ) : error ? (
        <div className="px-4 py-3">
          <p className="text-sm text-destructive">
            {(error as Error).message}
          </p>
        </div>
      ) : regenerateMutation.isError ? (
        <div className="px-4 py-3 space-y-1">
          {data && <SummaryContent data={data} />}
          <p className="text-xs text-destructive border-t border-destructive/20 pt-2 mt-2">
            Regenerate failed: {(regenerateMutation.error as Error).message}
          </p>
        </div>
      ) : data ? (
        <div className="max-h-[35vh] overflow-y-auto">
          <SummaryContent data={data} />
        </div>
      ) : null}
    </div>
  );
}
