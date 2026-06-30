"use client";

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";

import { getVideoStatus } from "@/lib/api";
import { useVideoStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

const STATUS_LABEL: Record<string, string> = {
  pending: "Queued",
  processing: "Processing…",
  done: "Complete",
  failed: "Failed",
};

const VIDEO_TYPE_LABEL: Record<string, string> = {
  meeting: "Meeting",
  lecture: "Lecture",
  unknown: "Auto-detect",
};

export function JobStatus() {
  const { videoId, videoType, filename, setStep, reset } = useVideoStore();

  const { data, error } = useQuery({
    queryKey: ["videoStatus", videoId],
    queryFn: () => getVideoStatus(videoId!),
    enabled: !!videoId,
    /**
     * TanStack Query v5: returning false from refetchInterval stops polling.
     * This runs after every successful fetch, so polling halts the moment
     * the status transitions to "done" or "failed".
     */
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "done" || status === "failed" ? false : 2000;
    },
  });

  // Sync final status into the Zustand store so sibling/parent components
  // (transcript viewer, chat — added in later prompts) can react to it.
  useEffect(() => {
    if (data?.status === "done" || data?.status === "failed") {
      setStep(data.status);
    }
  }, [data?.status, setStep]);

  const status = data?.status ?? "pending";
  const pct = data?.progress_percent ?? 0;

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle>{STATUS_LABEL[status] ?? status}</CardTitle>
        <div className="flex flex-col gap-0.5 text-sm text-muted-foreground">
          {filename && <span className="truncate">{filename}</span>}
          {videoType && (
            <span>{VIDEO_TYPE_LABEL[videoType] ?? videoType} video</span>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <Progress value={pct} className="h-2" />
        <p className="text-right text-sm text-muted-foreground">{pct}%</p>

        {status === "failed" && (
          <div className="space-y-3">
            <p className="text-sm text-destructive">
              {data?.error_message ?? "An unknown error occurred."}
            </p>
            <Button variant="outline" className="w-full" onClick={reset}>
              Try again
            </Button>
          </div>
        )}

        {/* Network / API error while polling */}
        {error && (
          <p className="text-sm text-destructive">
            Status check failed: {(error as Error).message}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
