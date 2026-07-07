"use client";

import { useCallback, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { getVideoUrl } from "@/lib/api";
import { useVideoStore } from "@/lib/store";
import { JobStatus } from "@/components/upload/JobStatus";
import { UploadForm } from "@/components/upload/UploadForm";
import { TranscriptViewer } from "@/components/transcript/TranscriptViewer";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { SummaryPanel } from "@/components/summary/SummaryPanel";
import { VideoPlayer } from "@/components/player/VideoPlayer";

export default function Home() {
  // ── Global store ───────────────────────────────────────────────────────────
  const { step, videoId, seekTo, setSeekTo } = useVideoStore();

  // ── Player state (done step only) ──────────────────────────────────────────
  const [currentTime, setCurrentTime] = useState(0);

  // ── Video URL query — enabled only when step === "done" ───────────────────
  // Presigned MinIO URL is valid for 1 hour (storage.py line 38).
  // staleTime set to 55 min so TanStack Query refetches before expiry.
  const { data: urlData } = useQuery({
    queryKey: ["videoUrl", videoId],
    queryFn: () => getVideoUrl(videoId!),
    enabled: !!videoId && step === "done",
    staleTime: 55 * 60 * 1000,
  });
  const videoUrl = urlData?.url ?? null;

  // ── Shared seek callback (transcript timestamps + citation chips) ──────────
  // useCallback keeps the reference stable so VideoPlayer's useEffect
  // doesn't re-fire due to prop reference changes.
  const handleTimestampClick = useCallback(
    (seconds: number) => {
      console.log(`[seek] ${seconds}s`);
      setSeekTo(seconds);
    },
    [setSeekTo],
  );

  const handleSeeked = useCallback(() => {
    setSeekTo(null);
  }, [setSeekTo]);

  const handleProgress = useCallback((seconds: number) => {
    setCurrentTime(seconds);
  }, []);

  // ── Render: processing / failed ───────────────────────────────────────────
  if (step === "processing" || step === "failed") {
    return (
      <main className="flex flex-1 items-center justify-center p-6 bg-zinc-50 dark:bg-zinc-950">
        <div className="w-full max-w-md">
          <JobStatus />
        </div>
      </main>
    );
  }

  // ── Render: done — full 2-column layout ───────────────────────────────────
  if (step === "done") {
    return (
      <main className="flex flex-1 flex-col items-center p-4 bg-zinc-50 dark:bg-zinc-950 md:p-6">
        {/*
          Layout:
          - Desktop (lg+): 2 columns — [video + transcript | chat]  3fr : 2fr
          - Mobile: single column — player → transcript → chat
        */}
        <div className="w-full max-w-7xl grid grid-cols-1 gap-4 lg:grid-cols-[3fr_2fr] lg:items-start">

          {/* ── Left column: player + transcript ── */}
          <div className="flex flex-col gap-4">
            {videoUrl ? (
              <VideoPlayer
                src={videoUrl}
                seekTo={seekTo}
                onSeeked={handleSeeked}
                onTimeUpdate={handleProgress}
              />
            ) : (
              /* Placeholder while the presigned URL loads */
              <div
                className="flex items-center justify-center rounded-lg bg-muted text-sm text-muted-foreground"
                style={{ aspectRatio: "16/9" }}
              >
                Loading video…
              </div>
            )}

            <TranscriptViewer
              onTimestampClick={handleTimestampClick}
              currentTime={currentTime}
            />
          </div>

          {/* ── Right column: summary + chat ── */}
          <div className="flex flex-col gap-4">
            <SummaryPanel />
            <ChatPanel onCitationClick={handleTimestampClick} />
          </div>
        </div>

        {/* ── Upload another video ── */}
        <div className="w-full max-w-7xl mt-4 flex justify-center">
          <UploadForm title="Upload another video" />
        </div>
      </main>
    );
  }

  // ── Render: upload form (default / step === "upload") ─────────────────────
  return (
    <main className="flex flex-1 items-center justify-center p-6 bg-zinc-50 dark:bg-zinc-950">
      <UploadForm />
    </main>
  );
}
