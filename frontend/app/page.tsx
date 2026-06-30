"use client";

import { useCallback, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { getVideoUrl, uploadVideo } from "@/lib/api";
import type { VideoType } from "@/lib/api";
import { useVideoStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { JobStatus } from "@/components/upload/JobStatus";
import { TranscriptViewer } from "@/components/transcript/TranscriptViewer";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { SummaryPanel } from "@/components/summary/SummaryPanel";
import { VideoPlayer } from "@/components/player/VideoPlayer";

// Matches ALLOWED_EXTENSIONS in backend/app/api/videos.py line 16
const ACCEPTED_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"];
// Matches settings.max_file_size_mb in backend/app/config.py line 19
const MAX_SIZE_MB = 500;

type SelectableType = "meeting" | "lecture" | "auto";

const TYPE_LABELS: { value: SelectableType; label: string }[] = [
  { value: "meeting", label: "Meeting" },
  { value: "lecture", label: "Lecture" },
  { value: "auto", label: "Auto-detect" },
];

export default function Home() {
  // ── Upload form state ──────────────────────────────────────────────────────
  const [file, setFile] = useState<File | null>(null);
  const [selectedType, setSelectedType] = useState<SelectableType>("auto");
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── Global store ───────────────────────────────────────────────────────────
  const { step, videoId, seekTo, setVideo, setSeekTo } = useVideoStore();

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

  // ── Upload mutation ────────────────────────────────────────────────────────
  const mutation = useMutation({
    mutationFn: () => uploadVideo(file!, selectedType),
    onSuccess: (data) => {
      const videoType: VideoType =
        selectedType === "auto" ? "unknown" : selectedType;
      setVideo(data.video_id, videoType, data.filename);
    },
  });

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
      </main>
    );
  }

  // ── Render: upload form (default / step === "upload") ─────────────────────
  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setFile(files[0]);
    mutation.reset();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const fileSizeMB = file ? file.size / 1024 / 1024 : 0;

  return (
    <main className="flex flex-1 items-center justify-center p-6 bg-zinc-50 dark:bg-zinc-950">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Upload Video</CardTitle>
          <p className="text-sm text-muted-foreground">
            Meeting or lecture video — up to {MAX_SIZE_MB} MB
          </p>
        </CardHeader>

        <CardContent className="space-y-5">
          {/* ── Drop zone ── */}
          <div
            role="button"
            tabIndex={0}
            aria-label="Drop video file here or click to browse"
            className={[
              "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed",
              "cursor-pointer select-none p-8 text-center transition-colors",
              isDragging
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/50 hover:bg-accent/40",
            ].join(" ")}
            onDragOver={handleDragOver}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED_EXTENSIONS.join(",")}
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />

            {file ? (
              <>
                <span className="text-sm font-medium truncate max-w-full">
                  {file.name}
                </span>
                <span className="text-xs text-muted-foreground">
                  {fileSizeMB.toFixed(1)} MB — click to change
                </span>
              </>
            ) : (
              <>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-8 w-8 text-muted-foreground"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                  aria-hidden
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                  />
                </svg>
                <p className="text-sm font-medium">
                  Drop video here or click to browse
                </p>
                <p className="text-xs text-muted-foreground">
                  {ACCEPTED_EXTENSIONS.join(" · ")}
                </p>
              </>
            )}
          </div>

          {/* ── Video type selector ── */}
          <div className="space-y-1.5">
            <p className="text-sm font-medium">Video type</p>
            <div className="flex gap-2">
              {TYPE_LABELS.map(({ value, label }) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setSelectedType(value)}
                  className={[
                    "flex-1 rounded-md border px-3 py-2 text-sm transition-colors",
                    selectedType === value
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border bg-background hover:bg-accent",
                  ].join(" ")}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* ── Upload button ── */}
          <Button
            className="w-full"
            disabled={!file || mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            {mutation.isPending ? "Uploading…" : "Upload"}
          </Button>

          {mutation.isError && (
            <p className="text-sm text-destructive text-center">
              {(mutation.error as Error).message}
            </p>
          )}
        </CardContent>
      </Card>
    </main>
  );
}
