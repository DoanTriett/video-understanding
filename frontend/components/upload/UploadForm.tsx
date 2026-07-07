"use client";

import { useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { uploadVideo } from "@/lib/api";
import type { VideoType } from "@/lib/api";
import { useVideoStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

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

interface UploadFormProps {
  title?: string;
}

export function UploadForm({ title = "Upload Video" }: UploadFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [selectedType, setSelectedType] = useState<SelectableType>("auto");
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const { setVideo } = useVideoStore();

  const mutation = useMutation({
    mutationFn: () => uploadVideo(file!, selectedType),
    onSuccess: (data) => {
      const videoType: VideoType =
        selectedType === "auto" ? "unknown" : selectedType;
      setVideo(data.video_id, videoType, data.filename);
    },
  });

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
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
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
  );
}
