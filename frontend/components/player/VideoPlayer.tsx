"use client";

import { useEffect, useRef } from "react";
import ReactPlayer from "react-player";

interface Props {
  /** Presigned MinIO URL from GET /videos/{id}/url */
  src: string;
  /**
   * When non-null the player seeks to this position (seconds).
   * The parent must call onSeeked() so it can reset seekTo to null,
   * preventing the effect from re-firing on subsequent renders.
   */
  seekTo: number | null;
  /** Called immediately after the seek so the parent can clear seekTo. */
  onSeeked: () => void;
  /**
   * Fires on every timeupdate event (~4-25 times/s during playback).
   * Used by TranscriptViewer to highlight the active segment.
   */
  onTimeUpdate: (seconds: number) => void;
}

export function VideoPlayer({ src, seekTo, onSeeked, onTimeUpdate }: Props) {
  // React Player v3 forwards its ref to the underlying HTMLVideoElement.
  const videoRef = useRef<HTMLVideoElement>(null);

  // Stable ref pattern: keeps the latest onSeeked without putting it in the
  // effect dependency array, preventing re-seeks on parent re-renders.
  const onSeekedRef = useRef(onSeeked);
  useEffect(() => {
    onSeekedRef.current = onSeeked;
  });

  useEffect(() => {
    if (seekTo !== null && videoRef.current) {
      videoRef.current.currentTime = seekTo;
      onSeekedRef.current();
    }
  }, [seekTo]);

  return (
    <div className="relative w-full overflow-hidden rounded-lg bg-black" style={{ aspectRatio: "16/9" }}>
      <ReactPlayer
        ref={videoRef}
        src={src}
        controls
        width="100%"
        height="100%"
        crossOrigin="anonymous"
        onTimeUpdate={(e) => {
          onTimeUpdate((e.currentTarget as HTMLVideoElement).currentTime);
        }}
      />
    </div>
  );
}
