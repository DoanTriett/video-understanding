import { create } from "zustand";

import type { VideoType } from "./api";

export type UploadStep = "upload" | "processing" | "done" | "failed";

interface VideoStore {
  videoId: string | null;
  /** VideoType as stored in the backend: "meeting" | "lecture" | "unknown" */
  videoType: VideoType | null;
  filename: string | null;
  step: UploadStep;
  /**
   * Seconds to seek to in the video player.
   * Set by transcript/citation timestamp clicks; read by the player in prompt 5.
   */
  seekTo: number | null;

  /**
   * Called on successful upload. Sets videoId, videoType, filename and
   * atomically transitions step → "processing".
   */
  setVideo: (videoId: string, videoType: VideoType, filename: string) => void;
  /** Called by JobStatus when polling reveals the final status. */
  setStep: (step: UploadStep) => void;
  /**
   * Written by timestamp/citation clicks; read by VideoPlayer to seek.
   * Pass null to clear after the seek has been executed.
   */
  setSeekTo: (seconds: number | null) => void;
  /** Resets everything back to the initial upload form. */
  reset: () => void;
}

export const useVideoStore = create<VideoStore>((set) => ({
  videoId: null,
  videoType: null,
  filename: null,
  step: "upload",
  seekTo: null,

  setVideo: (videoId, videoType, filename) =>
    set({ videoId, videoType, filename, step: "processing" }),

  setStep: (step) => set({ step }),

  setSeekTo: (seconds) => set({ seekTo: seconds }),

  reset: () =>
    set({
      videoId: null,
      videoType: null,
      filename: null,
      step: "upload",
      seekTo: null,
    }),
}));
