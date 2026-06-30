const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Shared enums ────────────────────────────────────────────────────────────

/** backend/app/models/video.py: class VideoType */
export type VideoType = "meeting" | "lecture" | "unknown";

/** backend/app/models/video.py: class JobStatus */
export type JobStatus = "pending" | "processing" | "done" | "failed";

// ── Response types ───────────────────────────────────────────────────────────

/**
 * backend/app/models/video.py: class VideoUploadResponse
 * Returned by POST /videos/upload
 */
export interface UploadVideoResponse {
  video_id: string;       // videos.py line 69
  filename: string;       // videos.py line 70
  file_size_mb: number;   // videos.py line 71
  status: JobStatus;      // videos.py line 72
  message: string;        // videos.py line 73
}

/**
 * backend/app/models/video.py: class VideoStatusResponse
 * Returned by GET /videos/{id}/status
 */
export interface VideoStatusResponse {
  video_id: string;           // videos.py line 86 / 101
  status: JobStatus;          // videos.py line 88 / 103
  video_type: VideoType;      // videos.py line 90 / 105
  progress_percent: number;   // videos.py line 89 / 104
  error_message: string | null; // videos.py line 91 / 106
  created_at: string;         // videos.py line 92 / 107 (ISO datetime string)
}

/**
 * Transcript word timestamp
 * backend/workers/tasks.py lines 51-53
 */
export interface WordTimestamp {
  word: string;   // tasks.py line 52
  start: number;  // tasks.py line 52
  end: number;    // tasks.py line 52
}

/**
 * Transcript segment
 * backend/workers/tasks.py lines 46-56
 */
export interface TranscriptSegment {
  text: string;               // tasks.py line 47
  start: number;              // tasks.py line 48
  end: number;                // tasks.py line 49
  words: WordTimestamp[];     // tasks.py line 50-53
}

/**
 * Full transcript response
 * Returned by GET /videos/{id}/transcript — raw JSON from transcript.json
 * backend/workers/tasks.py lines 43-57
 */
export interface TranscriptResponse {
  language: string;              // tasks.py line 44
  duration: number;              // tasks.py line 45
  segments: TranscriptSegment[]; // tasks.py line 46
}

/**
 * Single chunk object
 * backend/app/api/videos.py lines 144-151
 */
export interface Chunk {
  id: string;               // videos.py line 145
  speaker: string | null;   // videos.py line 146
  text: string;             // videos.py line 147
  start: number;            // videos.py line 148
  end: number;              // videos.py line 149
  chunk_type: string;       // videos.py line 150
}

/**
 * Chunks list response
 * Returned by GET /videos/{id}/chunks
 * backend/app/api/videos.py lines 140-154
 */
export interface ChunksResponse {
  video_id: string;     // videos.py line 141
  num_chunks: number;   // videos.py line 142
  chunks: Chunk[];      // videos.py line 143
}

/**
 * A single citation in a QA answer
 * backend/app/api/qa.py: class Citation (lines 26-31)
 */
export interface Citation {
  chunk_id: string;       // qa.py line 27
  speaker: string | null; // qa.py line 28
  start: number;          // qa.py line 29
  end: number;            // qa.py line 30
  text: string;           // qa.py line 31
}

/**
 * QA answer response
 * backend/app/api/qa.py: class AskResponse (lines 34-36)
 * Returned by POST /videos/{id}/ask
 */
export interface AskResponse {
  answer: string;       // qa.py line 35
  citations: Citation[]; // qa.py line 36
}

/**
 * Video presigned URL response
 * Returned by GET /videos/{id}/url
 * backend/app/api/videos.py line 162
 */
export interface VideoUrlResponse {
  url: string; // MinIO presigned URL, valid for 1 hour
}

/**
 * Meeting summary content
 * backend/app/api/summary.py: class MeetingContent (lines 22-26)
 */
export interface MeetingContent {
  agenda: string[];         // summary.py line 23
  decisions: string[];      // summary.py line 24
  action_items: string[];   // summary.py line 25
  participants: string[];   // summary.py line 26
}

/**
 * Lecture summary content
 * backend/app/api/summary.py: class LectureContent (lines 29-32)
 */
export interface LectureContent {
  topic_outline: string[];  // summary.py line 30
  key_concepts: string[];   // summary.py line 31
  examples: string[];       // summary.py line 32
}

/**
 * Summary response
 * backend/app/api/summary.py: class SummaryResponse (lines 35-39)
 * Returned by GET /videos/{id}/summary and POST /videos/{id}/summary/regenerate
 */
export interface SummaryResponse {
  video_id: string;                       // summary.py line 36
  video_type: string;                     // summary.py line 37
  created_at: string;                     // summary.py line 38 (ISO datetime)
  content: MeetingContent | LectureContent; // summary.py line 39
}

// ── API functions ────────────────────────────────────────────────────────────

/**
 * POST /videos/upload
 * Uploads a video file and starts async processing.
 */
export async function uploadVideo(
  file: File,
  videoType: "meeting" | "lecture" | "auto",
): Promise<UploadVideoResponse> {
  const form = new FormData();
  form.append("file", file);
  // backend VideoType enum: "meeting" | "lecture" | "unknown"
  // "auto" maps to "unknown" (VideoType.UNKNOWN is the default)
  form.append("video_type", videoType === "auto" ? "unknown" : videoType);

  const res = await fetch(`${BASE_URL}/videos/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Upload failed");
  }

  return res.json() as Promise<UploadVideoResponse>;
}

/**
 * GET /videos/{id}/status
 * Returns the current processing status of a video.
 */
export async function getVideoStatus(
  videoId: string,
): Promise<VideoStatusResponse> {
  const res = await fetch(`${BASE_URL}/videos/${videoId}/status`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Status fetch failed");
  }

  return res.json() as Promise<VideoStatusResponse>;
}

/**
 * GET /videos/{id}/transcript
 * Returns the raw transcript with word-level timestamps.
 * Only available when status === "done".
 */
export async function getTranscript(
  videoId: string,
): Promise<TranscriptResponse> {
  const res = await fetch(`${BASE_URL}/videos/${videoId}/transcript`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Transcript fetch failed");
  }

  return res.json() as Promise<TranscriptResponse>;
}

/**
 * GET /videos/{id}/chunks
 * Returns the list of indexed chunks from Postgres.
 * Only available when status === "done".
 */
export async function getChunks(videoId: string): Promise<ChunksResponse> {
  const res = await fetch(`${BASE_URL}/videos/${videoId}/chunks`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Chunks fetch failed");
  }

  return res.json() as Promise<ChunksResponse>;
}

/**
 * GET /videos/{id}/url
 * Returns a 1-hour presigned MinIO URL for streaming the video.
 * Source: backend/app/api/videos.py lines 157-162
 */
export async function getVideoUrl(videoId: string): Promise<VideoUrlResponse> {
  const res = await fetch(`${BASE_URL}/videos/${videoId}/url`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Video URL fetch failed");
  }

  return res.json() as Promise<VideoUrlResponse>;
}

/**
 * GET /videos/{id}/summary
 * Returns the auto-generated summary. On first call the backend runs the LLM
 * (lazy generation — may take up to ~3 min); subsequent calls return from DB.
 * Only available when status === "done".
 * backend/app/api/summary.py lines 55-75
 */
export async function getSummary(videoId: string): Promise<SummaryResponse> {
  const res = await fetch(`${BASE_URL}/videos/${videoId}/summary`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Summary fetch failed");
  }

  return res.json() as Promise<SummaryResponse>;
}

/**
 * POST /videos/{id}/summary/regenerate
 * Forces the LLM to regenerate and overwrite the stored summary.
 * backend/app/api/summary.py lines 78-95
 */
export async function regenerateSummary(videoId: string): Promise<SummaryResponse> {
  const res = await fetch(`${BASE_URL}/videos/${videoId}/summary/regenerate`, {
    method: "POST",
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Regenerate failed");
  }

  return res.json() as Promise<SummaryResponse>;
}

/**
 * POST /videos/{id}/ask
 * Asks a question about a processed video; returns a grounded answer with citations.
 * Only callable when status === "done".
 */
export async function askQuestion(
  videoId: string,
  question: string,
  topK: number = 6,
): Promise<AskResponse> {
  const res = await fetch(`${BASE_URL}/videos/${videoId}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Ask failed");
  }

  return res.json() as Promise<AskResponse>;
}
