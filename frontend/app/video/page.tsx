"use client";

import { useState, useRef, useCallback } from "react";

/* ── Types ─────────────────────────────────────────────────────────── */
interface Detection {
  label: string;
  confidence: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

interface VideoFrame {
  frame_number: number;
  timestamp: number;
  detections: Detection[];
  count: number;
  inference_time: number;
  annotated_image: string;
}

interface VideoResult {
  frames: VideoFrame[];
  total_potholes: number;
  total_frames_analyzed: number;
  total_video_frames: number;
  video_fps: number;
  total_inference_time: number;
}

/* ── Icons ─────────────────────────────────────────────────────────── */

function IconScan() {
  return (
    <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
    </svg>
  );
}

function IconWarning() {
  return (
    <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
    </svg>
  );
}

function IconTrash() {
  return (
    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
    </svg>
  );
}

function IconCamera() {
  return (
    <svg width="48" height="48" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z" />
    </svg>
  );
}

/* ── Main Component ────────────────────────────────────────────────── */

export default function VideoPage() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoResult, setVideoResult] = useState<VideoResult | null>(null);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [videoDragOver, setVideoDragOver] = useState(false);
  const [selectedFrameSrc, setSelectedFrameSrc] = useState<string | null>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);

  /* ── File Selection ─────────────────────────────────────────────── */
  const handleVideoSelect = useCallback((file: File) => {
    if (!file.type.startsWith("video/")) {
      setVideoError("Please upload a video file (MP4, AVI, MOV, WebM)");
      return;
    }
    setVideoFile(file);
    setVideoResult(null);
    setVideoError(null);
  }, []);

  const onVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleVideoSelect(e.target.files[0]);
  };

  const onVideoDragOver = (e: React.DragEvent) => { e.preventDefault(); setVideoDragOver(true); };
  const onVideoDragLeave = () => setVideoDragOver(false);
  const onVideoDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setVideoDragOver(false);
    if (e.dataTransfer.files[0]) handleVideoSelect(e.dataTransfer.files[0]);
  };

  /* ── Detection ──────────────────────────────────────────────────── */
  const runVideoDetection = async () => {
    if (!videoFile) return;
    setVideoLoading(true);
    setVideoError(null);

    try {
      const formData = new FormData();
      formData.append("file", videoFile);

      const res = await fetch("http://127.0.0.1:8000/api/detect-video", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Server error ${res.status}`);
      }

      const data: VideoResult = await res.json();
      setVideoResult(data);
    } catch (err) {
      setVideoError(err instanceof Error ? err.message : "Video detection failed.");
    } finally {
      setVideoLoading(false);
    }
  };

  /* ── Clear ──────────────────────────────────────────────────────── */
  const clearVideo = () => {
    setVideoFile(null);
    setVideoResult(null);
    setVideoError(null);
    if (videoInputRef.current) videoInputRef.current.value = "";
  };

  /* ── Render ─────────────────────────────────────────────────────── */
  return (
    <div style={{ position: "relative", zIndex: 1, minHeight: "calc(100vh - 64px)" }}>
      <main style={{ maxWidth: 960, margin: "0 auto", padding: "32px 24px 80px" }}>

        <h2 className="page-heading">Video Detection</h2>

        <div className="animate-fade-in">
          <div className="glass-card" style={{ padding: 28, marginBottom: 28 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
              <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Upload Dashcam Video</h3>
              {videoFile && (
                <button onClick={clearVideo} className="btn-secondary" style={{ padding: "6px 14px", fontSize: 13 }}>
                  <IconTrash /> Clear
                </button>
              )}
            </div>

            <div
              className={`upload-zone ${videoDragOver ? "drag-over" : ""}`}
              onClick={() => videoInputRef.current?.click()}
              onDragOver={onVideoDragOver}
              onDragLeave={onVideoDragLeave}
              onDrop={onVideoDrop}
              style={{
                padding: "48px 24px",
                textAlign: "center",
                minHeight: 200,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <div style={{ color: "var(--text-muted)", marginBottom: 16, position: "relative", zIndex: 1 }}><IconCamera /></div>
              <p style={{ margin: "0 0 6px", fontSize: 15, fontWeight: 600, color: "var(--text-primary)", position: "relative", zIndex: 1 }}>
                {videoFile ? videoFile.name : "Drop your dashcam video here"}
              </p>
              <p style={{ margin: 0, fontSize: 13, color: "var(--text-muted)", position: "relative", zIndex: 1 }}>
                {videoFile
                  ? `${(videoFile.size / 1024 / 1024).toFixed(1)} MB`
                  : "MP4, AVI, MOV, WebM — analyzed every 15th frame"}
              </p>
              <input ref={videoInputRef} type="file" accept="video/*" onChange={onVideoChange} style={{ display: "none" }} />
            </div>

            {videoError && (
              <div style={{ marginTop: 14, padding: "12px 16px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: "var(--radius-sm)", fontSize: 13, color: "#f87171", display: "flex", alignItems: "center", gap: 8 }}>
                <IconWarning /> {videoError}
              </div>
            )}

            <div style={{ marginTop: 20 }}>
              <button className="btn-primary" onClick={runVideoDetection} disabled={!videoFile || videoLoading} style={{ width: "100%" }}>
                {videoLoading ? (<><div className="spinner" /> Analyzing Video...</>) : (<><IconScan /> Analyze Dashcam Footage</>)}
              </button>
            </div>

            {videoLoading && (
              <div style={{ marginTop: 16 }}>
                <div className="shimmer-bar" />
                <p style={{ marginTop: 8, fontSize: 12, color: "var(--text-muted)", textAlign: "center" }}>
                  Processing frames through YOLOv8...
                </p>
              </div>
            )}
          </div>

          {/* Video Results */}
          {videoResult && (
            <div className="animate-fade-in">
              {/* Resource Metrics */}
              <div className="metrics-row">
                <div className="metric-card">
                  <div className="metric-value" style={{ color: "var(--accent)" }}>{videoResult.total_potholes}</div>
                  <div className="metric-label">Total Potholes</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value" style={{ color: "var(--info)" }}>{videoResult.total_frames_analyzed}</div>
                  <div className="metric-label">Frames Analyzed</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">{Number(videoResult.video_fps).toFixed(2)}</div>
                  <div className="metric-label">Video FPS</div>
                </div>
              </div>
              
              <div className="metrics-row" style={{ marginTop: -8, marginBottom: 18 }}>
                <div className="metric-card">
                  <div className="metric-value" style={{ color: "var(--success)" }}>{Number(videoResult.total_inference_time).toFixed(3)}s</div>
                  <div className="metric-label">Total Inference Time</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value" style={{ fontSize: 16 }}>
                    {videoResult.total_frames_analyzed > 0
                      ? `${(Number(videoResult.total_inference_time) / videoResult.total_frames_analyzed).toFixed(3)}s`
                      : "—"}
                  </div>
                  <div className="metric-label">Avg Inference/Frame</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value" style={{ fontSize: 16 }}>Every 15 frames</div>
                  <div className="metric-label">Sampling Rate</div>
                </div>
              </div>

              {/* Frame results */}
              <div className="glass-card" style={{ padding: 20 }}>
                <h3 style={{ margin: "0 0 16px", fontSize: 16, fontWeight: 600 }}>Frame-by-Frame Results</h3>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 16 }}>
                  {videoResult.frames.map((frame, i) => (
                    <div key={i} className="frame-card" style={{ cursor: "pointer" }} onClick={() => setSelectedFrameSrc(frame.annotated_image)}>
                      <img src={frame.annotated_image} alt={`Frame ${frame.frame_number}`} />
                      <div className="frame-card-info">
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                            Frame {frame.frame_number} · {frame.timestamp}s
                          </span>
                          <span style={{
                            fontSize: 12,
                            fontWeight: 700,
                            color: frame.count > 0 ? "var(--accent)" : "var(--success)",
                          }}>
                            {frame.count > 0 ? `${frame.count} pothole${frame.count > 1 ? "s" : ""}` : "Clear"}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Frame Viewer Modal */}
      {selectedFrameSrc && (
        <div style={{
          position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
          background: "rgba(0,0,0,0.8)", backdropFilter: "blur(8px)",
          zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center",
          padding: 20
        }} onClick={() => setSelectedFrameSrc(null)}>
          <div style={{ position: "relative", maxWidth: "90vw", maxHeight: "90vh" }}>
            <span style={{
              position: "absolute", top: -40, right: 0, color: "#fff", cursor: "pointer",
              fontSize: 24, fontWeight: "bold"
            }}>&times;</span>
            <img src={selectedFrameSrc} alt="Enlarged Frame" style={{
              maxWidth: "100%", maxHeight: "85vh",
              objectFit: "contain", borderRadius: 8, boxShadow: "0 10px 30px rgba(0,0,0,0.5)"
            }} onClick={e => e.stopPropagation()} />
          </div>
        </div>
      )}
    </div>
  );
}
