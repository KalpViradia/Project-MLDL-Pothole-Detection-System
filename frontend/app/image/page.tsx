"use client";

import { useState, useRef, useCallback, useEffect } from "react";

/* ── Types ─────────────────────────────────────────────────────────── */
interface Detection {
  label: string;
  confidence: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

interface DetectionResult {
  detections: Detection[];
  count: number;
  inference_time: number;
  annotated_image: string;
  image_size: { width: number; height: number };
}

/* ── Icons ─────────────────────────────────────────────────────────── */

function IconUpload() {
  return (
    <svg width="48" height="48" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
    </svg>
  );
}

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

function IconClock() {
  return (
    <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

function IconImage() {
  return (
    <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
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

function IconDownload() {
  return (
    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
    </svg>
  );
}

/* ── Main Component ────────────────────────────────────────────────── */

export default function ImagePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [urlErrorDetails, setUrlErrorDetails] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [deviceInfo, setDeviceInfo] = useState<string>("—");
  const fileInputRef = useRef<HTMLInputElement>(null);

  /* ── Fetch device info on mount ───────────────────────────────── */
  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "ok" && data.device) {
          setDeviceInfo(data.device);
        } else {
          setDeviceInfo("GPU / CPU");
        }
      })
      .catch(() => setDeviceInfo("Unknown"));
  }, []);

  /* ── File Selection ─────────────────────────────────────────────── */
  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPEG, PNG, WebP, AVIF, etc.)");
      return;
    }
    setSelectedFile(file);
    setImageUrl("");
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setUrlErrorDetails(null);
  }, []);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleFileSelect(e.target.files[0]);
  };

  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragOver(true); };
  const onDragLeave = () => setDragOver(false);
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]);
  };

  /* ── Detection ──────────────────────────────────────────────────── */
  const runDetection = async () => {
    if (!selectedFile && !imageUrl) return;
    setLoading(true);
    setError(null);
    setUrlErrorDetails(null);

    try {
      let res: Response;

      if (selectedFile) {
        const formData = new FormData();
        formData.append("file", selectedFile);
        res = await fetch("/api/detect", { method: "POST", body: formData });
      } else {
        res = await fetch("/api/detect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_url: imageUrl }),
        });
      }

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const rawError = body.error || `Server error ${res.status}`;

        if (imageUrl && !selectedFile) {
          setError("Could not load image from the provided URL.");
          setUrlErrorDetails(rawError);
        } else {
          setError(rawError);
        }
        return;
      }

      const data: DetectionResult = await res.json();
      setResult(data);
    } catch (err) {
      const rawMessage = err instanceof Error ? err.message : "Detection failed. Is the backend running?";

      if (imageUrl && !selectedFile) {
        setError("Could not load image from the provided URL.");
        setUrlErrorDetails(rawMessage);
      } else {
        setError(rawMessage);
      }
    } finally {
      setLoading(false);
    }
  };

  /* ── Clear ──────────────────────────────────────────────────────── */
  const clearImage = () => {
    setSelectedFile(null);
    setImageUrl("");
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setUrlErrorDetails(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  /* ── Download ───────────────────────────────────────────────────── */
  const downloadResult = () => {
    if (!result?.annotated_image) return;
    const link = document.createElement("a");
    link.href = result.annotated_image;
    link.download = `pothole_detection_${Date.now()}.jpg`;
    link.click();
  };

  /* ── Confidence color ───────────────────────────────────────────── */
  const confColor = (conf: number) => {
    if (conf >= 0.8) return "#22c55e";
    if (conf >= 0.5) return "#f97316";
    return "#ef4444";
  };

  /* ── Render ─────────────────────────────────────────────────────── */
  return (
    <div style={{ position: "relative", zIndex: 1, minHeight: "calc(100vh - 64px)" }}>
      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "32px 24px 80px" }}>

        <h2 className="page-heading">Image Detection</h2>

        <div
          className="animate-fade-in"
          style={{
            display: "grid",
            gridTemplateColumns: result ? "1fr 1fr" : "1fr",
            gap: 28,
            maxWidth: result ? 1200 : 680,
            margin: "0 auto",
            transition: "all 0.4s ease",
          }}
        >
          {/* Left: Upload */}
          <div>
            <div className="glass-card" style={{ padding: 28 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
                <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Upload Image</h3>
                {selectedFile && (
                  <button onClick={clearImage} className="btn-secondary" style={{ padding: "6px 14px", fontSize: 13 }}>
                    <IconTrash /> Clear
                  </button>
                )}
              </div>

              <div
                className={`upload-zone ${dragOver ? "drag-over" : ""}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                style={{
                  padding: previewUrl ? 0 : "48px 24px",
                  textAlign: "center",
                  minHeight: previewUrl ? "auto" : 240,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                }}
                id="upload-zone"
              >
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Preview"
                    style={{ width: "100%", height: "auto", maxHeight: 400, objectFit: "contain", borderRadius: "var(--radius)" }}
                  />
                ) : (
                  <>
                    <div style={{ color: "var(--text-muted)", marginBottom: 16, position: "relative", zIndex: 1 }}><IconUpload /></div>
                    <p style={{ margin: "0 0 6px", fontSize: 15, fontWeight: 600, color: "var(--text-primary)", position: "relative", zIndex: 1 }}>
                      Drop your road image here
                    </p>
                    <p style={{ margin: 0, fontSize: 13, color: "var(--text-muted)", position: "relative", zIndex: 1 }}>
                      or click to browse — JPEG, PNG, WebP, AVIF
                    </p>
                  </>
                )}
                <input ref={fileInputRef} type="file" accept="image/*" onChange={onFileChange} style={{ display: "none" }} />
              </div>

              <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ flex: 1, position: "relative" }}>
                  <div style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", pointerEvents: "none" }}>
                    🔗
                  </div>
                  <input
                    type="url"
                    placeholder="Or paste an image URL..."
                    value={imageUrl}
                    onChange={(e) => {
                      setImageUrl(e.target.value);
                      if (e.target.value) {
                        setSelectedFile(null);
                        setPreviewUrl(e.target.value);
                        setResult(null);
                      } else if (!selectedFile) {
                        setPreviewUrl(null);
                      }
                    }}
                    className="input-field"
                    style={{ width: "100%", paddingLeft: 36 }}
                  />
                </div>
              </div>

              {selectedFile && (
                <div style={{ marginTop: 14, padding: "10px 14px", background: "var(--bg-secondary)", borderRadius: "var(--radius-sm)", fontSize: 13, color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 8 }}>
                  <IconImage />
                  <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{selectedFile.name}</span>
                  <span style={{ color: "var(--text-muted)" }}>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
              )}

              {/* URL Error with friendly message */}
              {error && urlErrorDetails && (
                <div style={{
                  marginTop: 14,
                  padding: "16px 18px",
                  background: "rgba(239,68,68,0.06)",
                  border: "1px solid rgba(239,68,68,0.15)",
                  borderRadius: "var(--radius-sm)",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, color: "#f87171", fontSize: 14, fontWeight: 600 }}>
                    <IconWarning /> {error}
                  </div>
                  <p style={{ margin: "0 0 8px", fontSize: 13, color: "var(--text-secondary)" }}>
                    This can happen when:
                  </p>
                  <ul style={{ margin: "0 0 12px", paddingLeft: 20, fontSize: 13, color: "var(--text-muted)", lineHeight: 1.8 }}>
                    <li>The link is broken or expired</li>
                    <li>The image is private or requires authentication</li>
                    <li>The URL does not directly point to an image file</li>
                  </ul>
                  <details className="collapsible-details">
                    <summary>Technical details</summary>
                    <pre style={{
                      marginTop: 8,
                      padding: "10px 14px",
                      background: "rgba(0,0,0,0.3)",
                      borderRadius: 8,
                      fontSize: 12,
                      color: "var(--text-muted)",
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-all",
                      overflowX: "auto",
                    }}>
                      {urlErrorDetails}
                    </pre>
                  </details>
                </div>
              )}

              {/* Generic error (non-URL) */}
              {error && !urlErrorDetails && (
                <div style={{ marginTop: 14, padding: "12px 16px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: "var(--radius-sm)", fontSize: 13, color: "#f87171", display: "flex", alignItems: "center", gap: 8 }}>
                  <IconWarning /> {error}
                </div>
              )}

              {/* Action buttons */}
              <div style={{ marginTop: 20, display: "flex", gap: 12, flexWrap: "wrap" }}>
                <button className="btn-primary" onClick={runDetection} disabled={(!selectedFile && !imageUrl) || loading} style={{ flex: 1, minWidth: 160 }}>
                  {loading ? (<><div className="spinner" /> Analyzing...</>) : (<><IconScan /> Detect Potholes</>)}
                </button>
                {result && (
                  <button className="btn-secondary" onClick={downloadResult} style={{ minWidth: 120 }}>
                    <IconDownload /> Download
                  </button>
                )}
              </div>

              {loading && (
                <div style={{ marginTop: 16 }}>
                  <div className="shimmer-bar" />
                  <p style={{ marginTop: 8, fontSize: 12, color: "var(--text-muted)", textAlign: "center" }}>
                    Running YOLOv8 inference...
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Right: Results */}
          {result && (
            <div className="animate-fade-in">
              {/* Resource Metrics */}
              <div className="metrics-row">
                <div className="metric-card">
                  <div className="metric-value" style={{ color: result.detections.length > 0 ? "var(--accent)" : "var(--success)" }}>{result.detections.length}</div>
                  <div className="metric-label">Detections</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value" style={{ color: "var(--info)" }}>{result.inference_time.toFixed(3)}s</div>
                  <div className="metric-label">Inference Time</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">{result.image_size.width}×{result.image_size.height}</div>
                  <div className="metric-label">Resolution</div>
                </div>
              </div>
              
              <div className="metrics-row" style={{ marginTop: -8, marginBottom: 18 }}>
                <div className="metric-card">
                  <div className="metric-value" style={{ fontSize: 16 }}>{deviceInfo}</div>
                  <div className="metric-label">Device Used</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value" style={{ fontSize: 16 }}>
                    {selectedFile ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB` : "—"}
                  </div>
                  <div className="metric-label">Image File Size</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value" style={{ fontSize: 16 }}>0.35</div>
                  <div className="metric-label">Conf Thresh</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value" style={{ fontSize: 16 }}>0.45</div>
                  <div className="metric-label">IoU Thresh</div>
                </div>
              </div>

              <div className="glass-card" style={{ padding: 16, marginBottom: 16 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                  <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Detection Result</h3>
                  <div className="detection-badge"><IconWarning /> {result.count} detected</div>
                </div>
                <div className="result-image-container">
                  <img src={result.annotated_image} alt="Annotated result" style={{ width: "100%", height: "auto" }} />
                </div>
              </div>

              {result.detections.length > 0 && (
                <div className="glass-card" style={{ padding: 16 }}>
                  <h3 style={{ margin: "0 0 14px", fontSize: 15, fontWeight: 600 }}>Detections</h3>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {result.detections.map((det, i) => (
                      <div key={i} className="detection-item">
                        <div style={{ flex: 1 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                            <span style={{ width: 22, height: 22, borderRadius: 6, background: "rgba(249,115,22,0.15)", color: "var(--accent)", fontSize: 11, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center" }}>{i + 1}</span>
                            <span style={{ fontWeight: 600, fontSize: 14, textTransform: "capitalize" }}>{det.label}</span>
                          </div>
                          <div className="confidence-bar" style={{ width: 140 }}>
                            <div className="confidence-bar-fill" style={{ width: `${det.confidence * 100}%`, background: `linear-gradient(90deg, ${confColor(det.confidence)}, ${confColor(det.confidence)}dd)` }} />
                          </div>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <div style={{ fontSize: 18, fontWeight: 700, color: confColor(det.confidence) }}>{(det.confidence * 100).toFixed(1)}%</div>
                          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>confidence</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.detections.length === 0 && (
                <div className="glass-card" style={{ padding: "32px 24px", textAlign: "center" }}>
                  <div style={{ fontSize: 40, marginBottom: 12 }}>✅</div>
                  <h3 style={{ margin: "0 0 6px", fontSize: 17, fontWeight: 700 }}>Road Looks Good!</h3>
                  <p style={{ margin: 0, fontSize: 14, color: "var(--text-secondary)" }}>No potholes were detected in this image.</p>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
