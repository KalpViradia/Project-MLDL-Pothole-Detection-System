export default function AboutPage() {
  return (
    <div style={{ position: "relative", zIndex: 1, minHeight: "calc(100vh - 64px)" }}>
      <main style={{ maxWidth: 900, margin: "0 auto", padding: "32px 24px 80px" }}>

        <h2 className="page-heading">About</h2>

        <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: 20 }}>

          {/* Project Overview */}
          <div className="content-card">
            <h3 className="content-card-title">Project Overview</h3>
            <p className="content-card-text">
              An end-to-end deep learning system for automated pothole detection from road images and dashcam video.
              The system accepts image uploads or URLs, runs inference through a YOLOv8 object detection model,
              and returns annotated images with bounding boxes, confidence scores, and detection counts.
              Video analysis is performed via frame sampling to balance coverage and processing time.
            </p>
          </div>

          {/* System Architecture */}
          <div className="content-card">
            <h3 className="content-card-title">System Architecture</h3>
            <p className="content-card-text">
              The system is structured as a decoupled client-server application:
            </p>
            <div className="arch-flow">
              <div className="arch-node">
                <div className="arch-node-title">Frontend</div>
                <div className="arch-node-desc">Next.js · React</div>
              </div>
              <div className="arch-arrow">→</div>
              <div className="arch-node">
                <div className="arch-node-title">Backend</div>
                <div className="arch-node-desc">FastAPI · Python</div>
              </div>
              <div className="arch-arrow">→</div>
              <div className="arch-node">
                <div className="arch-node-title">Model</div>
                <div className="arch-node-desc">YOLOv8 · PyTorch</div>
              </div>
              <div className="arch-arrow">→</div>
              <div className="arch-node">
                <div className="arch-node-title">Device</div>
                <div className="arch-node-desc">CUDA GPU / CPU</div>
              </div>
            </div>
            <p className="content-card-text" style={{ marginTop: 16 }}>
              The frontend communicates with the backend via REST API. Image and video data is sent as multipart form data
              or JSON payloads. The backend loads the model on startup, runs inference, annotates results, and returns
              base64-encoded images with detection metadata.
            </p>
          </div>

          {/* Model Details */}
          <div className="content-card">
            <h3 className="content-card-title">Model Details</h3>
            <div className="specs-grid">
              <div className="spec-item">
                <div className="spec-label">Architecture</div>
                <div className="spec-value">YOLOv8</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">Framework</div>
                <div className="spec-value">PyTorch (Ultralytics)</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">Confidence Threshold</div>
                <div className="spec-value">0.35</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">IoU Threshold</div>
                <div className="spec-value">0.45</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">Input Size</div>
                <div className="spec-value">640 × 640</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">Half Precision</div>
                <div className="spec-value">Enabled (CUDA only)</div>
              </div>
            </div>
          </div>

          {/* Training Summary */}
          <div className="content-card">
            <h3 className="content-card-title">Training Summary</h3>
            <div className="specs-grid">
              <div className="spec-item">
                <div className="spec-label">Initial Dataset</div>
                <div className="spec-value">665 images</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">Expanded Dataset</div>
                <div className="spec-value">18,976 images (5 merged datasets)</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">Training Environment</div>
                <div className="spec-value">Local GPU</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">Optimization</div>
                <div className="spec-value">Hyperparameter tuning + augmentation</div>
              </div>
            </div>
            <p className="content-card-text" style={{ marginTop: 16 }}>
              Training started with a 665-image dataset and was later expanded by merging five curated datasets
              to improve generalization across diverse road conditions, lighting, and camera angles.
              Optimization included systematic hyperparameter sweeps and data augmentation tuning.
            </p>
          </div>

          {/* Model Performance Metrics */}
          <div className="content-card">
            <h3 className="content-card-title">Model Performance Metrics</h3>
            <div className="specs-grid">
              <div className="spec-item">
                <div className="spec-label">mAP@50</div>
                <div className="spec-value">0.636</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">mAP@50-95</div>
                <div className="spec-value">0.341</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">Precision</div>
                <div className="spec-value">0.687</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">Recall</div>
                <div className="spec-value">0.581</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">F1 Score</div>
                <div className="spec-value">0.629</div>
              </div>
              <div className="spec-item">
                <div className="spec-label">Best Epoch</div>
                <div className="spec-value">68 / 102</div>
              </div>
            </div>
            <p className="content-card-text" style={{ marginTop: 12, fontSize: 13, color: "var(--text-muted)" }}>
              Metrics reported at the best mAP@50 checkpoint. Training conducted over 102 actual epochs (numbering jumps up to 832 due to session resumes) across Kaggle and local GPU environments.
            </p>
          </div>

        </div>
      </main>
    </div>
  );
}
