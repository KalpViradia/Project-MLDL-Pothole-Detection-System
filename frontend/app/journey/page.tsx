export default function JourneyPage() {
  return (
    <div style={{ position: "relative", zIndex: 1, minHeight: "calc(100vh - 64px)" }}>
      <main style={{ maxWidth: 900, margin: "0 auto", padding: "32px 24px 80px" }}>

        <h2 className="page-heading">The Training Journey</h2>
        <p style={{ textAlign: "center", color: "var(--text-muted)", fontSize: 15, marginBottom: 32, maxWidth: 640, margin: "0 auto 36px" }}>
          Building this pothole detection system proved to be an exercise in resilience. 
          From hardware bottlenecks to exploding gradients, here is the chronological story of how the final YOLOv8 model came to be.
        </p>

        <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: 20 }}>

          {/* Phase 1 */}
          <div className="content-card" style={{ borderLeft: "4px solid var(--info)" }}>
            <h3 className="content-card-title">1. Initial Local Attempts & Hardware Hurdles</h3>
            <p className="content-card-text">
              The project began with a modest <strong>665-image dataset</strong> sourced from Kaggle. 
              Initial attempts to train a <strong>YOLOv8s</strong> model locally were stalled by library compatibility issues 
              preventing the detection of the laptop's GPU (a 95W TDP RTX 3050 4GB). 
            </p>
            <p className="content-card-text">
              Once the GPU was properly detected, the first overnight training session was abruptly terminated by a power outage. 
              Training resumed the following night, but evaluation revealed that the small dataset and lightweight YOLOv8s architecture 
              were not detecting potholes effectively.
            </p>
          </div>

          {/* Phase 2 */}
          <div className="content-card" style={{ borderLeft: "4px solid var(--accent)" }}>
            <h3 className="content-card-title">2. Massive Dataset Expansion</h3>
            <p className="content-card-text">
              Realizing data was the bottleneck, <strong>five distinct datasets were merged</strong> and heavily curated to improve variation. 
              Using custom Python scripts (archived in the <code>dataset_construction</code> module), the images were cleaned, standardized, 
              and scaled to a uniform <strong>768 × 768</strong> resolution. 
            </p>
            <p className="content-card-text">
              The final consolidated dataset swelled to <strong>18,976 images</strong>. These were rigorously fractured into a 
              <strong> 70% Training, 20% Test, and 10% Validation</strong> split to prevent data leakage and ensure robust evaluation.
            </p>
          </div>

          {/* Phase 3 */}
          <div className="content-card" style={{ borderLeft: "4px solid var(--danger)" }}>
            <h3 className="content-card-title">3. The Colab NaN Corruption Loop</h3>
            <p className="content-card-text">
              Scaling up to the heavier <strong>YOLOv8m</strong> model required shifting compute to Google Colab. 
              Fetching images directly from Google Drive caused severe I/O bottlenecks, so the dataset was compressed into a <code>.tar</code> file, 
              extracted locally inside the Colab instance, and cached entirely into RAM.
            </p>
            <p className="content-card-text">
              Training proceeded smoothly until <strong>epoch 51</strong>, where the assigned model weights mysteriously corrupted into <code>NaN</code> and <code>Inf</code> values, 
              forcing a hard crash at epoch 54. 
            </p>
            <div style={{ padding: "12px 16px", background: "rgba(239,68,68,0.08)", borderRadius: 8, marginTop: 12 }}>
              <p className="content-card-text" style={{ fontSize: 13, color: "#f87171", margin: 0 }}>
                <strong>The Endless Loop:</strong> Implemented checkpoint scanning to resume from uncorrupted states. This resulted in an endless cycle: 
                the model would cleanly resume from epoch 17, train up to epoch 24 or 33, but the next day all subsequent checkpoints would be flagged as corrupt, 
                forcing it back to epoch 17. Bypassing the corruption check outright just yielded the original epoch 54 crash, even when resuming from earlier stable epochs like 35 or 45.
              </p>
            </div>
          </div>

          {/* Phase 4 */}
          <div className="content-card" style={{ borderLeft: "4px solid #a855f7" }}>
            <h3 className="content-card-title">4. Kaggle, Supercomputers, & YOLOv8l</h3>
            <p className="content-card-text">
              Abandoning Colab, operations shifted to Kaggle to train the massive <strong>YOLOv8l</strong> architecture overnight. 
              A hard lesson in volatile environments was learned when progress from epoch 21 to 43 was completely lost because the checkpoints 
              were not downloaded from the <code>/working</code> directory before the session expired.
            </p>
            <p className="content-card-text" style={{ marginTop: 12 }}>
              Resuming the Kaggle training salvaged the run, though session limits caused the internal logging to severely overlap the numbering—resulting in the logs 
              jumping up to epoch 832 despite calculating to exactly <strong>102 true linear epochs</strong> manually from the <code>results.csv</code>. 
              Simultaneously, a parallel YOLOv8l instance was deployed on the college's supercomputer, grinding continuously undisturbed for <strong>43.8 hours</strong>.
            </p>
          </div>

          {/* Phase 5 */}
          <div className="content-card" style={{ borderLeft: "4px solid var(--success)" }}>
            <h3 className="content-card-title">5. Final Selection & Backend Evolution</h3>
            <p className="content-card-text">
              When evaluating the surviving checkpoints across all local, supercomputer, and Kaggle environments, the gracefully-resumed <strong>Kaggle YOLOv8l model</strong> surprisingly emerged victorious over the 43.8-hour supercomputer model. 
            </p>
            <div style={{ padding: "12px 16px", background: "rgba(16, 185, 129, 0.08)", borderRadius: 8, marginTop: 12, marginBottom: 12 }}>
              <p className="content-card-text" style={{ fontSize: 13, color: "var(--success)", margin: 0 }}>
                <strong>Why the Kaggle Model Won:</strong> The supercomputer model experienced early overfitting due to continuous undisturbed momentum. Conversely, the forced session restarts on Kaggle effectively acted like a poor-man's <em>Cosine Annealing Warm Restarts</em>. Every time the session resumed, the optimizer stepped out of local minimas, allowing the Kaggle model to converge perfectly at the <strong>68th true training epoch</strong> (out of 102), achieving a dominant peak <strong>mAP@50 of 0.636</strong>.
              </p>
            </div>
            <p className="content-card-text">
              With the best model finalized, the system backend (documented in <code>backend/notebooks</code>) needed deployment. 
              It was initially authored in <strong>Flask</strong>, but processing intensive large-scale dashcam video feed uploads proved disastrously slow. 
              The entire API layer was completely rewritten into asynchronous <strong>FastAPI</strong>. This dramatically unblocked synchronous video frame ingestion, 
              resulting in the highly-performant, real-time detection platform you see today.
            </p>
          </div>

        </div>
      </main>
    </div>
  );
}
