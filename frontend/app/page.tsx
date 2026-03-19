import Link from "next/link";

export default function LandingPage() {
  return (
    <div style={{ position: "relative", zIndex: 1, minHeight: "calc(100vh - 64px)" }}>
      <section
        className="animate-fade-in-up"
        style={{
          maxWidth: 760,
          margin: "0 auto",
          padding: "100px 24px 80px",
          textAlign: "center",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <div style={{
          display: "inline-block",
          padding: "6px 16px",
          borderRadius: 999,
          fontSize: 13,
          fontWeight: 600,
          background: "rgba(249, 115, 22, 0.1)",
          color: "#fb923c",
          border: "1px solid rgba(249, 115, 22, 0.2)",
          marginBottom: 28,
        }}>
          Deep Learning Project · YOLOv8
        </div>

        <h1 style={{
          fontSize: "clamp(32px, 5vw, 52px)",
          fontWeight: 800,
          lineHeight: 1.08,
          margin: "0 0 20px",
          background: "linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          letterSpacing: "-0.03em",
        }}>
          AI-Powered Road Damage Detection
        </h1>

        <p style={{
          fontSize: 18,
          color: "var(--text-secondary)",
          maxWidth: 560,
          margin: "0 0 48px",
          lineHeight: 1.7,
        }}>
          Deep learning–based pothole detection using YOLOv8 and FastAPI.
        </p>

        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
          <Link href="/image" className="btn-primary" style={{ padding: "16px 36px", fontSize: 16, textDecoration: "none" }}>
            Go to Image Detection
          </Link>
          <Link href="/video" className="btn-secondary" style={{ padding: "16px 36px", fontSize: 16, textDecoration: "none" }}>
            Go to Video Detection
          </Link>
        </div>
      </section>

      <footer style={{
        textAlign: "center",
        padding: "24px",
        borderTop: "1px solid var(--border)",
        fontSize: 13,
        color: "var(--text-muted)",
      }}>
        Pothole Detection System · YOLOv8 Deep Learning
      </footer>
    </div>
  );
}
