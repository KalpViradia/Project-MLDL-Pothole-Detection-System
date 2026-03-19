"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();

  const links = [
    { href: "/image", label: "Image" },
    { href: "/video", label: "Video" },
    { href: "/journey", label: "Journey" },
    { href: "/about", label: "About" },
  ];

  return (
    <header className="navbar">
      <div className="navbar-inner">
        <Link href="/" className="navbar-brand">
          <div className="navbar-logo">🕳️</div>
          <div>
            <div className="navbar-title">Pothole Detector</div>
            <div className="navbar-subtitle">YOLOv8 · Deep Learning</div>
          </div>
        </Link>

        <nav className="navbar-links">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`nav-link ${pathname === link.href ? "active" : ""}`}
            >
              {link.label}
            </Link>
          ))}
        </nav>

        <div className="navbar-badge">● Model Loaded</div>
      </div>
    </header>
  );
}
