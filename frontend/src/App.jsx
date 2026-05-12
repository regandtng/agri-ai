// src/App.jsx
import { useState } from "react";
import UploadPage from "./pages/UploadPage";
import HistoryPage from "./pages/HistoryPage";
import StatsPage from "./pages/StatsPage";
import "./App.css";

export default function App() {
  const [tab, setTab] = useState("upload");
  const [refreshKey, setRefreshKey] = useState(0);

  const onNewResult = () => setRefreshKey((k) => k + 1);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-brand">
          <span className="header-icon">🌿</span>
          <div>
            <h1>Agri-AI ĐBSCL</h1>
            <p>Hệ thống phân loại & đánh giá chất lượng trái cây xuất khẩu</p>
          </div>
        </div>
        <nav className="header-nav">
          {[
            { id: "upload",  label: "Phân tích ảnh", icon: "📷" },
            { id: "history", label: "Lịch sử",       icon: "📋" },
            { id: "stats",   label: "Thống kê",       icon: "📊" },
          ].map((t) => (
            <button
              key={t.id}
              className={`nav-btn ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              <span>{t.icon}</span> {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="app-main">
        {tab === "upload"  && <UploadPage onNewResult={onNewResult} />}
        {tab === "history" && <HistoryPage key={refreshKey} />}
        {tab === "stats"   && <StatsPage   key={refreshKey} />}
      </main>
    </div>
  );
}
