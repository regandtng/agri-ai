// src/pages/StatsPage.jsx
import { useState, useEffect } from "react";

const API = "http://localhost:5000/api";

const GRADE_COLOR = {
  "Xuất khẩu": "#16a34a",
  "Loại 1":    "#2563eb",
  "Loại 2":    "#d97706",
  "Loại bỏ":  "#dc2626",
};

export default function StatsPage() {
  const [stats, setStats]     = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/stats`)
      .then((r) => r.json())
      .then((d) => { setStats(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <div className="center-msg">⏳ Đang tải thống kê...</div>;
  if (!stats || stats.message)
    return <div className="center-msg">📊 Chưa có dữ liệu. Hãy phân tích ít nhất 1 ảnh.</div>;

  const grades  = stats.grade_distribution || {};
  const fruits  = stats.fruit_distribution || {};
  const maxFruit = Math.max(...Object.values(fruits), 1);

  return (
    <div className="stats-page">
      {/* Metric cards */}
      <div className="metric-grid">
        <MetricCard label="Tổng trái đã phân tích" value={stats.total_analyzed} />
        <MetricCard label="Số phiên làm việc"      value={stats.sessions} />
        <MetricCard label="Tỉ lệ xuất khẩu" value={`${grades["Xuất khẩu"]?.pct ?? 0}%`} color="#16a34a" />
        <MetricCard label="Tỉ lệ loại bỏ"   value={`${grades["Loại bỏ"]?.pct ?? 0}%`}  color="#dc2626" />
      </div>

      {/* Grade distribution */}
      <div className="stats-section">
        <h3>Phân bố chất lượng</h3>
        {Object.entries(grades).map(([grade, info]) => (
          <div key={grade} className="stat-bar-row">
            <span className="stat-bar-label">{grade}</span>
            <div className="stat-bar-track">
              <div className="stat-bar-fill" style={{ width: `${info.pct}%`, background: GRADE_COLOR[grade] }} />
            </div>
            <span className="stat-bar-val">{info.count} ({info.pct}%)</span>
          </div>
        ))}
      </div>

      {/* Fruit distribution */}
      <div className="stats-section">
        <h3>Loại trái cây phân tích nhiều nhất</h3>
        {Object.entries(fruits).map(([fruit, count]) => (
          <div key={fruit} className="stat-bar-row">
            <span className="stat-bar-label">{fruit}</span>
            <div className="stat-bar-track">
              <div className="stat-bar-fill" style={{ width: `${(count / maxFruit) * 100}%`, background: "#0891b2" }} />
            </div>
            <span className="stat-bar-val">{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MetricCard({ label, value, color }) {
  return (
    <div className="metric-card">
      <p className="metric-label">{label}</p>
      <p className="metric-value" style={{ color: color || "var(--text)" }}>{value}</p>
    </div>
  );
}
