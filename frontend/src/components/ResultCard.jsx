// src/components/ResultCard.jsx
const GRADE_COLOR = {
  "Xuất khẩu": "#16a34a",
  "Loại 1":    "#2563eb",
  "Loại 2":    "#d97706",
  "Loại bỏ":  "#dc2626",
};
const GRADE_BG = {
  "Xuất khẩu": "#f0fdf4",
  "Loại 1":    "#eff6ff",
  "Loại 2":    "#fffbeb",
  "Loại bỏ":  "#fef2f2",
};

export default function ResultCard({ detection: d }) {
  const q = d.quality || {};
  return (
    <div className="result-card" style={{ borderTopColor: GRADE_COLOR[d.grade] }}>
      <div className="card-top" style={{ background: GRADE_BG[d.grade] }}>
        <div>
          <p className="card-fruit">{d.fruit_vi}</p>
          <p className="card-conf">{d.confidence}% độ tin cậy</p>
        </div>
        <span className="card-grade" style={{ background: GRADE_COLOR[d.grade] }}>
          {d.grade}
        </span>
      </div>
      <div className="card-body">
        <Row label="Kích thước"  value={q.size_label}  />
        <Row label="Màu sắc"     value={q.color_name}  />
        <Row label="Độ chín"     value={q.ripeness}    />
        <Row label="Khuyết tật"  value={`${q.defect_level} (${q.defect_pct}%)`} />
      </div>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="card-row">
      <span className="card-row-label">{label}</span>
      <span className="card-row-value">{value}</span>
    </div>
  );
}
