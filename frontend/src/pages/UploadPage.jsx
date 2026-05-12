// src/pages/UploadPage.jsx
import { useState, useRef } from "react";
import ResultCard from "../components/ResultCard";


const API = "http://localhost:5000/api";

const GRADE_COLOR = {
  "Xuất khẩu": "#16a34a",
  "Loại 1":    "#2563eb",
  "Loại 2":    "#d97706",
  "Loại bỏ":  "#dc2626",
};

export default function UploadPage({ onNewResult }) {
  const [preview, setPreview]     = useState(null);
  const [file, setFile]           = useState(null);
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const [dragging, setDragging]   = useState(false);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("image", file);
      const res  = await fetch(`${API}/predict`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Lỗi server");
      setResult(data);
      onNewResult?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="upload-page">
      {/* ── Upload zone ── */}
      {!result && (
        <div
          className={`drop-zone ${dragging ? "dragging" : ""} ${preview ? "has-preview" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => !preview && inputRef.current.click()}
        >
          <input
            ref={inputRef} type="file" accept="image/*" hidden
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {preview ? (
            <div className="preview-wrap">
              <img src={preview} alt="preview" className="preview-img" />
              <div className="preview-actions">
                <button className="btn-primary" onClick={(e) => { e.stopPropagation(); analyze(); }} disabled={loading}>
                  {loading ? "⏳ Đang phân tích..." : "🔍 Phân tích ngay"}
                </button>
                <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); reset(); }}>
                  Chọn ảnh khác
                </button>
              </div>
            </div>
          ) : (
            <div className="drop-hint">
              <div className="drop-icon">📷</div>
              <p className="drop-title">Kéo thả ảnh vào đây</p>
              <p className="drop-sub">hoặc nhấn để chọn từ máy tính</p>
              <p className="drop-sub">Hỗ trợ: JPG, PNG, WEBP</p>
            </div>
          )}
        </div>
      )}

      {/* ── Error ── */}
      {error && (
        <div className="error-box">
          ⚠️ {error}
          <button onClick={reset} style={{ marginLeft: 12 }}>Thử lại</button>
        </div>
      )}

      {/* ── Result ── */}
      {result && (
        <div className="result-wrap">
          <div className="result-header">
            <h2>Kết quả phân tích</h2>
            <div className="result-meta">
              <span>⏱ {result.processing_ms} ms</span>
              <span>🍎 {result.total_fruits} trái</span>
              <button className="btn-ghost" onClick={reset}>Phân tích ảnh khác</button>
            </div>
          </div>

          {/* Ảnh kết quả */}
          <div className="result-images">
            <div>
              <p className="img-label">Ảnh gốc</p>
              <img src={preview} alt="original" className="result-img" />
            </div>
            <div>
              <p className="img-label">Kết quả nhận diện</p>
              <img src={`${API}/result/${result.image_url?.split("/").pop()}`}
                   alt="annotated" className="result-img"
                   onError={(e) => { e.target.style.display = "none"; }} />
            </div>
          </div>

          {/* Tóm tắt theo loại */}
          <div className="grade-summary">
            {Object.entries(result.summary).map(([grade, count]) => (
              <div key={grade} className="grade-pill" style={{ borderColor: GRADE_COLOR[grade] }}>
                <span className="grade-count" style={{ color: GRADE_COLOR[grade] }}>{count}</span>
                <span className="grade-label">{grade}</span>
              </div>
            ))}
          </div>

          {/* Chi tiết từng trái */}
          <h3 style={{ marginBottom: 12 }}>Chi tiết từng trái</h3>
          <div className="cards-grid">
            {result.detections.map((d, i) => (
              <ResultCard key={i} detection={d} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
