import { useState, useEffect } from "react";

const API = "http://localhost:5000/api";

const GRADE_DOT = {
  "Xuất khẩu": "#16a34a",
  "Loại 1":    "#2563eb",
  "Loại 2":    "#d97706",
  "Loại bỏ":  "#dc2626",
};

export default function HistoryPage() {
  const [data, setData]       = useState(null);
  const [page, setPage]       = useState(1);
  const [loading, setLoading] = useState(true);

  const load = async (p = 1) => {
    setLoading(true);
    try {
      const res  = await fetch(`${API}/history?page=${p}&limit=10`);
      const json = await res.json();
      setData(json);
      setPage(p);
    } catch { setData(null); }
    setLoading(false);
  };

  useEffect(() => { load(1); }, []);

  if (loading) return <div className="center-msg">⏳ Đang tải lịch sử...</div>;
  if (!data || !data.items?.length)
    return <div className="center-msg">📭 Chưa có lịch sử phân tích nào.</div>;

  const totalPages = Math.ceil(data.total / 10);

  return (
    <div className="history-page">
      <h2>Lịch sử phân tích ({data.total} lần)</h2>
      <div style={{ overflowX: "auto" }}>
        <table className="history-table">
          <thead>
            <tr>
              <th>Thời gian</th>
              <th>Số trái</th>
              <th>Xuất khẩu</th>
              <th>Loại 1</th>
              <th>Loại 2</th>
              <th>Loại bỏ</th>
              <th>Xử lý</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((r) => (
              <tr key={r.id}>
                <td>{new Date(r.timestamp).toLocaleString("vi-VN")}</td>
                <td><strong>{r.total_fruits}</strong></td>
                {["Xuất khẩu","Loại 1","Loại 2","Loại bỏ"].map((g) => (
                  <td key={g}>
                    <span style={{ color: GRADE_DOT[g], fontWeight: 600 }}>
                      {r.summary?.[g] ?? 0}
                    </span>
                  </td>
                ))}
                <td>{r.processing_ms} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div className="pagination">
          <button disabled={page <= 1} onClick={() => load(page - 1)}>← Trước</button>
          <span>Trang {page} / {totalPages}</span>
          <button disabled={page >= totalPages} onClick={() => load(page + 1)}>Tiếp →</button>
        </div>
      )}
    </div>
  );
}