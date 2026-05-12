"""
=============================================================
  GIAI ĐOẠN 5 — OpenCV đánh giá chất lượng trái cây
=============================================================
4 tiêu chí:
  1. Kích thước  → cv2.findContours()
  2. Màu sắc     → cv2.cvtColor() HSV
  3. Độ chín     → cv2.calcHist()
  4. Khuyết tật  → cv2.threshold() + morphology
"""

import cv2
import numpy as np
from dataclasses import dataclass, asdict


# ─── Ngưỡng màu HSV cho từng loại trái cây ──────────────────
# Format: (H_low, S_low, V_low), (H_high, S_high, V_high), tên màu chín
RIPENESS_PROFILES = {
    "xoai":       [((20, 100, 100), (35, 255, 255), "vàng")],
    "thanh_long": [((0, 100, 100), (10, 255, 255), "đỏ"),
                   ((150, 100, 100), (180, 255, 255), "đỏ")],
    "buoi":       [((15, 80, 80), (30, 255, 255), "vàng xanh")],
    "cam":        [((10, 150, 150), (25, 255, 255), "cam")],
    "quit":       [((10, 150, 150), (25, 255, 255), "cam")],
    "dua_hau":    [((35, 80, 80), (85, 255, 255), "xanh đậm")],
    "chuoi":      [((20, 100, 100), (30, 255, 255), "vàng")],
    "oi":         [((30, 60, 100), (50, 200, 255), "vàng xanh")],
    "mit":        [((15, 80, 80), (30, 255, 255), "vàng")],
    "sau_rieng":  [((15, 80, 80), (35, 255, 255), "vàng nâu")],
    "chom_chom":  [((0, 100, 100), (10, 255, 255), "đỏ")],
    "mang_cut":   [((120, 50, 30), (160, 200, 120), "tím")],
}

# Tiêu chuẩn kích thước xuất khẩu (diện tích pixel tương đối)
SIZE_THRESHOLDS = {
    "xoai":       (8000, 20000),   # (nhỏ, lớn) pixels²
    "thanh_long": (15000, 40000),
    "buoi":       (30000, 80000),
    "cam":        (8000, 25000),
    "quit":       (6000, 18000),
    "dua_hau":    (50000, 150000),
    "chuoi":      (5000, 15000),
    "oi":         (4000, 12000),
    "mit":        (40000, 120000),
    "sau_rieng":  (35000, 100000),
    "chom_chom":  (2000, 8000),
    "mang_cut":   (5000, 15000),
}


@dataclass
class QualityResult:
    size_label:    str   # "Nhỏ" / "Trung bình" / "Lớn"
    size_area:     int
    color_name:    str
    ripeness:      str   # "Xanh" / "Chín vừa" / "Chín" / "Quá chín"
    ripeness_pct:  float
    defect_pct:    float
    defect_level:  str   # "Không" / "Nhẹ" / "Nặng"
    grade:         str   # "Xuất khẩu" / "Loại 1" / "Loại 2" / "Loại bỏ"

    def to_dict(self):
        return asdict(self)


# ─── 1. Kích thước ──────────────────────────────────────────
def assess_size(crop_bgr: np.ndarray, fruit_class: str) -> tuple[str, int]:
    gray  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "Không xác định", 0

    area = int(max(cv2.contourArea(c) for c in contours))
    thresholds = SIZE_THRESHOLDS.get(fruit_class, (5000, 20000))

    if area < thresholds[0]:
        label = "Nhỏ"
    elif area > thresholds[1]:
        label = "Lớn"
    else:
        label = "Trung bình"

    return label, area


# ─── 2. Màu sắc ─────────────────────────────────────────────
def assess_color(crop_bgr: np.ndarray, fruit_class: str) -> tuple[str, float]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    profiles = RIPENESS_PROFILES.get(fruit_class, [])

    best_pct   = 0.0
    best_color = "Không xác định"
    total_px   = crop_bgr.shape[0] * crop_bgr.shape[1]

    for (lo, hi, color_name) in profiles:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        pct  = float(cv2.countNonZero(mask)) / total_px * 100
        if pct > best_pct:
            best_pct   = pct
            best_color = color_name

    return best_color, round(best_pct, 2)


# ─── 3. Độ chín ─────────────────────────────────────────────
def assess_ripeness(crop_bgr: np.ndarray, fruit_class: str) -> tuple[str, float]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # Tính histogram Saturation (S) & Value (V)
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()

    # Tỷ lệ pixel có S cao (màu rực) → chín
    total  = float(hsv.shape[0] * hsv.shape[1])
    rich_s = float(s_hist[128:].sum()) / total * 100  # S > 128
    bright = float(v_hist[100:].sum()) / total * 100  # V > 100

    # Điểm chín 0–100
    ripeness_score = rich_s * 0.6 + bright * 0.4

    if ripeness_score < 25:
        label = "Xanh"
    elif ripeness_score < 50:
        label = "Chín vừa"
    elif ripeness_score < 80:
        label = "Chín"
    else:
        label = "Quá chín"

    return label, round(ripeness_score, 2)


# ─── 4. Khuyết tật ──────────────────────────────────────────
def assess_defect(crop_bgr: np.ndarray) -> tuple[str, float]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Phát hiện vùng tối bất thường (vết thâm, vết thối)
    _, dark_mask = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)

    # Morphology để loại nhiễu nhỏ
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    total_px  = crop_bgr.shape[0] * crop_bgr.shape[1]
    defect_px = int(cv2.countNonZero(dark_mask))
    defect_pct = round(defect_px / total_px * 100, 2)

    if defect_pct < 2:
        level = "Không"
    elif defect_pct < 8:
        level = "Nhẹ"
    else:
        level = "Nặng"

    return level, defect_pct


# ─── 5. Xếp loại tổng thể ───────────────────────────────────
def grade_fruit(size: str, ripeness: str, defect_level: str) -> str:
    # Loại bỏ ngay nếu khuyết tật nặng hoặc quá chín / xanh
    if defect_level == "Nặng" or ripeness in ("Xanh", "Quá chín"):
        return "Loại bỏ"
    if defect_level == "Nhẹ" or size == "Nhỏ":
        return "Loại 2" if ripeness != "Chín" else "Loại 1"
    if size == "Lớn" and ripeness == "Chín" and defect_level == "Không":
        return "Xuất khẩu"
    if size == "Trung bình" and ripeness in ("Chín", "Chín vừa") and defect_level == "Không":
        return "Loại 1"
    return "Loại 2"


# ─── 6. Hàm tổng hợp — dùng trong backend ───────────────────
def evaluate_quality(crop_bgr: np.ndarray, fruit_class: str) -> QualityResult:
    """
    Đánh giá toàn bộ chất lượng 1 ảnh crop.
    Trả về QualityResult (có .to_dict()).
    """
    size_label, size_area = assess_size(crop_bgr, fruit_class)
    color_name, _         = assess_color(crop_bgr, fruit_class)
    ripeness, ripe_pct    = assess_ripeness(crop_bgr, fruit_class)
    defect_level, def_pct = assess_defect(crop_bgr)
    grade = grade_fruit(size_label, ripeness, defect_level)

    return QualityResult(
        size_label   = size_label,
        size_area    = size_area,
        color_name   = color_name,
        ripeness     = ripeness,
        ripeness_pct = ripe_pct,
        defect_pct   = def_pct,
        defect_level = defect_level,
        grade        = grade,
    )


# ─── 7. Visualize kết quả ───────────────────────────────────
def draw_quality_overlay(image_bgr: np.ndarray, detections: list) -> np.ndarray:
    """
    Vẽ bounding box + nhãn chất lượng lên ảnh gốc.
    detections: list dict từ pipeline (fruit_name, bbox, grade, confidence).
    """
    GRADE_COLORS = {
        "Xuất khẩu": (0, 200, 0),
        "Loại 1":    (0, 165, 255),
        "Loại 2":    (0, 255, 255),
        "Loại bỏ":   (0, 0, 255),
    }
    img = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        color = GRADE_COLORS.get(d.get("grade", "Loại 2"), (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d.get('fruit_vi','?')} {d.get('confidence', 0):.0f}% | {d.get('grade','?')}"
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img


# ─── Demo local ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python quality_assessment.py <image_path> <fruit_class>")
        print("Example: python quality_assessment.py test.jpg xoai")
        sys.exit(0)

    img_path     = sys.argv[1]
    fruit_class  = sys.argv[2]
    img          = cv2.imread(img_path)
    if img is None:
        print(f"❌ Không đọc được ảnh: {img_path}")
        sys.exit(1)

    result = evaluate_quality(img, fruit_class)
    print("\n📊 KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG:")
    for k, v in result.to_dict().items():
        print(f"   {k:15}: {v}")
