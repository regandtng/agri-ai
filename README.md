
#  Agri-AI ĐBSCL — Hệ thống phân loại trái cây xuất khẩu
## CẤU TRÚC PROJECT
agri-ai/
├── dataset/
│   ├── data.yaml           # Cấu hình YOLO
│   ├── train/images/       # Ảnh train (bạn tự bỏ vào)
│   ├── valid/images/
│   └── test/images/
│
├── detection/yolov8/
│   └── train_yolo.py       # Train + predict YOLO
│
├── classification/efficientnet/
│   └── train_efficientnet.py  # Train + predict EfficientNet
│
├── quality/opencv/
│   └── quality_assessment.py  # Đánh giá chất lượng
│
├── backend/
│   └── app.py              # Flask API
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx / App.css
│   │   ├── pages/          # UploadPage, HistoryPage, StatsPage
│   │   └── components/     # ResultCard
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
└── models/                 # Model sau khi train lưu ở đây

---

## GIAI ĐOẠN 1 — Cài đặt môi trường

### Python (3.10)
```bash
pip install ultralytics tensorflow flask flask-cors opencv-python numpy pillow matplotlib
```

### NodeJS (v18+)
```bash
cd frontend
npm install
```

---

## GIAI ĐOẠN 2 — Chuẩn bị Dataset

1. Vào https://roboflow.com → tạo project mới
2. Upload ảnh 12 loại trái cây (500–1000 ảnh/loại)
3. Annotate theo format YOLO
4. Export → "YOLOv8" → download ZIP
5. Giải nén vào thư mục `dataset/`:
   ```
   dataset/
   ├── train/images/   ← ảnh train
   ├── train/labels/   ← label .txt YOLO
   ├── valid/images/
   ├── valid/labels/
   └── data.yaml       ← file này đã có sẵn
   ```

---

## GIAI ĐOẠN 3 — Train YOLOv8n

```bash
python detection/yolov8/train_yolo.py

# Đánh giá sau khi train:
python detection/yolov8/train_yolo.py eval
```

Model tốt: mAP@0.5 > 0.85

---

## GIAI ĐOẠN 4 — Train EfficientNet-B0

Tạo thêm thư mục classification trong dataset:
```
dataset/train/xoai/        ← ảnh xoài
dataset/train/thanh_long/  ← ảnh thanh long
...
```

```bash
python classification/efficientnet/train_efficientnet.py
```

Model tốt: val_accuracy > 90%

---

## GIAI ĐOẠN 5 — Test OpenCV quality

```bash
python quality/opencv/quality_assessment.py path/to/fruit.jpg xoai
```

---

## GIAI ĐOẠN 6 — Chạy Flask Backend

```bash
python backend/app.py
```

API chạy tại: http://localhost:5000

Test nhanh:
```bash
curl http://localhost:5000/api/health
```

---

## GIAI ĐOẠN 7 — Chạy React Frontend

```bash
cd frontend
npm run dev
```

Web chạy tại: http://localhost:3000

---

## LUỒNG DỮ LIỆU

Upload ảnh (React)
    ↓ POST /api/predict
Flask nhận ảnh
    ↓
YOLOv8n detect → Crop từng trái
    ↓
EfficientNet-B0 phân loại giống
    ↓
OpenCV đánh giá: kích thước, màu, độ chín, khuyết tật
    ↓
Xếp loại: Xuất khẩu / Loại 1 / Loại 2 / Loại bỏ
    ↓
Trả JSON + ảnh kết quả
    ↓
React hiển thị Dashboard

---

## MẪU JSON TRẢ VỀ

```json
{
  "id": "abc123",
  "timestamp": "2025-01-15T10:30:00",
  "processing_ms": 234,
  "total_fruits": 3,
  "detections": [
    {
      "id": "d1a2b3c4",
      "fruit": "xoai",
      "fruit_vi": "Xoài",
      "confidence": 97.5,
      "bbox": [120, 80, 340, 290],
      "grade": "Xuất khẩu",
      "quality": {
        "size_label": "Lớn",
        "size_area": 21000,
        "color_name": "vàng",
        "ripeness": "Chín",
        "ripeness_pct": 72.3,
        "defect_pct": 0.8,
        "defect_level": "Không",
        "grade": "Xuất khẩu"
      }
    }
  ],
  "summary": {
    "Xuất khẩu": 2,
    "Loại 1": 1,
    "Loại 2": 0,
    "Loại bỏ": 0
  }
}
```

---

## TIPS

- Chưa có model? Flask vẫn chạy được, chỉ return empty detections
- Dataset ít? Dùng augmentation mạnh trong EfficientNet (đã cấu hình sẵn)
- GPU? Đặt device=0 trong train_yolo.py, tensorflow tự nhận GPU
- Deploy? Dùng gunicorn thay flask dev server
