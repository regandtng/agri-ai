"""
=============================================================
  GIAI ĐOẠN 3 — Train YOLOv8n detect 12 loại trái cây ĐBSCL
=============================================================
Chạy:
    python detection/yolov8/train_yolo.py
"""

import os
from pathlib import Path
from ultralytics import YOLO


# ─── Cấu hình ───────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
DATA_YAML  = BASE_DIR / "dataset" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_CONFIG = {
    "model":   "yolov8n.pt",   # Nano – nhẹ, phù hợp Jetson Nano sau này
    "data":    str(DATA_YAML),
    "epochs":  50,
    "imgsz":   640,
    "batch":   16,
    "lr0":     0.01,
    "patience": 10,             # Early stopping
    "project":  str(MODELS_DIR / "yolo_runs"),
    "name":    "fruit_detect",
    "save":    True,
    "plots":   True,
    "device":  "cpu",               # GPU; đổi thành 'cpu' nếu không có GPU
}


def train():
    print("=" * 60)
    print("  Train YOLOv8n — Phát hiện 12 loại trái cây ĐBSCL")
    print("=" * 60)

    # Tải model pretrained
    model = YOLO(TRAIN_CONFIG["model"])

    # Bắt đầu train
    results = model.train(**TRAIN_CONFIG)

    # Lưu best model ra thư mục chung
    best_weights = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"] / "weights" / "best.pt"
    dest = MODELS_DIR / "yolo_best.pt"
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, dest)
        print(f"\n✅ Best model lưu tại: {dest}")

    return results


def evaluate():
    """Đánh giá model sau khi train."""
    model_path = MODELS_DIR / "yolo_best.pt"
    if not model_path.exists():
        print("❌ Chưa có model. Hãy train trước!")
        return

    model   = YOLO(str(model_path))
    metrics = model.val(data=str(DATA_YAML))

    print("\n📊 KẾT QUẢ ĐÁNH GIÁ:")
    print(f"   mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   Precision   : {metrics.box.mp:.4f}")
    print(f"   Recall      : {metrics.box.mr:.4f}")


def predict_image(image_path: str):
    """
    Dự đoán 1 ảnh, trả về list bounding boxes.
    Dùng trong pipeline backend.
    """
    model_path = MODELS_DIR / "yolo_best.pt"
    if not model_path.exists():
        raise FileNotFoundError("Model YOLO chưa được train!")

    model   = YOLO(str(model_path))
    results = model(image_path, conf=0.5, iou=0.45)

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class_id":   int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox":       box.xyxy[0].tolist(),  # [x1,y1,x2,y2]
            })

    return detections


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate()
    else:
        train()
