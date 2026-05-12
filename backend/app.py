"""
=============================================================
  GIAI ĐOẠN 6 — Flask API Backend
=============================================================
Routes:
  POST /api/predict   — Upload ảnh → phân tích toàn bộ
  GET  /api/history   — Lịch sử dự đoán
  GET  /api/stats     — Thống kê tổng hợp
  GET  /api/health    — Kiểm tra server

Chạy:
    pip install flask flask-cors pillow
    python backend/app.py
"""

import os, sys, json, uuid, time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from quality.opencv.quality_assessment import evaluate_quality, draw_quality_overlay

app = Flask(__name__)
CORS(app)  # Cho phép ReactJS gọi API

# ─── Thư mục lưu ảnh upload & kết quả ──────────────────────
UPLOAD_DIR = BASE_DIR / "backend" / "uploads"
RESULT_DIR = BASE_DIR / "backend" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Lịch sử (lưu RAM, production nên dùng DB) ─────────────
history_store: list[dict] = []

# ─── Lazy-load models ───────────────────────────────────────
_yolo_model       = None
_efficientnet_model = None
_class_mapping    = None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        model_path = BASE_DIR / "models" / "yolo_best.pt"
        if not model_path.exists():
            return None
        _yolo_model = YOLO(str(model_path))
    return _yolo_model


def get_classifier():
    global _efficientnet_model, _class_mapping
    if _efficientnet_model is None:
        try:
            from classification.efficientnet.train_efficientnet import load_classifier
            _efficientnet_model, _class_mapping = load_classifier()
        except Exception:
            return None, None
    return _efficientnet_model, _class_mapping


# ─── Helper: xử lý pipeline ─────────────────────────────────
def run_pipeline(image_bgr: np.ndarray) -> dict:
    detections = []
    yolo = get_yolo()

    if yolo is not None:
        # YOLO detect
        results = yolo(image_bgr, conf=0.45, iou=0.45, verbose=False)
        boxes   = results[0].boxes if results else []

        clf, mapping = get_classifier()

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            class_id   = int(box.cls[0])
            yolo_name  = results[0].names[class_id]
            conf_yolo  = float(box.conf[0])

            # Crop
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # EfficientNet classify
            if clf is not None:
                from classification.efficientnet.train_efficientnet import predict_crop
                cls_result = predict_crop(crop, clf, mapping)
                fruit_class = cls_result["class_name"]
                fruit_vi    = cls_result["class_name_vi"]
                confidence  = cls_result["confidence"]
            else:
                fruit_class = yolo_name
                fruit_vi    = yolo_name
                confidence  = round(conf_yolo * 100, 2)

            # OpenCV quality
            quality = evaluate_quality(crop, fruit_class)

            detections.append({
                "id":          str(uuid.uuid4())[:8],
                "fruit":       fruit_class,
                "fruit_vi":    fruit_vi,
                "confidence":  confidence,
                "bbox":        [x1, y1, x2, y2],
                "quality":     quality.to_dict(),
                "grade":       quality.grade,
            })
    else:
        # Fallback: không có YOLO, chạy EfficientNet trên cả ảnh
        clf, mapping = get_classifier()
        if clf is not None:
            from classification.efficientnet.train_efficientnet import predict_crop
            cls_result = predict_crop(image_bgr, clf, mapping)
            quality    = evaluate_quality(image_bgr, cls_result["class_name"])
            detections.append({
                "id":         str(uuid.uuid4())[:8],
                "fruit":      cls_result["class_name"],
                "fruit_vi":   cls_result["class_name_vi"],
                "confidence": cls_result["confidence"],
                "bbox":       [0, 0, image_bgr.shape[1], image_bgr.shape[0]],
                "quality":    quality.to_dict(),
                "grade":      quality.grade,
            })

    return {"detections": detections, "count": len(detections)}


# ─── Routes ─────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "yolo":        (BASE_DIR / "models" / "yolo_best.pt").exists(),
            "efficientnet": (BASE_DIR / "models" / "efficientnet_best.keras").exists(),
        },
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh"}), 400

    file     = request.files["image"]
    import re
    safe_name = re.sub(r'[^\w.]', '_', file.filename)
    filename = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    # Đọc ảnh
    img_bgr = cv2.imread(str(save_path))
    if img_bgr is None:
        return jsonify({"error": "Không đọc được ảnh"}), 400

    start  = time.time()
    result = run_pipeline(img_bgr)
    elapsed = round(time.time() - start, 3)

    # Vẽ kết quả lên ảnh
    annotated = draw_quality_overlay(img_bgr, result["detections"])
    result_filename = f"result_{filename}"
    cv2.imwrite(str(RESULT_DIR / result_filename), annotated)

    # Thống kê nhanh
    grades = [d["grade"] for d in result["detections"]]
    summary = {
        "Xuất khẩu": grades.count("Xuất khẩu"),
        "Loại 1":    grades.count("Loại 1"),
        "Loại 2":    grades.count("Loại 2"),
        "Loại bỏ":  grades.count("Loại bỏ"),
    }

    response = {
        "id":            uuid.uuid4().hex[:12],
        "timestamp":     datetime.now().isoformat(),
        "processing_ms": int(elapsed * 1000),
        "image_url":     f"/api/result/{result_filename}",
        "total_fruits":  result["count"],
        "detections":    result["detections"],
        "summary":       summary,
    }

    # Lưu vào history
    history_store.insert(0, {
        **response,
        "thumbnail": result_filename,
    })
    if len(history_store) > 100:   # Giữ tối đa 100 bản ghi
        history_store.pop()

    return jsonify(response)


@app.route("/api/result/<filename>", methods=["GET"])
def get_result_image(filename):
    path = RESULT_DIR / filename
    if not path.exists():
        return jsonify({"error": "Không tìm thấy ảnh kết quả"}), 404
    return send_file(str(path), mimetype="image/jpeg")


@app.route("/api/history", methods=["GET"])
def get_history():
    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 20))
    start = (page - 1) * limit
    items = history_store[start: start + limit]
    return jsonify({
        "page":  page,
        "limit": limit,
        "total": len(history_store),
        "items": items,
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    if not history_store:
        return jsonify({"message": "Chưa có dữ liệu"})

    all_grades = []
    all_fruits = {}
    for record in history_store:
        for d in record.get("detections", []):
            all_grades.append(d["grade"])
            fruit = d.get("fruit_vi", d.get("fruit", "?"))
            all_fruits[fruit] = all_fruits.get(fruit, 0) + 1

    total = len(all_grades)
    return jsonify({
        "total_analyzed": total,
        "grade_distribution": {
            g: {"count": all_grades.count(g),
                "pct":   round(all_grades.count(g) / total * 100, 1)}
            for g in ["Xuất khẩu", "Loại 1", "Loại 2", "Loại bỏ"]
        },
        "fruit_distribution": dict(
            sorted(all_fruits.items(), key=lambda x: -x[1])
        ),
        "sessions": len(history_store),
    })


@app.route("/api/history/<record_id>", methods=["DELETE"])
def delete_history(record_id):
    global history_store
    history_store = [r for r in history_store if r.get("id") != record_id]
    return jsonify({"message": "Đã xóa"})


if __name__ == "__main__":
    print("🚀 Agri-AI Backend đang khởi động...")
    print(f"   Base dir: {BASE_DIR}")
    print("   API: http://localhost:5000/api/health")
    app.run(host="0.0.0.0", port=5000, debug=True)
