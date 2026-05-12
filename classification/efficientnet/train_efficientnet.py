"""
=================================================================
  GIAI ĐOẠN 4 — Train EfficientNet-B0 phân loại 12 loại trái cây
=================================================================
Input : ảnh crop từ YOLO (hoặc thư mục dataset/train)
Output: tên loại + % confidence

Chạy:
    python classification/efficientnet/train_efficientnet.py
"""

import os, json
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ─── Cấu hình ───────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR  = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE    = 224          # EfficientNetB0 yêu cầu 224×224
BATCH_SIZE  = 32
EPOCHS      = 30
NUM_CLASSES = 12

CLASS_NAMES = [
    "xoai", "thanh_long", "buoi", "cam", "quit",
    "dua_hau", "chuoi", "oi", "mit", "sau_rieng",
    "chom_chom", "mang_cut",
]
CLASS_NAMES_VI = [
    "Xoài", "Thanh long", "Bưởi", "Cam", "Quýt",
    "Dưa hấu", "Chuối", "Ổi", "Mít", "Sầu riêng",
    "Chôm chôm", "Măng cụt",
]


# ─── 1. Data Generators (với Augmentation) ──────────────────
def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR / "train",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR / "valid",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen


# ─── 2. Xây dựng Model với Transfer Learning ────────────────
def build_model():
    # Load EfficientNetB0 pretrained trên ImageNet, bỏ top layer
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    # Giai đoạn 1: Freeze toàn bộ base – chỉ train top layers
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


# ─── 3. Train ───────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Train EfficientNet-B0 — Phân loại 12 loại trái cây")
    print("=" * 60)

    train_gen, val_gen = build_generators()
    model, base_model = build_model()

    # ── Phase 1: Train top layers ──
    print("\n[Phase 1] Train top layers (base frozen)...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cb_list = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        callbacks.ModelCheckpoint(
            str(MODELS_DIR / "efficientnet_phase1.keras"),
            save_best_only=True,
        ),
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=15, callbacks=cb_list)

    # ── Phase 2: Fine-tune – mở 30 lớp cuối của base ──
    print("\n[Phase 2] Fine-tuning (unfreeze top 30 layers)...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cb_list[2] = callbacks.ModelCheckpoint(
        str(MODELS_DIR / "efficientnet_best.keras"),
        save_best_only=True,
    )
    model.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS, callbacks=cb_list,
    )

    # Lưu class mapping
    mapping = {i: {"en": CLASS_NAMES[i], "vi": CLASS_NAMES_VI[i]} for i in range(NUM_CLASSES)}
    with open(MODELS_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Model lưu tại: {MODELS_DIR / 'efficientnet_best.keras'}")
    print(f"✅ Class mapping: {MODELS_DIR / 'class_mapping.json'}")


# ─── 4. Hàm predict dùng trong backend ──────────────────────
def load_classifier():
    """Load model và mapping, dùng trong Flask API."""
    model_path = MODELS_DIR / "efficientnet_best.keras"
    mapping_path = MODELS_DIR / "class_mapping.json"

    if not model_path.exists():
        raise FileNotFoundError("Model EfficientNet chưa được train!")

    model = tf.keras.models.load_model(str(model_path))
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    return model, mapping


def predict_crop(image_array: np.ndarray, model, mapping: dict) -> dict:
    """
    Phân loại 1 ảnh crop (numpy array BGR từ OpenCV).
    Trả về dict: class_id, class_name_vi, confidence.
    """
    import cv2
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id]) * 100

    return {
        "class_id":    class_id,
        "class_name":  mapping[str(class_id)]["en"],
        "class_name_vi": mapping[str(class_id)]["vi"],
        "confidence":  round(confidence, 2),
        "all_probs":   {mapping[str(i)]["vi"]: round(float(preds[i]) * 100, 2)
                        for i in range(NUM_CLASSES)},
    }


if __name__ == "__main__":
    train()
