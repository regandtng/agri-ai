import os
from pathlib import Path
from collections import defaultdict

OLD_TO_NEW = {
    14: 0,   # mango       → xoai
    7:  1,   # dragon fruit → thanh_long
    17: 3,   # orange      → cam
    26: 5,   # watermelon  → dua_hau
    3:  6,   # banana      → chuoi
    9:  7,   # guava       → oi
    11: 8,   # jackfruit   → mit
    8:  9,   # durian      → sau_rieng
    20: 10,  # rambutan    → chom_chom
    15: 11,  # mangosteen  → mang_cut
}

CLASS_VI = [
    "Xoài","Thanh long","Bưởi ⚠","Cam","Quýt ⚠",
    "Dưa hấu","Chuối","Ổi","Mít","Sầu riêng","Chôm chôm","Măng cụt"
]

BASE_DIR = Path(__file__).resolve().parent.parent / "dataset"
SPLITS   = ["train", "valid", "test"]

def main():
    print("="*55)
    print("  Remap Labels: 27 class → 12 class ĐBSCL")
    print("="*55)
    stats = defaultdict(int)

    for split in SPLITS:
        label_dir = BASE_DIR / split / "labels"
        if not label_dir.exists():
            print(f"  ⚠️  Không tìm thấy: {label_dir}")
            continue
        files = list(label_dir.glob("*.txt"))
        print(f"\n  [{split}] {len(files)} file...")
        for lf in files:
            lines = lf.read_text().strip().splitlines()
            new_lines = []
            for line in lines:
                if not line.strip(): continue
                parts = line.split()
                old_id = int(parts[0])
                if old_id in OLD_TO_NEW:
                    parts[0] = str(OLD_TO_NEW[old_id])
                    new_lines.append(" ".join(parts))
                    stats[OLD_TO_NEW[old_id]] += 1
            lf.write_text("\n".join(new_lines))
        print(f"    ✅ Xong!")

    print("\n" + "="*55)
    print("  📊 KẾT QUẢ:")
    for cid in range(12):
        count = stats.get(cid, 0)
        bar = "█" * min(count//50, 25)
        warn = " ← ⚠️ THIẾU" if count == 0 else ""
        print(f"  {cid:2d}. {CLASS_VI[cid]:15s} {count:5d}  {bar}{warn}")

    print("\n  ✅ Xong! Chạy tiếp: python detection/yolov8/train_yolo.py")

if __name__ == "__main__":
    main()