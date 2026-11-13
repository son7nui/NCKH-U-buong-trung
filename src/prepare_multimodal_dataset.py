import os
import shutil
import cv2
import numpy as np
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MMOTU_DIR = BASE_DIR / "MMOTU"
OTU_2D_DIR = MMOTU_DIR / "OTU_2d"
OTU_3D_DIR = MMOTU_DIR / "OTU_3d"

# Output Directories
OUTPUT_BINARY_DIR = BASE_DIR / "dataset_multimodal_binary"
OUTPUT_CROPS_DIR = BASE_DIR / "dataset_multimodal_crops_7cls"

# 7 Fine-grained Tumor Classes (Excluding Normal Ovary)
TUMOR_7_CLASSES = {
    0: "Chocolate_cyst",
    1: "Serous_cystadenoma",
    2: "Teratoma",
    3: "Theca_cell_tumor",
    4: "Simple_cyst",
    5: "Mucinous_cystadenoma",  # mapped from original class 6
    6: "High_grade_serous"      # mapped from original class 7
}

# Mapping from original MMOTU class ID to 7-class ID
ORIGINAL_TO_7CLS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    6: 5,
    7: 6
}

def load_cls_mapping(cls_file):
    """Loads <image_name>.JPG <class_id> mapping."""
    mapping = {}
    if not os.path.exists(cls_file):
        return mapping
    with open(cls_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = int(parts[1])
    return mapping

def extract_bboxes_from_mask(mask_path, img_w, img_h, min_area=30):
    """
    Extracts bounding boxes from binary mask.
    Returns:
      - yolo_boxes: normalized (xc, yc, w, h)
      - pixel_boxes: integer (x1, y1, x2, y2)
    """
    if not os.path.exists(mask_path):
        return [], []

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], []

    thresh = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_boxes = []
    pixel_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)

        # Clamping
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        pixel_boxes.append((x1, y1, x2, y2))

        # Normalized YOLO format
        xc = min(1.0, max(0.0, (x + w / 2.0) / img_w))
        yc = min(1.0, max(0.0, (y + h / 2.0) / img_h))
        norm_w = min(1.0, max(0.0, w / float(img_w)))
        norm_h = min(1.0, max(0.0, h / float(img_h)))
        yolo_boxes.append((xc, yc, norm_w, norm_h))

    return yolo_boxes, pixel_boxes

def crop_roi_with_padding(img, bbox, pad_ratio=0.10, target_size=(300, 300)):
    """Crops bounding box with padding margin and resizes to target size."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    crop_x1 = max(0, x1 - pad_w)
    crop_y1 = max(0, y1 - pad_h)
    crop_x2 = min(w, x2 + pad_w)
    crop_y2 = min(h, y2 + pad_h)

    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    return crop_resized

def process_modality(data_dir, prefix, split_name, binary_img_dir, binary_lbl_dir, crops_dir, stats):
    """Processes a single modality (OTU_2d or OTU_3d) for a given split (train or val)."""
    cls_file = data_dir / f"{split_name}_cls.txt"
    cls_mapping = load_cls_mapping(cls_file)

    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"

    print(f"  -> Xử lý {prefix.upper()} [{split_name.upper()}]: {len(cls_mapping)} ảnh...")

    for img_name, orig_cls_id in cls_mapping.items():
        base_id = Path(img_name).stem
        src_img_path = images_dir / img_name

        if not src_img_path.exists():
            continue

        # Target unique name with prefix
        unique_img_name = f"{prefix}_{base_id}.JPG"
        unique_lbl_name = f"{prefix}_{base_id}.txt"

        # Read image
        img = cv2.imread(str(src_img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Find mask
        mask_path = annotations_dir / f"{base_id}_binary.PNG"
        if not mask_path.exists():
            mask_path = annotations_dir / f"{base_id}.PNG"

        yolo_boxes, pixel_boxes = extract_bboxes_from_mask(mask_path, w, h)

        # 1. Prepare YOLO Binary Dataset
        dst_bin_img = binary_img_dir / unique_img_name
        shutil.copy2(src_img_path, dst_bin_img)

        bin_lbl_path = binary_lbl_dir / unique_lbl_name
        with open(bin_lbl_path, "w", encoding="utf-8") as lf:
            if orig_cls_id == 5:
                # Normal ovary: empty label
                stats["normal_images"] += 1
            else:
                for xc, yc, bw, bh in yolo_boxes:
                    lf.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    stats["tumor_boxes"] += 1
                stats["tumor_images"] += 1

        # 2. Prepare EfficientNet-B3 Cropped ROI Dataset (Only for Diseased Images)
        if orig_cls_id != 5 and orig_cls_id in ORIGINAL_TO_7CLS:
            target_7cls_id = ORIGINAL_TO_7CLS[orig_cls_id]
            cls_name = TUMOR_7_CLASSES[target_7cls_id]
            target_crop_folder = crops_dir / split_name / cls_name
            os.makedirs(target_crop_folder, exist_ok=True)

            # If bboxes detected, crop each box
            if pixel_boxes:
                for idx, pbox in enumerate(pixel_boxes):
                    roi = crop_roi_with_padding(img, pbox, pad_ratio=0.10, target_size=(300, 300))
                    if roi is not None:
                        crop_filename = f"{prefix}_{base_id}_crop{idx}.jpg"
                        cv2.imwrite(str(target_crop_folder / crop_filename), roi)
                        stats["crops_saved"] += 1
            else:
                # Fallback if mask was empty: resize whole image to 300x300
                whole_roi = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
                crop_filename = f"{prefix}_{base_id}_whole.jpg"
                cv2.imwrite(str(target_crop_folder / crop_filename), whole_roi)
                stats["crops_saved"] += 1

def main():
    print("=" * 65)
    print("  HỢP NHẤT ĐA PHƯƠNG THÁI MMOTU (2D + 3D CEUS) CHO HỆ THỐNG 2 GIAI ĐOẠN  ")
    print("=" * 65)

    for split in ["train", "val"]:
        print(f"\n[1] Chuẩn bị phân chia {split.upper()}...")
        bin_img_dir = OUTPUT_BINARY_DIR / "images" / split
        bin_lbl_dir = OUTPUT_BINARY_DIR / "labels" / split
        os.makedirs(bin_img_dir, exist_ok=True)
        os.makedirs(bin_lbl_dir, exist_ok=True)

        stats = {"tumor_images": 0, "normal_images": 0, "tumor_boxes": 0, "crops_saved": 0}

        # Process 2D
        process_modality(OTU_2D_DIR, "2d", split, bin_img_dir, bin_lbl_dir, OUTPUT_CROPS_DIR, stats)
        # Process 3D
        process_modality(OTU_3D_DIR, "3d", split, bin_img_dir, bin_lbl_dir, OUTPUT_CROPS_DIR, stats)

        print(f"Tổng kết {split.upper()}:")
        print(f"  • Ảnh có u: {stats['tumor_images']} (với {stats['tumor_boxes']} boxes)")
        print(f"  • Ảnh bình thường: {stats['normal_images']} (Negative)")
        print(f"  • Tổng ảnh nhị phân: {stats['tumor_images'] + stats['normal_images']}")
        print(f"  • Tổng ảnh crop ROI (7 nhóm u): {stats['crops_saved']}")

    # Create dataset.yaml for YOLO11 Binary
    yaml_content = f"""# MMOTU Multimodal Binary Ovarian Tumor Detection (2D + 3D)
path: {OUTPUT_BINARY_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: Tumor
"""
    with open(OUTPUT_BINARY_DIR / "dataset.yaml", "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print("\n" + "=" * 65)
    print("✅ ĐÃ HOÀN TẤT CHUẨN BỊ BỘ DỮ LIỆU ĐA PHƯƠNG THÁI 2D + 3D THÀNH CÔNG!")
    print(f"📁 Dataset YOLO Binary: {OUTPUT_BINARY_DIR}")
    print(f"📁 Dataset Crops 7 Lớp: {OUTPUT_CROPS_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    main()
