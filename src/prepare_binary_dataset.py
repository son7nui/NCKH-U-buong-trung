import os
import shutil
import cv2
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MMOTU_DIR = BASE_DIR / "MMOTU" / "OTU_2d"
IMAGES_DIR = MMOTU_DIR / "images"
ANNOTATIONS_DIR = MMOTU_DIR / "annotations"

OUTPUT_DIR = BASE_DIR / "dataset_binary"
OUTPUT_IMAGES_TRAIN = OUTPUT_DIR / "images" / "train"
OUTPUT_IMAGES_VAL = OUTPUT_DIR / "images" / "val"
OUTPUT_LABELS_TRAIN = OUTPUT_DIR / "labels" / "train"
OUTPUT_LABELS_VAL = OUTPUT_DIR / "labels" / "val"

# 1 Single class for binary detection
CLASSES = {
    0: "Tumor"
}

def load_cls_file(filepath):
    """Loads <image_name>.JPG <class_id> mapping."""
    mapping = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_name = parts[0]
                cls_id = int(parts[1])
                mapping[img_name] = cls_id
    return mapping

def extract_bboxes_from_mask(mask_path, img_w, img_h, min_area=30):
    """
    Extracts bounding boxes from binary mask.
    Returns list of normalized (x_center, y_center, w, h).
    """
    if not os.path.exists(mask_path):
        return []

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    thresh = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        
        x_center = (x + w / 2.0) / img_w
        y_center = (y + h / 2.0) / img_h
        norm_w = w / float(img_w)
        norm_h = h / float(img_h)
        
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))
        
        bboxes.append((x_center, y_center, norm_w, norm_h))
        
    return bboxes

def process_binary_split(cls_mapping, target_img_dir, target_lbl_dir, split_name="train"):
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_lbl_dir, exist_ok=True)
    
    stats = {"Normal": 0, "Tumor_images": 0, "Tumor_boxes": 0}
    total_images = len(cls_mapping)
    print(f"\nProcessing {split_name.upper()} split ({total_images} images)...")
    
    for img_name, cls_id in cls_mapping.items():
        base_id = Path(img_name).stem
        src_img_path = IMAGES_DIR / img_name
        
        if not src_img_path.exists():
            print(f"[Warning] Image missing: {src_img_path}")
            continue
            
        # Copy image to target
        dst_img_path = target_img_dir / img_name
        shutil.copy2(src_img_path, dst_img_path)
        
        # Read image shape
        img = cv2.imread(str(src_img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Try binary mask first
        mask_path = ANNOTATIONS_DIR / f"{base_id}_binary.PNG"
        if not mask_path.exists():
            mask_path = ANNOTATIONS_DIR / f"{base_id}.PNG"
            
        bboxes = extract_bboxes_from_mask(mask_path, w, h)
        label_file = target_lbl_dir / f"{base_id}.txt"
        
        with open(label_file, "w", encoding="utf-8") as lf:
            # Class 5 is Normal ovary -> 0 boxes (negative control sample)
            if cls_id == 5:
                stats["Normal"] += 1
            else:
                # All other classes (0, 1, 2, 3, 4, 6, 7) are mapped to 0 (Tumor)
                for x_c, y_c, bw, bh in bboxes:
                    lf.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                    stats["Tumor_boxes"] += 1
                stats["Tumor_images"] += 1

    print(f"Results for {split_name.upper()}:")
    print(f" - Bệnh (Tumor Images): {stats['Tumor_images']} ảnh (với {stats['Tumor_boxes']} bounding boxes)")
    print(f" - Bình thường (Normal Images): {stats['Normal']} ảnh (negative samples)")
    return stats

def create_binary_yaml_config(output_dir=OUTPUT_DIR):
    yaml_path = output_dir / "dataset.yaml"
    content = f"""# MMOTU Binary Ovarian Tumor Detection Dataset configuration for YOLO
path: {output_dir.as_posix()}
train: images/train
val: images/val

# Single Class: Tumor
names:
  0: Tumor
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nCreated Binary YAML configuration at: {yaml_path}")

def main():
    train_cls = load_cls_file(MMOTU_DIR / "train_cls.txt")
    val_cls = load_cls_file(MMOTU_DIR / "val_cls.txt")
    
    print(f"Loaded {len(train_cls)} train images and {len(val_cls)} val images.")
    process_binary_split(train_cls, OUTPUT_IMAGES_TRAIN, OUTPUT_LABELS_TRAIN, "train")
    process_binary_split(val_cls, OUTPUT_IMAGES_VAL, OUTPUT_LABELS_VAL, "val")
    create_binary_yaml_config()
    print("\n[SUCCESS] Binary YOLO dataset preparation completed successfully!")

if __name__ == "__main__":
    main()

