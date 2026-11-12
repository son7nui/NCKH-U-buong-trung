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

OUTPUT_DIR = BASE_DIR / "dataset_yolo"
OUTPUT_IMAGES_TRAIN = OUTPUT_DIR / "images" / "train"
OUTPUT_IMAGES_VAL = OUTPUT_DIR / "images" / "val"
OUTPUT_LABELS_TRAIN = OUTPUT_DIR / "labels" / "train"
OUTPUT_LABELS_VAL = OUTPUT_DIR / "labels" / "val"

CLASSES = {
    0: "Chocolate cyst",
    1: "Serous cystadenoma",
    2: "Teratoma",
    3: "Theca cell tumor",
    4: "Simple cyst",
    5: "Normal ovary",
    6: "Mucinous cystadenoma",
    7: "High-grade serous"
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

    # Read mask as grayscale
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # Binary mask in MMOTU has pixel values 0 and 1 (or 255)
    thresh = (mask > 0).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Calculate normalized YOLO format
        x_center = (x + w / 2.0) / img_w
        y_center = (y + h / 2.0) / img_h
        norm_w = w / float(img_w)
        norm_h = h / float(img_h)
        
        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))
        
        bboxes.append((x_center, y_center, norm_w, norm_h))
        
    return bboxes

def process_split(cls_mapping, target_img_dir, target_lbl_dir, split_name="train"):
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_lbl_dir, exist_ok=True)
    
    stats = {cls_id: {"images": 0, "boxes": 0} for cls_id in CLASSES}
    total_images = len(cls_mapping)
    print(f"\nProcessing {split_name} split ({total_images} images)...")
    
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
        
        # Write YOLO label file
        label_file = target_lbl_dir / f"{base_id}.txt"
        with open(label_file, "w", encoding="utf-8") as lf:
            # If Normal Ovary (class 5), mask should have 0 boxes (background image in YOLO)
            if cls_id != 5:
                # If there are extracted bboxes, use them with cls_id
                for x_c, y_c, bw, bh in bboxes:
                    lf.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                    stats[cls_id]["boxes"] += 1
            # If class != 5 but bboxes is empty, mask might have small artifacts; we record it
        
        stats[cls_id]["images"] += 1

    print(f"\n{split_name.upper()} Statistics:")
    print(f"{'Class ID':<10} {'Class Name':<25} {'Images':<10} {'Boxes':<10}")
    print("-" * 55)
    for c_id, name in CLASSES.items():
        print(f"{c_id:<10} {name:<25} {stats[c_id]['images']:<10} {stats[c_id]['boxes']:<10}")
        
    return stats

def create_yaml_config():
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    content = f"""# MMOTU OTU_2d Ovarian Tumor Dataset configuration for YOLO
path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val

# Classes (8 fine-grained classes)
names:
  0: Chocolate cyst
  1: Serous cystadenoma
  2: Teratoma
  3: Theca cell tumor
  4: Simple cyst
  5: Normal ovary
  6: Mucinous cystadenoma
  7: High-grade serous
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nCreated YAML configuration at: {yaml_path}")

def main():
    train_cls = load_cls_file(MMOTU_DIR / "train_cls.txt")
    val_cls = load_cls_file(MMOTU_DIR / "val_cls.txt")
    
    print(f"Loaded {len(train_cls)} train images and {len(val_cls)} val images.")
    
    process_split(train_cls, OUTPUT_IMAGES_TRAIN, OUTPUT_LABELS_TRAIN, "train")
    process_split(val_cls, OUTPUT_IMAGES_VAL, OUTPUT_LABELS_VAL, "val")
    create_yaml_config()
    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    main()
