import os
import sys

# Ensure torch DLLs on Windows are discoverable
torch_lib = os.path.expanduser(r"~\AppData\Roaming\Python\Python314\site-packages\torch\lib")
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

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

CLASS_COLORS = {
    0: (255, 105, 180),  # HotPink
    1: (30, 144, 255),   # DodgerBlue
    2: (50, 205, 50),    # LimeGreen
    3: (255, 165, 0),    # Orange
    4: (147, 112, 219),  # MediumPurple
    5: (192, 192, 192),  # Gray
    6: (0, 206, 209),    # DarkTurquoise
    7: (220, 20, 60)     # Crimson (Malignant)
}

def parse_yolo_label(label_path, img_w, img_h):
    """Parses YOLO label txt file into list of (class_id, x1, y1, x2, y2)."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                c_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                x1 = int((xc - w / 2.0) * img_w)
                y1 = int((yc - h / 2.0) * img_h)
                x2 = int((xc + w / 2.0) * img_w)
                y2 = int((yc + h / 2.0) * img_h)
                boxes.append((c_id, x1, y1, x2, y2))
    return boxes

def visualize_comparison(model, val_img_dir, val_lbl_dir, output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    images = sorted(list(val_img_dir.glob("*.JPG")))[:num_samples]
    
    print(f"\nGenerating visual comparisons for {len(images)} samples...")
    for img_path in images:
        base_name = img_path.stem
        lbl_path = val_lbl_dir / f"{base_name}.txt"
        
        # Read image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        
        # Create copies for GT and Pred
        gt_vis = img_bgr.copy()
        pred_vis = img_bgr.copy()
        
        # Draw GT boxes (Green)
        gt_boxes = parse_yolo_label(lbl_path, w, h)
        for c_id, x1, y1, x2, y2 in gt_boxes:
            c_name = CLASSES.get(c_id, f"Class {c_id}")
            cv2.rectangle(gt_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(gt_vis, f"GT: {c_name}", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if not gt_boxes:
            cv2.putText(gt_vis, "GT: Normal Ovary (No tumor)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Run inference
        results = model.predict(source=str(img_path), conf=0.25, verbose=False)
        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                c_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                pred_boxes.append((c_id, conf, xyxy))
                
                c_name = CLASSES.get(c_id, f"Class {c_id}")
                color = (0, 0, 255) if c_id == 7 else (255, 0, 0) # Red for malignant
                cv2.rectangle(pred_vis, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(pred_vis, f"Pred: {c_name} {conf:.2f}", (xyxy[0], max(15, xyxy[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if not pred_boxes:
            cv2.putText(pred_vis, "Pred: No tumor detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
        # Combine side-by-side
        combined = cv2.hconcat([gt_vis, pred_vis])
        out_file = output_dir / f"compare_{base_name}.jpg"
        cv2.imwrite(str(out_file), combined)
        
    print(f"Saved visual comparisons to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and Visualize Ovarian Tumor YOLO Model")
    parser.add_argument("--weights", type=str, default="runs/detect/ovarian_tumor_yolo11n/weights/best.pt", help="Model weights path")
    parser.add_argument("--samples", type=int, default=12, help="Number of sample comparisons to save")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = base_dir / weights_path
        
    if not weights_path.exists():
        # Try finding last.pt or fall back
        alt_weights = base_dir / "runs" / "detect" / "ovarian_tumor_yolo11n" / "weights" / "last.pt"
        if alt_weights.exists():
            weights_path = alt_weights
        else:
            raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    print(f"Loading weights from: {weights_path}")
    model = YOLO(str(weights_path))
    
    val_img_dir = base_dir / "dataset_yolo" / "images" / "val"
    val_lbl_dir = base_dir / "dataset_yolo" / "labels" / "val"
    out_vis_dir = base_dir / "results" / "visualizations"
    
    visualize_comparison(model, val_img_dir, val_lbl_dir, out_vis_dir, num_samples=args.samples)

if __name__ == "__main__":
    main()
