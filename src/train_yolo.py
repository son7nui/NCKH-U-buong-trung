import os
import sys

# Ensure torch DLLs on Windows are discoverable
torch_lib = os.path.expanduser(r"~\AppData\Roaming\Python\Python314\site-packages\torch\lib")
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO on MMOTU Ovarian Tumor Dataset")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model weights (e.g. yolo11n.pt, yolov8n.pt, yolo11s.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default 16 for 8GB GPU)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image input resolution")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader worker processes")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device (0 for GPU, or cpu)")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output directory")
    parser.add_argument("--name", type=str, default="ovarian_tumor_yolo11n", help="Experiment name")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    yaml_config = base_dir / "dataset_yolo" / "dataset.yaml"

    if not yaml_config.exists():
        raise FileNotFoundError(f"YAML config not found at: {yaml_config}. Run prepare_yolo_dataset.py first.")

    print("=" * 60)
    print("      HUẤN LUYỆN MÔ HÌNH PHÁT HIỆN KHỐI U BUỒNG TRỨNG (MMOTU)      ")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Allocated: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Target Device: {args.device}")
    print(f"Base Model: {args.model}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch} | Image Size: {args.imgsz}")
    print(f"Dataset Config: {yaml_config}")
    print("=" * 60)

    # Load pretrained YOLO model
    model = YOLO(args.model)

    # Train model
    results = model.train(
        data=str(yaml_config),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        patience=15,
        save=True,
        plots=True,
        verbose=True
    )

    print("\nTraining completed! Evaluating on validation set...")
    val_results = model.val(data=str(yaml_config), split="val")
    print(f"Validation mAP@50: {val_results.box.map50:.4f}")
    print(f"Validation mAP@50-95: {val_results.box.map:.4f}")

if __name__ == "__main__":
    main()
