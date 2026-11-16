import os
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MMOTU_DIR = BASE_DIR / "MMOTU"
OUTPUT_KFOLD_DIR = BASE_DIR / "dataset_kfold"

# 7 Fine-grained Tumor Classes (Excluding Normal Ovary)
TUMOR_7_CLASSES = {
    0: "Chocolate_cyst",
    1: "High_grade_serous",
    2: "Mucinous_cystadenoma",
    3: "Serous_cystadenoma",
    4: "Simple_cyst",
    5: "Teratoma",
    6: "Theca_cell_tumor"
}

# Mapping from original MMOTU class ID (0..7) to Alphabetical 7-Class ID (0..6)
# 0: Chocolate_cyst -> 0
# 1: Serous_cystadenoma -> 3
# 2: Teratoma -> 5
# 3: Theca_cell_tumor -> 6
# 4: Simple_cyst -> 4
# 5: Normal ovary -> Excluded
# 6: Mucinous_cystadenoma -> 2
# 7: High_grade_serous -> 1
ORIGINAL_TO_ALPHABETICAL_7CLS = {
    0: 0,
    1: 3,
    2: 5,
    3: 6,
    4: 4,
    6: 2,
    7: 1
}

def load_cls_file(cls_file):
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

def pure_python_stratified_kfold(samples, strata, n_splits=5, seed=42):
    """
    Pure Python implementation of Stratified K-Fold.
    Works anywhere without requiring third-party C-extensions.
    """
    random.seed(seed)
    # Group indices by stratum
    stratum_to_indices = defaultdict(list)
    for idx, st in enumerate(strata):
        stratum_to_indices[st].append(idx)
        
    folds = [[] for _ in range(n_splits)]
    
    for st, indices in stratum_to_indices.items():
        # Shuffle indices within this stratum
        shuffled = indices.copy()
        random.shuffle(shuffled)
        # Distribute round-robin into folds
        for i, idx in enumerate(shuffled):
            folds[i % n_splits].append(idx)
            
    # For each fold k, val is folds[k], train is union of other folds
    fold_splits = []
    for k in range(n_splits):
        val_idx = sorted(folds[k])
        train_idx = sorted([idx for j in range(n_splits) if j != k for idx in folds[j]])
        fold_splits.append((train_idx, val_idx))
        
    return fold_splits

def main():
    print("=" * 70)
    print("  TẠO BỘ DỮ LIỆU STRATIFIED 5-FOLD CROSS-VALIDATION (MMOTU 2D + 3D)  ")
    print("  Phân tầng Khóa gộp: Nhãn Bệnh Học (0..7) x Phương thái (2D/3D)   ")
    print("=" * 70)

    # 1. Thu thập toàn bộ 1.639 mẫu
    all_samples = []
    for mod_name, prefix in [("OTU_2d", "2d"), ("OTU_3d", "3d")]:
        mod_dir = MMOTU_DIR / mod_name
        for split_name in ["train", "val"]:
            cls_f = mod_dir / f"{split_name}_cls.txt"
            mapping = load_cls_file(cls_f)
            for img_name, orig_cls in mapping.items():
                base_id = Path(img_name).stem
                all_samples.append({
                    "unique_id": f"{prefix}_{base_id}",
                    "orig_img_name": img_name,
                    "modality": prefix,
                    "orig_class": orig_cls,
                    "is_tumor": (orig_cls != 5),
                    "stratum": f"{prefix}_cls{orig_cls}",
                    "rel_img_path": f"MMOTU/{mod_name}/images/{img_name}",
                    "rel_mask_path": f"MMOTU/{mod_name}/annotations/{base_id}_binary.PNG"
                })

    print(f"\n[+] Tổng số mẫu thu thập được: {len(all_samples)} ảnh (1.469 2D + 170 3D)")
    
    # Thống kê phân bố strata
    strata = [s["stratum"] for s in all_samples]
    strata_counts = Counter(strata)
    print("\n[+] Phân bố các nhóm kết hợp (Nhãn x Phương thái):")
    for st, cnt in sorted(strata_counts.items()):
        print(f"  • {st:12s}: {cnt:3d} mẫu")

    # 2. Phân tầng 5 Folds
    N_SPLITS = 5
    SEED = 42
    
    try:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_splits = list(skf.split(all_samples, strata))
        print("\n[+] Đã phân tầng bằng scikit-learn StratifiedKFold.")
    except ImportError:
        fold_splits = pure_python_stratified_kfold(all_samples, strata, n_splits=N_SPLITS, seed=SEED)
        print("\n[+] Đã phân tầng bằng Pure-Python Stratified K-Fold (Deterministic).")

    # 3. Xuất tệp cấu hình và Manifests
    manifest_dir = OUTPUT_KFOLD_DIR / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    meta_report = {
        "n_splits": N_SPLITS,
        "seed": SEED,
        "total_samples": len(all_samples),
        "classes_7": TUMOR_7_CLASSES,
        "original_to_7cls": ORIGINAL_TO_ALPHABETICAL_7CLS,
        "folds": {}
    }

    print("\n[+] Phân bố từng Fold kiểm định độc lập:")
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]

        # Kiểm tra tính toàn vẹn (Train giao Val = rỗng)
        assert len(set(train_idx).intersection(set(val_idx))) == 0, f"Lỗi rò rỉ tại fold {fold}!"

        # Đếm số ca ác tính High-grade serous và Normal trong Val fold
        val_hgs = sum(1 for s in val_samples if s["orig_class"] == 7)
        val_theca = sum(1 for s in val_samples if s["orig_class"] == 3)
        val_norm = sum(1 for s in val_samples if s["orig_class"] == 5)
        val_3d = sum(1 for s in val_samples if s["modality"] == "3d")

        print(f"  📁 Fold {fold}: Train {len(train_samples):4d} | Val {len(val_samples):3d} "
              f"(High-grade serous: {val_hgs:2d} | Theca: {val_theca:2d} | Normal: {val_norm:2d} | 3D: {val_3d:2d})")

        # Lưu manifest dạng JSON chi tiết cho từng fold
        fold_manifest = {
            "fold": fold,
            "train_unique_ids": [s["unique_id"] for s in train_samples],
            "val_unique_ids": [s["unique_id"] for s in val_samples],
            "train_samples": train_samples,
            "val_samples": val_samples
        }
        with open(manifest_dir / f"fold_{fold}_manifest.json", "w", encoding="utf-8") as f:
            json.dump(fold_manifest, f, ensure_ascii=False, indent=2)

        meta_report["folds"][f"fold_{fold}"] = {
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "val_high_grade_serous": val_hgs,
            "val_theca": val_theca,
            "val_normal": val_norm,
            "val_3d_ceus": val_3d
        }

    # Lưu metadata toàn cục
    with open(OUTPUT_KFOLD_DIR / "kfold_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta_report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("✅ ĐÃ TẠO THÀNH CÔNG BỘ MANIFEST 5-FOLD CHUẨN KHOA HỌC!")
    print(f"📁 Thư mục lưu trữ: {OUTPUT_KFOLD_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()

