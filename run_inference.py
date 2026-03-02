#!/usr/bin/env python3
"""
run_inference.py
-----------------
Runs SAM-Med3D-turbo zero-shot inference on all 10 Pancreas CT NIfTI images.

Prerequisites:
  - conda activate sammed3d
  - python convert_dicom_to_nifti.py   (generates data/PancreasCT/nifti/)
  - checkpoints/sam_med3d_turbo.pth    (downloaded from HuggingFace)

Output:
  - results/predictions/PANCREAS_000{1..10}.nii.gz  (binary segmentation masks)
  - results/dice_scores.json
"""

import json
import os
import sys
import time
import glob
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
NIFTI_DIR   = os.path.join(BASE_DIR, "data", "PancreasCT", "nifti")
LABEL_DIR   = os.path.join(BASE_DIR, "data", "PancreasCT", "label")
PRED_DIR    = os.path.join(BASE_DIR, "results", "predictions")
CKPT_PATH   = os.path.join(BASE_DIR, "checkpoints", "sam_med3d_turbo.pth")
SCORES_JSON = os.path.join(BASE_DIR, "results", "dice_scores.json")

# Number of interactive click iterations per image (more = better dice, slower)
NUM_CLICKS  = 5

# Pancreas is class 1 in the NIH Pancreas-CT labels
PANCREAS_CLASS = 1

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)


def build_file_pairs():
    """
    Match NIfTI images to GT labels.
    Image:  PANCREAS_0001.nii.gz  →  Label: label0001.nii.gz
    """
    nifti_files = sorted(glob.glob(os.path.join(NIFTI_DIR, "PANCREAS_*.nii.gz")))
    if not nifti_files:
        print(f"[ERROR] No NIfTI images found in: {NIFTI_DIR}")
        print("        Please run convert_dicom_to_nifti.py first.")
        sys.exit(1)

    pairs = []
    for img_path in nifti_files:
        basename = os.path.basename(img_path)               # PANCREAS_0001.nii.gz
        idx = basename.replace("PANCREAS_", "").replace(".nii.gz", "")   # 0001
        label_name = f"label{idx}.nii.gz"
        label_path = os.path.join(LABEL_DIR, label_name)
        if not os.path.exists(label_path):
            print(f"[WARNING] Label not found for {basename}: {label_path}")
            continue
        pred_path = os.path.join(PRED_DIR, basename)
        pairs.append((img_path, label_path, pred_path))

    return pairs


def compute_dice_from_paths(gt_path: str, pred_path: str) -> float:
    """Quick numpy Dice for class=1 (pancreas)."""
    import nibabel as nib
    gt   = nib.load(gt_path).get_fdata().astype(np.uint8)
    pred = nib.load(pred_path).get_fdata().astype(np.uint8)
    gt_m = (gt == PANCREAS_CLASS)
    pd_m = (pred == PANCREAS_CLASS)
    denom = gt_m.sum() + pd_m.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * (gt_m & pd_m).sum() / denom)


def main():
    print("=" * 65)
    print("  SAM-Med3D Pancreas Segmentation — Inference")
    print("=" * 65)

    # ── Load model ────────────────────────────────────────────────────────────
    import medim
    import torch

    if not os.path.exists(CKPT_PATH):
        print(f"[INFO] Checkpoint not found locally. Downloading from HuggingFace …")
        print(f"       Target: {CKPT_PATH}")

    print(f"\n[1/3] Loading SAM-Med3D-turbo …  (checkpoint: {CKPT_PATH})")
    model = medim.create_model(
        "SAM-Med3D",
        pretrained=True,
        checkpoint_path=CKPT_PATH,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"      Model loaded on: {device}")

    # ── Build (image, label, prediction) file pairs ───────────────────────────
    pairs = build_file_pairs()
    print(f"\n[2/3] Found {len(pairs)} image-label pairs.")

    # ── Run inference ─────────────────────────────────────────────────────────
    from utils.infer_utils import validate_paired_img_gt
    from utils.metric_utils import compute_metrics, print_computed_metrics

    print(f"\n[3/3] Running inference  (num_clicks={NUM_CLICKS}) …\n")

    all_dice_scores = {}
    total_start = time.time()

    for i, (img_path, label_path, pred_path) in enumerate(pairs, 1):
        case_name = os.path.basename(img_path).replace(".nii.gz", "")
        print(f"  [{i:02d}/{len(pairs):02d}] {case_name}")
        t0 = time.time()

        try:
            validate_paired_img_gt(
                model=model,
                img_path=img_path,
                gt_path=label_path,
                output_path=pred_path,
                num_clicks=NUM_CLICKS,
            )
            elapsed = time.time() - t0

            # Compute Dice for this case
            metrics = compute_metrics(
                gt_path=label_path,
                pred_path=pred_path,
                metrics=["dice"],
                classes=[PANCREAS_CLASS],
            )
            dice = metrics.get(str(PANCREAS_CLASS), {}).get("dsc", float("nan"))
            all_dice_scores[case_name] = round(dice, 6)
            print(f"         Dice={dice:.4f}   ({elapsed:.1f}s)\n")

        except Exception as e:
            print(f"         [ERROR] {e}\n")
            all_dice_scores[case_name] = None

    total_elapsed = time.time() - total_start

    # ── Summary ───────────────────────────────────────────────────────────────
    valid_scores = [v for v in all_dice_scores.values() if v is not None and not np.isnan(v)]
    mean_dice = float(np.mean(valid_scores)) if valid_scores else float("nan")
    std_dice  = float(np.std(valid_scores))  if valid_scores else float("nan")

    print("=" * 65)
    print(f"  Inference complete  (total: {total_elapsed/60:.1f} min)")
    print(f"  Cases processed  : {len(valid_scores)}/{len(pairs)}")
    print(f"  Mean Dice (class 1): {mean_dice:.4f} ± {std_dice:.4f}")
    print("=" * 65)

    # Per-case table
    print(f"\n{'Case':<20}  {'Dice':>8}")
    print("-" * 32)
    for case_name, dice in sorted(all_dice_scores.items()):
        if dice is None or np.isnan(dice):
            print(f"  {case_name:<18}  {'  ERROR':>8}")
        else:
            print(f"  {case_name:<18}  {dice:>8.4f}")
    print("-" * 32)
    print(f"  {'Mean':<18}  {mean_dice:>8.4f}")
    print(f"  {'Std':<18}  {std_dice:>8.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    summary = {
        "per_case_dice": all_dice_scores,
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "num_clicks": NUM_CLICKS,
        "num_cases": len(pairs),
    }
    with open(SCORES_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Dice scores saved → {SCORES_JSON}")
    print(f"  Predictions saved → {PRED_DIR}/\n")


if __name__ == "__main__":
    main()
