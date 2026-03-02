#!/usr/bin/env python3
"""
compute_dice.py
----------------
Standalone script to (re-)compute Dice scores between saved predictions and
ground-truth labels without re-running inference.

Usage:
    conda activate sammed3d
    python compute_dice.py

Input:
    results/predictions/PANCREAS_000{1..10}.nii.gz
    data/PancreasCT/label/label000{1..10}.nii.gz

Output:
    results/dice_summary.csv   (per-case and aggregate Dice scores)
"""

import csv
import glob
import json
import os
import sys
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PRED_DIR   = os.path.join(BASE_DIR, "results", "predictions")
LABEL_DIR  = os.path.join(BASE_DIR, "data", "PancreasCT", "label")
CSV_OUT    = os.path.join(BASE_DIR, "results", "dice_summary.csv")

# Pancreas = class 1 in NIH Pancreas-CT labels
PANCREAS_CLASS = 1


def build_pairs():
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "PANCREAS_*.nii.gz")))
    if not pred_files:
        print(f"[ERROR] No prediction files found in: {PRED_DIR}")
        print("        Please run run_inference.py first.")
        sys.exit(1)

    pairs = []
    for pred_path in pred_files:
        basename = os.path.basename(pred_path)               # PANCREAS_0001.nii.gz
        idx = basename.replace("PANCREAS_", "").replace(".nii.gz", "")
        label_name = f"label{idx}.nii.gz"
        label_path = os.path.join(LABEL_DIR, label_name)
        if not os.path.exists(label_path):
            print(f"[WARNING] Label not found for {basename}")
            continue
        pairs.append((basename.replace(".nii.gz", ""), label_path, pred_path))
    return pairs


def main():
    from utils.metric_utils import compute_metrics

    print("=" * 60)
    print("  SAM-Med3D Pancreas Segmentation — Dice Evaluation")
    print("=" * 60)

    pairs = build_pairs()
    print(f"\nFound {len(pairs)} prediction-label pairs.\n")

    rows = []
    dice_scores = []

    for case_name, label_path, pred_path in pairs:
        try:
            metrics = compute_metrics(
                gt_path=label_path,
                pred_path=pred_path,
                metrics=["dice"],
                classes=[PANCREAS_CLASS],
            )
            dice = metrics.get(str(PANCREAS_CLASS), {}).get("dsc", float("nan"))
        except Exception as e:
            print(f"  [ERROR] {case_name}: {e}")
            dice = float("nan")

        dice_scores.append(dice)
        rows.append({"case": case_name, "dice": dice})
        sign = "✓" if not np.isnan(dice) else "✗"
        print(f"  [{sign}] {case_name:<20}  Dice = {dice:.4f}")

    # Aggregate statistics
    valid = [d for d in dice_scores if not np.isnan(d)]
    mean_dice = float(np.mean(valid)) if valid else float("nan")
    std_dice  = float(np.std(valid))  if valid else float("nan")
    min_dice  = float(np.min(valid))  if valid else float("nan")
    max_dice  = float(np.max(valid))  if valid else float("nan")

    print()
    print("=" * 60)
    print(f"  Cases evaluated  : {len(valid)}/{len(pairs)}")
    print(f"  Mean Dice        : {mean_dice:.4f}")
    print(f"  Std  Dice        : {std_dice:.4f}")
    print(f"  Min  Dice        : {min_dice:.4f}")
    print(f"  Max  Dice        : {max_dice:.4f}")
    print("=" * 60)

    # Save CSV
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "dice"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"case": "MEAN", "dice": round(mean_dice, 6)})
        writer.writerow({"case": "STD",  "dice": round(std_dice,  6)})
        writer.writerow({"case": "MIN",  "dice": round(min_dice,  6)})
        writer.writerow({"case": "MAX",  "dice": round(max_dice,  6)})

    print(f"\n  Summary saved → {CSV_OUT}")


if __name__ == "__main__":
    main()
