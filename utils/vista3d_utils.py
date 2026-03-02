#!/usr/bin/env python3
"""
Utility helpers for running VISTA3D MONAI bundle inference and post-processing
outputs into this repository's PancreasCT evaluation format.
"""

from __future__ import annotations

import glob
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import nibabel as nib
import numpy as np
import SimpleITK as sitk


def _is_nifti_path(path: Path) -> bool:
    name = path.name
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _collect_nifti_recursive(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and _is_nifti_path(p):
            out.append(p.resolve())
    return sorted(set(out))


def collect_candidate_output_files(vista_dir: Path) -> Set[Path]:
    """
    Collect likely prediction outputs produced by VISTA bundle inference.
    """
    search_roots = [
        vista_dir / "eval",
        vista_dir / "output",
        vista_dir / "outputs",
        vista_dir / "result",
        vista_dir / "results",
    ]
    files: Set[Path] = set()
    for root in search_roots:
        for f in _collect_nifti_recursive(root):
            files.add(f)

    # Fallback to broad search if expected output directories are empty.
    if not files:
        excluded = {".git", "pretrained_model", "configs", "docs"}
        for f in _collect_nifti_recursive(vista_dir):
            try:
                rel_parts = set(f.relative_to(vista_dir).parts)
            except ValueError:
                rel_parts = set()
            if rel_parts.intersection(excluded):
                continue
            files.add(f)
    return files


def run_bundle_inference_for_case(
    *,
    vista_dir: Path,
    image_path: Path,
    label_prompt: int,
    python_exe: str,
    timeout_sec: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """
    Run MONAI bundle inference for one image with a single class label prompt.
    """
    image_abs = str(image_path.resolve())
    input_dict = f"{{'image':'{image_abs}','label_prompt':[{int(label_prompt)}]}}"
    cmd = [
        python_exe,
        "-m",
        "monai.bundle",
        "run",
        "--config_file",
        "configs/inference.json",
        "--input_dict",
        input_dict,
    ]
    return subprocess.run(
        cmd,
        cwd=str(vista_dir),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


def find_bundle_output(
    *,
    vista_dir: Path,
    case_name: str,
    before_files: Set[Path],
    run_start_time: float,
) -> Path:
    """
    Find the prediction NIfTI produced by the most recent inference run.
    """
    after_files = collect_candidate_output_files(vista_dir)
    new_files = [p for p in after_files if p not in before_files and p.exists()]

    if new_files:
        case_matches = [p for p in new_files if case_name in p.name]
        if case_matches:
            return sorted(case_matches, key=lambda p: p.stat().st_mtime)[-1]
        return sorted(new_files, key=lambda p: p.stat().st_mtime)[-1]

    touched = [p for p in after_files if p.exists() and p.stat().st_mtime >= (run_start_time - 1.0)]
    if touched:
        case_matches = [p for p in touched if case_name in p.name]
        if case_matches:
            return sorted(case_matches, key=lambda p: p.stat().st_mtime)[-1]
        return sorted(touched, key=lambda p: p.stat().st_mtime)[-1]

    raise FileNotFoundError(
        "Could not locate bundle output NIfTI under VISTA directories. "
        "Expected output under 'eval/' or similar after bundle run."
    )


def save_binary_prediction(
    *,
    source_pred_path: Path,
    output_pred_path: Path,
    pancreas_label: int = 4,
) -> Tuple[Path, List[float]]:
    """
    Convert VISTA output to binary pancreas mask (0/1) and save as NIfTI.
    """
    pred_img = nib.load(str(source_pred_path))
    pred_data = np.asarray(pred_img.get_fdata())
    unique_vals = np.unique(pred_data).astype(np.float64).tolist()

    if np.any(pred_data == pancreas_label):
        mask = (pred_data == pancreas_label)
    else:
        mask = (pred_data > 0)

    out_data = mask.astype(np.uint8)
    out_header = pred_img.header.copy()
    out_header.set_data_dtype(np.uint8)

    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(out_data, affine=pred_img.affine, header=out_header)
    nib.save(out_img, str(output_pred_path))
    return output_pred_path, unique_vals


def normalize_slice_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    s = np.asarray(slice_2d, dtype=np.float64)
    if s.size == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    return (s * 255).clip(0, 255).astype(np.uint8)


def overlay_red_mask(background_uint8: np.ndarray, mask_slice: np.ndarray, opacity: float = 0.5) -> np.ndarray:
    rgb = cv2.cvtColor(background_uint8, cv2.COLOR_GRAY2BGR)
    mask = (np.asarray(mask_slice, dtype=np.float64) > 0)
    if not np.any(mask):
        return rgb
    red_bgr = np.array([0, 0, 255], dtype=np.float64)
    for c in range(3):
        rgb[:, :, c] = np.where(
            mask,
            (1 - opacity) * rgb[:, :, c].astype(np.float64) + opacity * red_bgr[c],
            rgb[:, :, c].astype(np.float64),
        ).clip(0, 255).astype(np.uint8)
    return rgb


def _resolve_original_path(nifti_dir: Path, base_name: str) -> Optional[Path]:
    for ext in (".nii.gz", ".nii"):
        p = nifti_dir / f"{base_name}{ext}"
        if p.exists():
            return p
    return None


def extract_overlay_slices(
    *,
    pred_dir: Path,
    nifti_dir: Path,
    out_dir: Path,
    num_slices: int = 10,
    slice_min: int = 100,
    slice_max_inclusive: int = 200,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Extract side-by-side CT and CT+mask overlay PNGs for each prediction volume.
    """
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(set(glob.glob(str(pred_dir / "*.nii.gz")) + glob.glob(str(pred_dir / "*.nii"))))
    if not pred_files:
        return {"cases_processed": 0, "images_written": 0}

    random.seed(seed)
    cases_processed = 0
    images_written = 0

    for pred_path in pred_files:
        path = Path(pred_path)
        base_name = path.stem.replace(".nii", "") if path.suffix == ".gz" else path.stem

        try:
            pred_img = sitk.ReadImage(str(path))
            pred_data = sitk.GetArrayFromImage(pred_img)
        except Exception:
            continue

        if pred_data.ndim != 3:
            continue

        depth = pred_data.shape[0]
        start_z = slice_min if depth > slice_min else 0
        end_z = min(slice_max_inclusive, depth - 1)
        if start_z > end_z:
            continue

        candidate_indices = [z for z in range(start_z, end_z + 1) if np.any(pred_data[z, :, :] > 0)]
        if not candidate_indices:
            continue

        chosen_indices = (
            candidate_indices if len(candidate_indices) <= num_slices else random.sample(candidate_indices, num_slices)
        )

        original_path = _resolve_original_path(nifti_dir, base_name)
        if original_path is None:
            continue

        try:
            orig_img = sitk.ReadImage(str(original_path))
            orig_data = sitk.GetArrayFromImage(orig_img)
        except Exception:
            continue

        for slice_idx in chosen_indices:
            if slice_idx >= orig_data.shape[0]:
                continue

            orig_slice = orig_data[slice_idx, :, :]
            mask_slice = pred_data[slice_idx, :, :]

            if orig_slice.shape != mask_slice.shape:
                mask_slice = cv2.resize(
                    mask_slice.astype(np.uint8),
                    (orig_slice.shape[1], orig_slice.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            orig_uint8 = normalize_slice_to_uint8(orig_slice)
            orig_bgr = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2BGR)
            overlay = overlay_red_mask(orig_uint8, mask_slice, opacity=0.5)
            combined = np.hstack([orig_bgr, overlay])

            out_path = out_dir / f"{base_name}_{slice_idx}_pred.png"
            cv2.imwrite(str(out_path), combined)
            images_written += 1

        cases_processed += 1

    return {"cases_processed": cases_processed, "images_written": images_written}


def _compute_binary_dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Optional[float]:
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    denom = gt_mask.sum() + pred_mask.sum()
    if denom == 0:
        return None
    return float((2.0 * intersection) / denom)


def _case_name_from_pred(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def compute_dice_json(
    *,
    pred_dir: Path,
    label_dir: Path,
    json_out: Path,
    gt_class: int = 1,
    pred_class: int = 1,
) -> Dict[str, object]:
    """
    Compute per-case and aggregate Dice scores and save JSON.
    """
    pred_files = sorted(set(glob.glob(str(pred_dir / "PANCREAS_*.nii.gz")) + glob.glob(str(pred_dir / "PANCREAS_*.nii"))))
    if not pred_files:
        raise FileNotFoundError(f"No predictions found in: {pred_dir}")

    per_case_dice: Dict[str, Optional[float]] = {}
    valid: List[float] = []

    for pred_path_str in pred_files:
        pred_path = Path(pred_path_str)
        case_name = _case_name_from_pred(pred_path)
        idx = case_name.replace("PANCREAS_", "")
        label_path = label_dir / f"label{idx}.nii.gz"
        if not label_path.exists():
            per_case_dice[case_name] = None
            continue

        gt = nib.load(str(label_path)).get_fdata()
        pred = nib.load(str(pred_path)).get_fdata()
        gt_mask = (gt == gt_class)
        pred_mask = (pred == pred_class)
        dice = _compute_binary_dice(gt_mask, pred_mask)
        per_case_dice[case_name] = round(dice, 6) if dice is not None else None
        if dice is not None and not np.isnan(dice):
            valid.append(float(dice))

    mean_dice = float(np.mean(valid)) if valid else float("nan")
    std_dice = float(np.std(valid)) if valid else float("nan")

    summary: Dict[str, object] = {
        "per_case_dice": per_case_dice,
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "num_cases": len(pred_files),
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_summary_txt(summary: Dict[str, object], summary_out: Path) -> None:
    per_case = summary.get("per_case_dice", {})
    mean_dice = summary.get("mean_dice")
    std_dice = summary.get("std_dice")

    lines = []
    lines.append(f"{'Case':<24}  {'Dice':>8}")
    lines.append("-" * 36)
    for case_name in sorted(per_case.keys()):
        dice = per_case[case_name]
        if dice is None:
            lines.append(f"  {case_name:<22}  {'ERROR':>8}")
        else:
            lines.append(f"  {case_name:<22}  {float(dice):>8.4f}")
    lines.append("-" * 36)
    if mean_dice is None or (isinstance(mean_dice, float) and np.isnan(mean_dice)):
        lines.append(f"  {'Mean':<22}  {'NaN':>8}")
    else:
        lines.append(f"  {'Mean':<22}  {float(mean_dice):>8.4f}")
    if std_dice is None or (isinstance(std_dice, float) and np.isnan(std_dice)):
        lines.append(f"  {'Std':<22}  {'NaN':>8}")
    else:
        lines.append(f"  {'Std':<22}  {float(std_dice):>8.4f}")

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

