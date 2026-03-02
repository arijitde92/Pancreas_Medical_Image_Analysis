#!/usr/bin/env python3
"""
Extract 10 random 2D slices (that contain segmentation) from each NIfTI prediction,
create side-by-side comparison (original CT | CT + segmentation overlay), and save as PNG.

- Slice range: 100 to 200 (or 100 to last slice if volume has < 200 slices).
- Only slices that contain at least some segmentation label are considered.
- Output: two images concatenated horizontally — left: original CT; right: CT + red overlay (50% opacity).

Reads:  results/predictions/*.nii, *.nii.gz
        data/PancreasCT/nifti/*.nii, *.nii.gz (corresponding originals)
Writes: results/pred_images/<input_file_name>_<slice_num>_pred.png
"""

import glob
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import cv2


def normalize_slice_to_uint8(slice_2d):
    """Normalize 2D slice to 0-255 for display."""
    s = np.asarray(slice_2d, dtype=np.float64)
    if s.size == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    return (s * 255).clip(0, 255).astype(np.uint8)


def overlay_red_mask(background_uint8, mask_slice, opacity=0.5):
    """
    Overlay binary mask in red on grayscale background.
    background_uint8: (H, W) uint8
    mask_slice: (H, W) same shape, non-zero where segmentation is present
    opacity: 0-1, fraction of red in the overlay (0.5 = 50% red, 50% original)
    Returns: (H, W, 3) BGR uint8 for saving with cv2.
    """
    rgb = cv2.cvtColor(background_uint8, cv2.COLOR_GRAY2BGR)
    mask = (np.asarray(mask_slice, dtype=np.float64) > 0)
    if not np.any(mask):
        return rgb
    red_bgr = np.array([0, 0, 255], dtype=np.float64)  # BGR
    for c in range(3):
        rgb[:, :, c] = np.where(
            mask,
            (1 - opacity) * rgb[:, :, c].astype(np.float64) + opacity * red_bgr[c],
            rgb[:, :, c].astype(np.float64),
        ).clip(0, 255).astype(np.uint8)
    return rgb


def resolve_original_path(base_dir, base_name):
    """Return path to original NIfTI for this prediction, or None if not found."""
    nifti_dir = base_dir / "data" / "PancreasCT" / "nifti"
    for ext in (".nii.gz", ".nii"):
        p = nifti_dir / f"{base_name}{ext}"
        if p.exists():
            return p
    return None


def main():
    base_dir = Path(__file__).resolve().parent
    pred_dir = base_dir / "results" / "predictions"
    nifti_dir = base_dir / "data" / "PancreasCT" / "nifti"
    out_dir = base_dir / "results" / "pred_images"

    pred_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        str(pred_dir / "*.nii.gz"),
        str(pred_dir / "*.nii"),
    ]
    pred_files = []
    for p in patterns:
        pred_files.extend(glob.glob(p))
    pred_files = sorted(set(pred_files))

    if not pred_files:
        print(f"No NIfTI files found in {pred_dir}")
        return

    num_slices = 10
    slice_min = 100
    slice_max_inclusive = 200  # use slices in [100, 200] inclusive
    random.seed(42)

    for pred_path in pred_files:
        path = Path(pred_path)
        if path.suffix == ".gz":
            base_name = path.stem.replace(".nii", "")
        else:
            base_name = path.stem

        try:
            pred_img = sitk.ReadImage(str(path))
            pred_data = sitk.GetArrayFromImage(pred_img)  # (Z, Y, X)
        except Exception as e:
            print(f"Skip {path.name}: {e}")
            continue

        if pred_data.ndim != 3:
            print(f"Skip {path.name}: expected 3D, got shape {pred_data.shape}")
            continue

        depth = pred_data.shape[0]
        # Slice range: 100 to min(200, depth-1). If depth <= 100, use 0 to last.
        start_z = slice_min if depth > slice_min else 0
        end_z = min(slice_max_inclusive, depth - 1)
        if start_z > end_z:
            print(f"Skip {path.name}: no slices in range [100, 200] (depth={depth})")
            continue

        # Indices that contain at least one segmentation pixel
        candidate_indices = []
        for z in range(start_z, end_z + 1):
            if np.any(pred_data[z, :, :] > 0):
                candidate_indices.append(z)

        if not candidate_indices:
            print(f"Skip {path.name}: no slices with segmentation in range [{start_z}, {end_z}]")
            continue

        if len(candidate_indices) <= num_slices:
            chosen_indices = candidate_indices
        else:
            chosen_indices = random.sample(candidate_indices, num_slices)

        # Load corresponding original image
        original_path = resolve_original_path(base_dir, base_name)
        if original_path is None:
            print(f"Skip {path.name}: original not found in {nifti_dir}")
            continue

        try:
            orig_img = sitk.ReadImage(str(original_path))
            orig_data = sitk.GetArrayFromImage(orig_img)  # (Z, Y, X)
        except Exception as e:
            print(f"Skip {path.name}: failed to load original: {e}")
            continue

        if orig_data.shape != pred_data.shape:
            print(f"Warning: {path.name} shape {pred_data.shape} != original {orig_data.shape}; resizing mask to original for overlay.")

        for slice_idx in chosen_indices:
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
            out_name = f"{base_name}_{slice_idx}_pred.png"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), combined)
            print(f"  {out_name} (slice {slice_idx}, has label)")

        print(f"Saved {len(chosen_indices)} slices for {path.name}")

    print(f"Done. Images saved to {out_dir}")


if __name__ == "__main__":
    main()
