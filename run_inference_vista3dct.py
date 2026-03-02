#!/usr/bin/env python3
"""
run_inference_vista3dct.py
--------------------------
Pancreas CT segmentation pipeline using VISTA3D-CT MONAI bundle inference.

Outputs are written in SAMMed3D-compatible structure:
  - results_VISTA3DCT/ct_predictions/*.nii.gz
  - results_VISTA3DCT/ct_pred_images/*_pred.png
  - results_VISTA3DCT/ct_dice_scores.json
  - results_VISTA3DCT/ct_summary.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from utils.vista3d_utils import (
    collect_candidate_output_files,
    compute_dice_json,
    extract_overlay_slices,
    find_bundle_output,
    run_bundle_inference_for_case,
    save_binary_prediction,
    write_summary_txt,
)


def _case_name_from_path(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def _collect_cases(nifti_dir: Path) -> List[Path]:
    files = sorted(list(nifti_dir.glob("PANCREAS_*.nii.gz")) + list(nifti_dir.glob("PANCREAS_*.nii")))
    dedup: Dict[str, Path] = {}
    for f in files:
        dedup[_case_name_from_path(f)] = f.resolve()
    return [dedup[k] for k in sorted(dedup.keys())]


def _setup_error_message(base_dir: Path, vista_dir: Path) -> str:
    return "\n".join(
        [
            "[SETUP] Missing required VISTA/NV-Segment assets.",
            "Expected:",
            f"  - VISTA repo: {base_dir / 'external' / 'VISTA'}",
            f"  - Bundle dir with configs/inference.json: {vista_dir}",
            f"  - NV-Segment-CTMR repo: {base_dir / 'external' / 'NV-Segment-CTMR'}",
            "  - NV-Segment-CT model checkpoint: external/NV-Segment-CTMR/NV-Segment-CT/models/model.pt",
            "",
            "Suggested setup commands:",
            "  git clone https://github.com/Project-MONAI/VISTA.git external/VISTA",
            "  git clone https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git external/NV-Segment-CTMR",
            "  mkdir -p external/NV-Segment-CTMR/NV-Segment-CT/models",
            "  wget -O external/NV-Segment-CTMR/NV-Segment-CT/models/model.pt "
            "https://huggingface.co/nvidia/NV-Segment-CT/resolve/main/vista3d_pretrained_model/model.pt",
            "  # optional legacy layout for this script's --vista-dir default",
            "  mkdir -p external/VISTA/vista3d/pretrained_model",
            "  cp -r external/NV-Segment-CTMR/NV-Segment-CT external/VISTA/vista3d/pretrained_model/",
        ]
    )


def preflight_checks(args: argparse.Namespace, base_dir: Path) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not args.nifti_dir.exists():
        errors.append(f"NIfTI directory not found: {args.nifti_dir}")
    elif not _collect_cases(args.nifti_dir):
        errors.append(f"No PANCREAS_*.nii or *.nii.gz files found in: {args.nifti_dir}")

    if not args.vista_dir.exists():
        errors.append(f"VISTA directory not found: {args.vista_dir}")
    else:
        config_file = args.vista_dir / "configs" / "inference.json"
        if not config_file.exists():
            fallback_bundle = base_dir / "external" / "NV-Segment-CTMR" / "NV-Segment-CT"
            fallback_hint = ""
            if (fallback_bundle / "configs" / "inference.json").exists():
                fallback_hint = (
                    "\nDetected NV-Segment bundle at: "
                    f"{fallback_bundle}\nTry: --vista-dir {fallback_bundle}"
                )
            errors.append(
                f"Required config missing: {config_file}\n"
                "Provide a MONAI bundle directory (e.g. NV-Segment-CT) that contains configs/inference.json."
                f"{fallback_hint}"
            )

        model_ckpt = args.vista_dir / "models" / "model.pt"
        legacy_model_ckpt = args.vista_dir / "pretrained_model" / "NV-Segment-CT" / "models" / "model.pt"
        if not model_ckpt.exists() and not legacy_model_ckpt.exists():
            errors.append(
                "Required model checkpoint not found. Expected one of:\n"
                f"  - {model_ckpt}\n"
                f"  - {legacy_model_ckpt}"
            )

    external_nv = base_dir / "external" / "NV-Segment-CTMR"
    if not external_nv.exists():
        warnings.append(f"Optional source repo not found (used for setup/reference): {external_nv}")

    if not args.skip_dice and not args.label_dir.exists():
        errors.append(f"Label directory not found for Dice computation: {args.label_dir}")

    if not torch.cuda.is_available():
        warnings.append("CUDA is not available in current runtime; VISTA3D inference may be very slow on CPU.")

    return errors, warnings


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run VISTA3D-CT pancreas segmentation for PancreasCT NIfTI volumes.")
    parser.add_argument(
        "--nifti-dir",
        type=Path,
        default=base_dir / "data" / "PancreasCT" / "nifti",
        help="Directory containing PANCREAS_*.nii.gz images.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=base_dir / "data" / "PancreasCT" / "label",
        help="Directory containing labelXXXX.nii.gz files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=base_dir / "results_VISTA3DCT",
        help="Root output directory for VISTA3D results.",
    )
    parser.add_argument(
        "--vista-dir",
        type=Path,
        default=base_dir / "external" / "VISTA" / "vista3d",
        help="Path to bundle root containing configs/inference.json (e.g. external/NV-Segment-CTMR/NV-Segment-CT).",
    )
    parser.add_argument(
        "--label-prompt",
        type=int,
        default=4,
        help="Class prompt ID passed to MONAI bundle (pancreas=4 for NV-Segment-CT).",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Run only first N cases.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions in ct_predictions.")
    parser.add_argument("--skip-slices", action="store_true", help="Skip PNG overlay extraction stage.")
    parser.add_argument("--skip-dice", action="store_true", help="Skip Dice computation stage.")
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used to run 'python -m monai.bundle run'.",
    )
    parser.add_argument(
        "--bundle-timeout-sec",
        type=int,
        default=None,
        help="Optional timeout (seconds) for each MONAI bundle run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    print("=" * 72)
    print(" VISTA3D-CT Pancreas Segmentation Pipeline")
    print("=" * 72)
    print(f"nifti_dir   : {args.nifti_dir}")
    print(f"label_dir   : {args.label_dir}")
    print(f"results_dir : {args.results_dir}")
    print(f"vista_dir   : {args.vista_dir}")
    print(f"label_prompt: {args.label_prompt}")
    print(f"python_exe  : {args.python_exe}")

    errors, warnings = preflight_checks(args, base_dir)
    for w in warnings:
        print(f"[WARNING] {w}")
    if errors:
        print("")
        for e in errors:
            print(f"[ERROR] {e}")
        print("")
        print(_setup_error_message(base_dir, args.vista_dir))
        return 1

    pred_dir = args.results_dir / "ct_predictions"
    pred_img_dir = args.results_dir / "ct_pred_images"
    dice_json = args.results_dir / "ct_dice_scores.json"
    summary_txt = args.results_dir / "ct_summary.txt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    case_paths = _collect_cases(args.nifti_dir)
    if args.max_cases is not None:
        case_paths = case_paths[: args.max_cases]
    print(f"\nFound {len(case_paths)} case(s) to process.\n")

    success = 0
    skipped = 0
    failed: List[str] = []
    total_start = time.time()

    for i, image_path in enumerate(case_paths, start=1):
        case_name = _case_name_from_path(image_path)
        out_pred = pred_dir / f"{case_name}.nii.gz"
        print(f"[{i:02d}/{len(case_paths):02d}] {case_name}")

        if out_pred.exists() and not args.overwrite:
            skipped += 1
            print(f"  - skip: prediction already exists ({out_pred.name})")
            continue

        before_files = collect_candidate_output_files(args.vista_dir)
        case_start = time.time()
        proc = run_bundle_inference_for_case(
            vista_dir=args.vista_dir,
            image_path=image_path,
            label_prompt=args.label_prompt,
            python_exe=args.python_exe,
            timeout_sec=args.bundle_timeout_sec,
        )
        if proc.returncode != 0:
            failed.append(case_name)
            print("  - failed: MONAI bundle command returned non-zero exit code")
            if proc.stderr:
                print("  - stderr (tail):")
                for line in proc.stderr.strip().splitlines()[-10:]:
                    print(f"    {line}")
            continue

        try:
            raw_pred = find_bundle_output(
                vista_dir=args.vista_dir,
                case_name=case_name,
                before_files=before_files,
                run_start_time=case_start,
            )
            _, unique_vals = save_binary_prediction(
                source_pred_path=raw_pred,
                output_pred_path=out_pred,
                pancreas_label=args.label_prompt,
            )
        except Exception as exc:
            failed.append(case_name)
            print(f"  - failed: {exc}")
            continue

        elapsed = time.time() - case_start
        success += 1
        print(f"  - raw output : {raw_pred}")
        print(f"  - saved mask : {out_pred}")
        print(f"  - raw unique : {unique_vals[:10]}")
        print(f"  - elapsed    : {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 72)
    print("Inference stage complete")
    print(f"  success: {success}")
    print(f"  skipped: {skipped}")
    print(f"  failed : {len(failed)}")
    print(f"  time   : {total_elapsed/60.0:.1f} min")
    if failed:
        print(f"  failed cases: {', '.join(failed)}")
    print("=" * 72)

    if not args.skip_slices:
        stats = extract_overlay_slices(
            pred_dir=pred_dir,
            nifti_dir=args.nifti_dir,
            out_dir=pred_img_dir,
        )
        print(
            f"[Slices] processed={stats['cases_processed']} "
            f"written={stats['images_written']} -> {pred_img_dir}"
        )

    if not args.skip_dice:
        summary = compute_dice_json(
            pred_dir=pred_dir,
            label_dir=args.label_dir,
            json_out=dice_json,
            gt_class=1,
            pred_class=1,
        )
        write_summary_txt(summary, summary_txt)
        print(f"[Dice] JSON saved -> {dice_json}")
        print(f"[Dice] TXT  saved -> {summary_txt}")

    if failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
