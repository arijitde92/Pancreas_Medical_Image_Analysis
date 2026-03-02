# -*- encoding: utf-8 -*-
# Taken directly from https://github.com/uni-medical/SAM-Med3D/blob/main/utils/metric_utils.py
# with minor adaptations for local use.

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import nibabel as nib
import numpy as np
from surface_distance import (compute_surface_dice_at_tolerance,
                              compute_surface_distances)


def compute_dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.sum(mask1 * mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    if sum_masks == 0:
        return np.nan
    return (2.0 * intersection) / sum_masks


def compute_metrics(gt_path: str,
                    pred_path: str,
                    classes: Optional[List[int]] = None,
                    metrics: Union[str, List[str]] = 'all') -> Dict[str, Dict[str, float]]:
    """
    Computes evaluation metrics between a ground truth and a prediction NIfTI file.

    Args:
        gt_path:   Path to the ground truth NIfTI file.
        pred_path: Path to the prediction NIfTI file.
        classes:   Optional list of class indices to compute metrics for.
                   If None, computed for all unique non-zero classes in GT.
        metrics:   'all' or list of ['dsc', 'nsd'].

    Returns:
        Dict mapping class label (str) → metric name → score.
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    results: Dict[str, Dict[str, float]] = defaultdict(dict)
    metrics = metrics.lower() if isinstance(metrics, str) else ['dsc' if m == 'dice' else m for m in metrics]
    available_metrics = {'dsc', 'nsd'}

    if isinstance(metrics, str) and metrics.lower() == 'all':
        metrics_to_compute = list(available_metrics)
    elif isinstance(metrics, list):
        metrics_to_compute = []
        for m in metrics:
            m_lower = m.lower()
            if m_lower not in available_metrics:
                raise ValueError(f"Unknown metric: {m}. Available: {available_metrics}")
            metrics_to_compute.append(m_lower)
        if not metrics_to_compute:
            return {}
    else:
        raise ValueError("Invalid 'metrics' argument. Must be 'all' or a list.")

    try:
        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)

        gt_data = gt_nii.get_fdata().astype(np.uint8)
        pred_data = pred_nii.get_fdata().astype(np.uint8)

        case_spacing = gt_nii.header.get_zooms()[:3]

        if classes is None:
            determined_classes = sorted(np.unique(gt_data).tolist())
            if 0 in determined_classes:
                determined_classes.remove(0)
            if not determined_classes:
                print(f"Warning: No non-zero classes found in GT: {gt_path}")
                return dict(results)
        else:
            determined_classes = sorted(list(set(classes)))

        if not determined_classes:
            return dict(results)

        for i in determined_classes:
            class_label = str(i)
            organ_i_gt = (gt_data == i)
            organ_i_pred = (pred_data == i)

            gt_empty = np.sum(organ_i_gt) == 0
            pred_empty = np.sum(organ_i_pred) == 0

            if gt_empty and pred_empty:
                if 'dsc' in metrics_to_compute:
                    results[class_label]['dsc'] = np.nan
                if 'nsd' in metrics_to_compute:
                    results[class_label]['nsd'] = np.nan
                continue
            elif gt_empty and not pred_empty:
                if 'dsc' in metrics_to_compute:
                    results[class_label]['dsc'] = 0.0
                if 'nsd' in metrics_to_compute:
                    results[class_label]['nsd'] = 0.0
                continue

            try:
                if 'dsc' in metrics_to_compute:
                    dsc_i = compute_dice_coefficient(organ_i_gt, organ_i_pred)
                    results[class_label]['dsc'] = float(dsc_i)

                if 'nsd' in metrics_to_compute:
                    if gt_empty:
                        nsd_i = 0.0 if not pred_empty else np.nan
                    else:
                        surface_distances = compute_surface_distances(
                            organ_i_gt, organ_i_pred, case_spacing)
                        nsd_i = compute_surface_dice_at_tolerance(
                            surface_distances, 1.0)
                    results[class_label]['nsd'] = float(nsd_i)

            except Exception as metric_error:
                print(f"Warning: Error computing metrics for class {i}: {metric_error}")
                if 'dsc' in metrics_to_compute and 'dsc' not in results[class_label]:
                    results[class_label]['dsc'] = np.nan
                if 'nsd' in metrics_to_compute and 'nsd' not in results[class_label]:
                    results[class_label]['nsd'] = np.nan

    except Exception as e:
        print(f"Error processing files {gt_path} and {pred_path}: {str(e)}")
        return dict(results) if results else {}

    return dict(results)


def print_computed_metrics(results_data: Dict[Any, Any], avg_only=False, title: str = "Computed Metrics"):
    """Prints computed metrics in a structured format."""
    print(f"\n--- {title} ---")
    if not results_data:
        print("No results to display.")
        return

    is_multi_group_type = False
    is_single_set_type = False
    processed_class_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)

    for _first_level_key, first_level_value in results_data.items():
        if isinstance(first_level_value, dict) and first_level_value:
            try:
                _second_level_key = next(iter(first_level_value))
                second_level_value = first_level_value[_second_level_key]

                if isinstance(second_level_value, dict):
                    is_multi_group_type = True
                    break
                elif isinstance(second_level_value, (float, int, np.number)):
                    is_single_set_type = True
                    break
            except StopIteration:
                continue

    if is_single_set_type:
        processed_class_metrics = {str(k): v for k, v in results_data.items()}
    elif is_multi_group_type:
        for group_id, class_metrics_dict in results_data.items():
            if isinstance(class_metrics_dict, dict):
                for class_label, metric_dict in class_metrics_dict.items():
                    if isinstance(metric_dict, dict):
                        for metric_name, value in metric_dict.items():
                            if metric_name not in processed_class_metrics[str(class_label)]:
                                processed_class_metrics[str(class_label)][metric_name] = []
                            if isinstance(processed_class_metrics[str(class_label)][metric_name], list):
                                processed_class_metrics[str(class_label)][metric_name].append(value)
        # Average over groups
        averaged = {}
        for cl, md in processed_class_metrics.items():
            averaged[cl] = {}
            for mn, vals in md.items():
                if isinstance(vals, list):
                    clean = [v for v in vals if not np.isnan(v)]
                    averaged[cl][mn] = np.mean(clean) if clean else np.nan
                else:
                    averaged[cl][mn] = vals
        processed_class_metrics = averaged
    else:
        print("Could not determine results structure.")
        print(results_data)
        return

    if not avg_only:
        print(f"{'Class':<10}", end="")
        all_metric_names = sorted({mn for md in processed_class_metrics.values() for mn in md})
        for mn in all_metric_names:
            print(f"  {mn.upper():>10}", end="")
        print()
        print("-" * (10 + 12 * len(all_metric_names)))

        for class_label in sorted(processed_class_metrics.keys(), key=lambda x: int(x) if x.isdigit() else x):
            print(f"{class_label:<10}", end="")
            for mn in all_metric_names:
                val = processed_class_metrics[class_label].get(mn, np.nan)
                if np.isnan(val):
                    print(f"  {'NaN':>10}", end="")
                else:
                    print(f"  {val:>10.4f}", end="")
            print()

    # Print averages
    print("\n" + "=" * (10 + 12 * len(all_metric_names)))
    print(f"{'Average':<10}", end="")
    for mn in all_metric_names:
        vals = [processed_class_metrics[cl].get(mn, np.nan) for cl in processed_class_metrics]
        clean = [v for v in vals if not np.isnan(v)]
        avg = np.mean(clean) if clean else np.nan
        if np.isnan(avg):
            print(f"  {'NaN':>10}", end="")
        else:
            print(f"  {avg:>10.4f}", end="")
    print()
