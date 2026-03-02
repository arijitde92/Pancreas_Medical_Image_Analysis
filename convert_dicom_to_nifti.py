#!/usr/bin/env python3
"""
convert_dicom_to_nifti.py
--------------------------
Converts the 10 Pancreas CT DICOM series to NIfTI (.nii.gz) format.

Input structure:
    data/PancreasCT/image/PANCREAS_000{1..10}/<StudyDate-Series>/<SeriesUID>/1-*.dcm

Output:
    data/PancreasCT/nifti/PANCREAS_000{1..10}.nii.gz
"""

import os
import sys
import glob
import SimpleITK as sitk


DICOM_ROOT = os.path.join(os.path.dirname(__file__), "data", "PancreasCT", "image")
NIFTI_OUT  = os.path.join(os.path.dirname(__file__), "data", "PancreasCT", "nifti")


def find_dicom_series_dir(patient_dir: str) -> str:
    """
    Walk the patient directory tree and return the deepest folder
    that contains at least one .dcm file.
    """
    best_dir = None
    for root, dirs, files in os.walk(patient_dir):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if dcm_files:
            best_dir = root
    if best_dir is None:
        raise RuntimeError(f"No DICOM files found under: {patient_dir}")
    return best_dir


def convert_patient(patient_name: str, patient_dir: str, out_dir: str) -> str:
    """Convert a single DICOM series to NIfTI. Returns the output path."""
    dicom_series_dir = find_dicom_series_dir(patient_dir)
    print(f"  Reading DICOM from: {dicom_series_dir}")

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_dir)
    if not dicom_names:
        raise RuntimeError(f"SimpleITK found no DICOM series in: {dicom_series_dir}")

    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    out_path = os.path.join(out_dir, f"{patient_name}.nii.gz")
    sitk.WriteImage(image, out_path)
    print(f"  [✓] Saved → {out_path}  shape={image.GetSize()}  spacing={image.GetSpacing()}")
    return out_path


def main():
    os.makedirs(NIFTI_OUT, exist_ok=True)

    patient_dirs = sorted(
        [d for d in glob.glob(os.path.join(DICOM_ROOT, "PANCREAS_*")) if os.path.isdir(d)]
    )
    if not patient_dirs:
        print(f"[ERROR] No PANCREAS_* directories found in: {DICOM_ROOT}")
        sys.exit(1)

    print(f"Found {len(patient_dirs)} patient directories.\n")
    converted = []
    errors = []

    for patient_dir in patient_dirs:
        patient_name = os.path.basename(patient_dir)   # e.g. PANCREAS_0001
        print(f"[{patient_name}] Converting …")
        try:
            out_path = convert_patient(patient_name, patient_dir, NIFTI_OUT)
            converted.append(out_path)
        except Exception as e:
            print(f"  [ERROR] {e}")
            errors.append(patient_name)
        print()

    print("=" * 60)
    print(f"Conversion complete: {len(converted)} OK, {len(errors)} failed.")
    if errors:
        print(f"  Failed: {errors}")
    print(f"  NIfTI files saved to: {NIFTI_OUT}")


if __name__ == "__main__":
    main()
