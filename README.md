# Pancreas CT Medical Image Analysis

Automated pancreas segmentation on abdominal CT scans using **SAM-Med3D-turbo**, a promptable 3D medical image segmentation model. This project runs zero-shot inference on the [Pancreas-CT](https://www.cancerimagingarchive.net/collection/pancreas-ct/) dataset from The Cancer Imaging Archive (TCIA).

**Inference is based on [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D)** by Shanghai AI Lab.

This repo also includes an alternative pipeline for **VISTA3DCT**, a 3D CT segmentation model from MONAI’s VISTA project, with dedicated scripts and outputs (`run_inference_vista3dct.py`, `results_VISTA3DCT/`). See the official VISTA3DCT implementation here: https://github.com/Project-MONAI/VISTA/tree/main/vista3d

---

## Repository Overview

| Component | Description |
|-----------|-------------|
| `convert_dicom_to_nifti.py` | Converts DICOM CT series to NIfTI (.nii.gz) format |
| `run_inference_sammed3d.py` | Runs SAM-Med3D-turbo inference with interactive prompts from ground-truth |
| `run_inference_vista3dct.py` | Runs VISTA3DCT inference on CT volumes |
| `compute_dice.py` | Computes Dice scores between predictions and ground-truth labels |
| `extract_pred_slices.py` | Extracts sample slices with segmentation overlay for visualization |
| `run_pipeline.sh` | End-to-end pipeline: DICOM→NIfTI → inference → Dice evaluation |
| `utils/` | Inference utilities and metric computation |

**Expected directory structure after setup:**

```
Pancreas_Medical_Image_Analysis/
├── data/
│   └── PancreasCT/
│       ├── image/          # DICOM series (PANCREAS_0001/, PANCREAS_0002/, ...)
│       ├── nifti/          # Converted NIfTI images (generated)
│       └── label/          # Ground-truth NIfTI labels (label0001.nii.gz, ...)
├── checkpoints/
│   └── sam_med3d_turbo.pth # Model weights (downloaded)
├── results/
│   ├── predictions/       # Segmentation outputs (generated)
│   ├── pred_images/       # Visualization slices (generated)
│   ├── dice_scores.json
│   └── dice_summary.csv
└── ...
```

---

## Downloading Data and Checkpoints

### 1. Pancreas CT Images (TCIA)

The **Pancreas-CT** dataset is hosted by [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/pancreas-ct/).

- **Images (DICOM):** [Download (9.95 GB)](https://www.cancerimagingarchive.net/wp-content/uploads/Pancreas-CT-20200910.tcia)  
  - Requires [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images) to download DICOM series.
  - After download, organize each patient's DICOM series into:
    ```
    data/PancreasCT/image/PANCREAS_0001/<StudyDate-Series>/<SeriesUID>/*.dcm
    data/PancreasCT/image/PANCREAS_0002/...
    ...
    ```
  - Rename patient folders to `PANCREAS_0001`, `PANCREAS_0002`, etc., to match the label naming.

- **Manual Annotations (Labels):** [Download (948 KB)](https://www.cancerimagingarchive.net/wp-content/uploads/TCIA%5Fpancreas%5Flabels-02-05-2017-1.zip)  
  - Extract the ZIP. The NIfTI label files (e.g. `label0001.nii.gz`) should be placed in:
    ```
    data/PancreasCT/label/
    ```
  - Ensure label filenames match the image indices: `label0001.nii.gz` ↔ `PANCREAS_0001`, etc.

**Data citation (required):**  
Roth, H., Farag, A., Turkbey, E. B., Lu, L., Liu, J., & Summers, R. M. (2016). Data From Pancreas-CT (Version 2) [Data set]. The Cancer Imaging Archive. <https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU>

### 2. SAM-Med3D Checkpoint

Download the **SAM-Med3D-turbo** pre-trained weights:

- **Hugging Face:** [sam_med3d_turbo.pth](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth) (402 MB)
- **Google Drive / HF-Mirror:** See [SAM-Med3D repo](https://github.com/uni-medical/SAM-Med3D) for alternatives.

Place the file at:

```
checkpoints/sam_med3d_turbo.pth
```

---

## Environment Setup

### Conda

```bash
# Create environment
conda create -n sammed3d python=3.10 -y
conda activate sammed3d

# Install PyTorch (CUDA 12.4; adjust for your GPU)
pip install uv
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
               --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
uv pip install -r requirements.txt
```

For CPU-only or a different CUDA version, see [PyTorch install guide](https://pytorch.org/get-started/locally/).

---

## Running the Pipeline

Run the following in order:

### Option A: Full pipeline (recommended)

```bash
conda activate sammed3d
bash run_pipeline.sh
```

This runs:

1. DICOM → NIfTI conversion  
2. SAM-Med3D inference  
3. Dice score computation  

### Option B: Step-by-step

**1. Convert DICOM to NIfTI**

```bash
conda activate sammed3d
python convert_dicom_to_nifti.py
```

- **Input:** `data/PancreasCT/image/PANCREAS_*/.../*.dcm`  
- **Output:** `data/PancreasCT/nifti/PANCREAS_0001.nii.gz`, etc.

**2. Run inference**

```bash
python run_inference.py
```

- **Input:** NIfTI images + labels in `data/PancreasCT/nifti/` and `data/PancreasCT/label/`  
- **Output:** `results/predictions/PANCREAS_*.nii.gz`, `results/dice_scores.json`

**3. Compute Dice scores (optional, also done during inference)**

```bash
python compute_dice.py
```

- **Output:** `results/dice_summary.csv`

**4. Extract visualization slices (optional)**

```bash
python extract_pred_slices.py
```

- Extracts 10 random slices (with segmentation) per case from slices 100–200.  
- Overlays segmentation in red on the original CT.  
- **Output:** `results/pred_images/<case>_<slice>_pred.png`

---

## Summary

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `python convert_dicom_to_nifti.py` | DICOM → NIfTI |
| 2 | `python run_inference.py` | SAM-Med3D segmentation |
| 3 | `python compute_dice.py` | Dice evaluation |
| 4 | `python extract_pred_slices.py` | Visualization slices |

Or run steps 1–3 with: `bash run_pipeline.sh`
