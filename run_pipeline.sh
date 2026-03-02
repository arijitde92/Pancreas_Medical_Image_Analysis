#!/usr/bin/env bash
# run_pipeline.sh
# ----------------
# End-to-end pipeline for SAM-Med3D Pancreas CT Segmentation.
# Activates the conda environment, converts DICOM to NIfTI,
# runs SAM-Med3D inference, and computes Dice scores.
#
# Usage:
#   bash run_pipeline.sh
#
# Prerequisites:
#   conda env 'sammed3d' must exist (see README.md for setup)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Color helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  SAM-Med3D │ Pancreas CT Segmentation Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Activate conda environment ────────────────────────────────────────────────
info "Activating conda environment 'sammed3d' …"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sammed3d || error "Could not activate conda env 'sammed3d'. See README for setup."

# ── Step 1: DICOM → NIfTI conversion ─────────────────────────────────────────
echo ""
info "Step 1/3 — Converting DICOM series to NIfTI …"
python "${SCRIPT_DIR}/convert_dicom_to_nifti.py" || error "DICOM conversion failed."

# Verify output
NIFTI_COUNT=$(ls "${SCRIPT_DIR}/data/PancreasCT/nifti/"*.nii.gz 2>/dev/null | wc -l)
info "  → ${NIFTI_COUNT} NIfTI file(s) created."

# ── Step 2: SAM-Med3D Inference ───────────────────────────────────────────────
echo ""
info "Step 2/3 — Running SAM-Med3D inference (this may take several minutes) …"
python "${SCRIPT_DIR}/run_inference.py" || error "Inference failed."

# ── Step 3: Dice Score Summary ────────────────────────────────────────────────
echo ""
info "Step 3/3 — Computing aggregate Dice scores …"
python "${SCRIPT_DIR}/compute_dice.py" || error "Dice computation failed."

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}Pipeline complete!${NC}"
echo ""
echo "  Predictions: ${SCRIPT_DIR}/results/predictions/"
echo "  Dice JSON  : ${SCRIPT_DIR}/results/dice_scores.json"
echo "  Dice CSV   : ${SCRIPT_DIR}/results/dice_summary.csv"
echo "═══════════════════════════════════════════════════════════════"
echo ""
