#!/bin/bash

set -e

show_help() {
    echo "Usage: ./prepare_data.sh [dataset]"
    echo ""
    echo "Available options:"
    echo "  colon    - Prepare the Colon dataset (Runs Stage 1 & 2)"
    echo "  ebhi     - Prepare the EBHI-SEG dataset"
    echo "  glas     - Prepare the GlaS dataset"
    echo "  all      - Prepare all datasets sequentially"
    echo ""
}

# Check if an argument was provided
if [ -z "$1" ]; then
    show_help
    exit 1
fi

DATASET=$(echo "$1" | tr '[:upper:]' '[:lower:]')

echo "========================================"
echo "  SAGE-UNet Data Preparation Pipeline   "
echo "========================================"

if [ "$DATASET" == "colon" ] || [ "$DATASET" == "all" ]; then
    echo -e "\n[+] Processing Colon Dataset..."
    echo " -> Running Stage 1: Patch extraction and unsupervised filtering..."
    python prepare_data/prepare_colon_stage1.py
    echo " -> Running Stage 2: Train/Val/Test stratified splitting..."
    python prepare_data/prepare_colon_stage2.py
fi

if [ "$DATASET" == "ebhi" ] || [ "$DATASET" == "all" ]; then
    echo -e "\n[+] Processing EBHI Dataset..."
    echo " -> Running train/val/test split and renaming..."
    python prepare_data/prepare_ebhi.py
fi

if [ "$DATASET" == "glas" ] || [ "$DATASET" == "all" ]; then
    echo -e "\n[+] Processing GlaS Dataset..."
    echo " -> Running train/val/test split and renaming..."
    python prepare_data/prepare_glas.py
fi

# Catch invalid arguments
if [[ "$DATASET" != "colon" && "$DATASET" != "ebhi" && "$DATASET" != "glas" && "$DATASET" != "all" ]]; then
    echo "Error: Invalid dataset option '$1'"
    show_help
    exit 1
fi

echo -e "\n========================================"
echo " ✅ Preparation complete! "
echo "========================================"