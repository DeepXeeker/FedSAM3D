#!/usr/bin/env bash
set -euo pipefail

echo "Preprocessing is dataset-specific and depends on how you downloaded/extracted data."
echo "Expected output:"
echo "  datasets/<name>/preprocessed/images/*.nii.gz"
echo "  datasets/<name>/preprocessed/labels/*.nii.gz"
echo "  datasets/<name>/splits/split.json"
echo ""
echo "Implement dataset unpacking -> canonical NIfTI names using your local data structure."
