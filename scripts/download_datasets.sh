#!/usr/bin/env bash
set -euo pipefail

# This script only downloads datasets that have public direct links.
# Many medical datasets require registration; see datasets/README.md

mkdir -p datasets

echo "MSD datasets can be downloaded from official mirrors or via MONAI's DecathlonDataset helper."
echo "KiTS19 and BTCV/Synapse usually require registration; download manually into datasets/<name>/raw/."
