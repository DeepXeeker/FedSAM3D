# FedSAM3D (Option B: paper-aligned reproduction)

FedSAM3D is a cross-silo federated framework for **volumetric multi-organ segmentation** under **label fragmentation**.

This repository implements the paper’s key components:
- **Masked supervision**: supervise only on locally-available organs (missing labels are *not* background).
- **GKD**: server-teacher distillation on missing organs with confidence gating.
- **PKD**: peer-expert distillation on missing organs with confidence gating.
- **Supervision-mass aggregation**: weight client updates by effective labeled signal.
- **SAM-style** prompt-conditioned pipeline adapted to 3D via parameter-efficient depth modules + **Auto Prompt Generator (APG)**.

The code runs a **single-machine simulation** of federated learning (multiple “clients” as separate dataloaders).

---

## Installation

### Conda (recommended; CUDA + cuDNN pinned)
```bash
conda env create -f environment.yml
conda activate fedsam3d
pip install -e .
```

### Segment Anything (SAM) weights (optional backbone)
This implementation can initialize from Meta’s official Segment Anything checkpoints (ViT-B/L/H).
Download a checkpoint and set it in your config:

```yaml
paths:
  sam_checkpoint: /path/to/sam_vit_b_01ec64.pth
model:
  sam_variant: "vit_b"
```

If you do not want to use SAM, you can replace the backbone module with any 3D encoder (not included).

---

## Data layout

Datasets are **not** included. Put each dataset under `datasets/<name>/raw/` and preprocessed outputs under `datasets/<name>/preprocessed/`.

See `datasets/README.md`.

---

## Global organ vocabulary (paper-aligned)

Global organ set `C=13` matches the common BTCV/Synapse 13-organ abdomen benchmark:

1. spleen
2. right_kidney
3. left_kidney
4. gallbladder
5. esophagus
6. liver
7. stomach
8. aorta
9. inferior_vena_cava
10. portal_vein_and_splenic_vein
11. pancreas
12. right_adrenal_gland
13. left_adrenal_gland

**Label mapping across clients** is in `fedsam3d/data/label_maps.py`.

> Note on KiTS19 (kidney): KiTS19 does not separate left/right kidneys; we supervise both kidney classes using the same kidney mask.

---

## Run federated training

Edit `configs/federated_5clients.yaml` to point to your local dataset folders.

```bash
python scripts/run_federated.py --config configs/federated_5clients.yaml
```

Outputs are written under `runs/<timestamp>/`.

---

## Evaluate

```bash
python scripts/evaluate.py --run_dir runs/<timestamp>
```

Metrics:
- Dice (DSC)
- HD95 (95th percentile Hausdorff)

Computed **only for locally-labeled organs**, matching the paper protocol.

---

## Reproduce ablations

```bash
python scripts/run_federated.py --config configs/ablations/no_gkd.yaml
python scripts/run_federated.py --config configs/ablations/no_pkd.yaml
python scripts/run_federated.py --config configs/ablations/no_apg.yaml
python scripts/run_federated.py --config configs/ablations/no_depth_adapter.yaml
```

---

## License

MIT (see `LICENSE`).
