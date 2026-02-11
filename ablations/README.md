# Datasets

This repo supports the 5-client federation described in the paper:

- MSD Spleen (single-organ spleen)
- MSD Liver (single-organ liver; tumor labels ignored)
- MSD Pancreas (single-organ pancreas; tumor labels ignored)
- KiTS19 Kidney (kidney+tumor; tumor ignored; left/right not separated)
- BTCV/Synapse 13-organ abdomen (multi-organ)

**Important:** Many medical datasets require registration / data-use agreements.
This repository does **not** redistribute any data.

---

## Expected folder layout

For each dataset:
```
datasets/<dataset_name>/
  raw/            # put original downloads here
  preprocessed/   # created by your preprocessing
  splits/         # json lists for train/val/test
```

---

## Preprocessed format expected by this repo

```
datasets/<dataset_name>/preprocessed/
  images/<case_id>.nii.gz
  labels/<case_id>.nii.gz
datasets/<dataset_name>/splits/split.json
```

Where `split.json` contains:
```json
{"train": ["case001", "..."], "val": ["case101", "..."]}
```

---

## Notes

- Medical Segmentation Decathlon (MSD) tasks provide a `dataset.json` describing label ids; we keep only the organ mask.
- KiTS19 labels are typically: 0 background, 1 kidney, 2 tumor; we keep kidney only.
- BTCV/Synapse uses a 13-organ abdomen label set.

You must adapt preprocessing to your local downloads.
