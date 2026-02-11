from __future__ import annotations
import argparse, os, json
import torch
from omegaconf import OmegaConf

from fedsam3d.models import FedSAM3D
from fedsam3d.data.loaders import build_client_loaders
from fedsam3d.engine.evaluator import evaluate_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()

    cfg = json.loads(open(os.path.join(args.run_dir, "config.json")).read())
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # load final checkpoint
    ckpt = torch.load(os.path.join(args.run_dir, "global_final.pt"), map_location="cpu")
    model = FedSAM3D(cfg)
    model.load_trainable_state_dict(ckpt["trainable_state"])
    model.to(device)

    results = {}
    for cid, cinfo in cfg["clients"].items():
        _, val_loader, meta = build_client_loaders(
            dataset_name=cinfo["dataset"],
            data_root=cinfo["data_root"],
            target_spacing=cfg["data"]["target_spacing"],
            patch_size=cfg["data"]["patch_size"],
            batch_size=1,
            num_workers=int(cfg["data"]["num_workers"]),
            intensity_cfg=cfg["data"]["intensity"],
        )
        metrics = evaluate_model(model, val_loader, meta["availability_mask"], device, cfg["data"]["patch_size"])
        results[cid] = metrics
        print(cid, metrics)

    out_path = os.path.join(args.run_dir, "eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
