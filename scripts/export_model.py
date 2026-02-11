from __future__ import annotations
import argparse, os, torch, json
from fedsam3d.models import FedSAM3D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cfg = json.loads(open(os.path.join(args.run_dir, "config.json")).read())
    ckpt = torch.load(os.path.join(args.run_dir, "global_final.pt"), map_location="cpu")

    model = FedSAM3D(cfg)
    model.load_trainable_state_dict(ckpt["trainable_state"])
    torch.save(model.state_dict(), args.out)
    print("Exported full state_dict to:", args.out)

if __name__ == "__main__":
    main()
