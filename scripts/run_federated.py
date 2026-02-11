from __future__ import annotations
import argparse, os, time, json
from omegaconf import OmegaConf
import torch

from fedsam3d.models import FedSAM3D
from fedsam3d.data.loaders import build_client_loaders
from fedsam3d.federated import FederatedClient, FederatedServer
from fedsam3d.engine.utils import seed_everything

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    seed_everything(int(cfg["seed"]))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # Create run folder
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["paths"]["runs_dir"], ts)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Global model
    global_model = FedSAM3D(cfg)

    # Clients
    clients = {}
    for cid, cinfo in cfg["clients"].items():
        train_loader, val_loader, meta = build_client_loaders(
            dataset_name=cinfo["dataset"],
            data_root=cinfo["data_root"],
            target_spacing=cfg["data"]["target_spacing"],
            patch_size=cfg["data"]["patch_size"],
            batch_size=int(cfg["data"]["batch_size"]),
            num_workers=int(cfg["data"]["num_workers"]),
            intensity_cfg=cfg["data"]["intensity"],
        )
        model = FedSAM3D(cfg)
        clients[cid] = FederatedClient(cid, model, train_loader, val_loader, meta, cfg)

    server = FederatedServer(global_model, clients, cfg)
    server.run(device=device, out_dir=run_dir)

    print(f"Done. Outputs in: {run_dir}")

if __name__ == "__main__":
    main()
