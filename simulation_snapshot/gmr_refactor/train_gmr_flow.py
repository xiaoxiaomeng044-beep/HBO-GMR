import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gmr_refactor.src.attention_vae_3d import AttentionVAE3D
from gmr_refactor.src.common import NPZPatchDataset, append_csv_row, load_json, save_json, set_seed
from gmr_refactor.src.latent_flow import RealNVPFlow, count_parameters


def load_vae_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    vae = AttentionVAE3D(
        in_channels=int(cfg["in_channels"]),
        input_shape_dhw=tuple(int(v) for v in cfg["input_shape_dhw"]),
        latent_dim=int(cfg["latent_dim"]),
        base_channels=int(cfg["base_channels"]),
    ).to(device)
    vae.load_state_dict(ckpt["model_state_dict"], strict=True)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, ckpt


def run_epoch(vae, flow, loader, optimizer, device, max_batches=0):
    is_train = optimizer is not None
    flow.train(is_train)
    total_nll = 0.0
    total_samples = 0
    n_batches = 0
    nll_values = []

    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.no_grad():
            z = vae.encode_mu(xb)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            nll = flow.nll(z)
            loss = nll.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
            optimizer.step()
        else:
            with torch.no_grad():
                nll = flow.nll(z)
                loss = nll.mean()

        batch_size = int(xb.shape[0])
        total_nll += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        n_batches += 1
        nll_values.append(nll.detach().cpu().numpy())
        if max_batches and n_batches >= max_batches:
            break

    all_nll = np.concatenate(nll_values) if nll_values else np.asarray([], dtype=float)
    denom = max(total_samples, 1)
    return {
        "nll": total_nll / denom,
        "samples": total_samples,
        "batches": n_batches,
        "nll_q50": float(np.quantile(all_nll, 0.50)) if all_nll.size else float("nan"),
        "nll_q75": float(np.quantile(all_nll, 0.75)) if all_nll.size else float("nan"),
        "nll_q90": float(np.quantile(all_nll, 0.90)) if all_nll.size else float("nan"),
    }


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    vae, vae_ckpt = load_vae_checkpoint(args.vae_ckpt, device=device)
    latent_dim = int(vae_ckpt["model_config"]["latent_dim"])
    flow = RealNVPFlow(
        dim=latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        scale_clip=args.scale_clip,
    ).to(device)

    train_ds = NPZPatchDataset(args.dataset_npz, split="train")
    val_ds = NPZPatchDataset(args.dataset_npz, split="val")
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "gmr_flow.pt")
    meta_path = os.path.join(args.out_dir, "gmr_flow_meta.json")
    history_path = os.path.join(args.out_dir, "gmr_flow_history.csv")

    best_val = float("inf")
    best_epoch = None
    t0 = time.time()
    print(f"[GMR-Flow] device={device} train={len(train_ds)} val={len(val_ds)} params={count_parameters(flow)}")

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_metrics = run_epoch(vae, flow, train_loader, optimizer, device, args.max_train_batches)
        val_metrics = run_epoch(vae, flow, val_loader, None, device, args.max_val_batches)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_nll": train_metrics["nll"],
            "train_samples": train_metrics["samples"],
            "val_nll": val_metrics["nll"],
            "val_samples": val_metrics["samples"],
            "val_nll_q50": val_metrics["nll_q50"],
            "val_nll_q75": val_metrics["nll_q75"],
            "val_nll_q90": val_metrics["nll_q90"],
            "epoch_sec": time.time() - t_epoch,
        }
        append_csv_row(history_path, row, encoding="utf-8")
        print(
            f"[GMR-Flow] epoch {epoch:03d}/{args.epochs:03d} "
            f"train_nll={row['train_nll']:.6f} val_nll={row['val_nll']:.6f} "
            f"q75={row['val_nll_q75']:.6f} q90={row['val_nll_q90']:.6f} ({row['epoch_sec']:.1f}s)"
        )

        if val_metrics["nll"] < best_val:
            best_val = val_metrics["nll"]
            best_epoch = epoch
            torch.save(
                {
                    "flow_state_dict": flow.state_dict(),
                    "flow_config": flow.config_dict(),
                    "vae_checkpoint": args.vae_ckpt.replace("\\", "/"),
                    "vae_model_config": vae_ckpt["model_config"],
                    "train_args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_nll": best_val,
                    "best_val_nll_q50": val_metrics["nll_q50"],
                    "best_val_nll_q75": val_metrics["nll_q75"],
                    "best_val_nll_q90": val_metrics["nll_q90"],
                },
                ckpt_path,
            )
            save_json(
                {
                    "checkpoint": ckpt_path.replace("\\", "/"),
                    "history_csv": history_path.replace("\\", "/"),
                    "flow_config": flow.config_dict(),
                    "vae_checkpoint": args.vae_ckpt.replace("\\", "/"),
                    "vae_model_config": vae_ckpt["model_config"],
                    "dataset_npz": args.dataset_npz.replace("\\", "/"),
                    "train_args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_nll": best_val,
                    "best_val_nll_q50": val_metrics["nll_q50"],
                    "best_val_nll_q75": val_metrics["nll_q75"],
                    "best_val_nll_q90": val_metrics["nll_q90"],
                    "runtime_sec_so_far": time.time() - t0,
                },
                meta_path,
            )

    final_meta = load_json(meta_path) if os.path.exists(meta_path) else {}
    final_meta["runtime_sec"] = time.time() - t0
    final_meta["finished"] = True
    save_json(final_meta, meta_path)
    print(f"[GMR-Flow] best_epoch={best_epoch} best_val_nll={best_val:.6f} saved={ckpt_path}")
    return final_meta


def parse_args():
    parser = argparse.ArgumentParser(description="Train RealNVP latent prior for GMR.")
    parser.add_argument("--dataset_npz", default=os.path.join("gmr_refactor", "data", "prior_patch_dataset.npz"))
    parser.add_argument("--vae_ckpt", default=os.path.join("gmr_refactor", "checkpoints", "gmr_vae.pt"))
    parser.add_argument("--out_dir", default=os.path.join("gmr_refactor", "checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--scale_clip", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
