import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gmr_refactor.src.attention_vae_3d import AttentionVAE3D, count_parameters, vae_loss
from gmr_refactor.src.common import NPZPatchDataset, append_csv_row, load_json, save_json, set_seed


def run_epoch(model, loader, optimizer, device, beta, recon_loss, max_batches=0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_samples = 0
    n_batches = 0

    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            losses = vae_loss(out["recon"], xb, out["mu"], out["logvar"], beta=beta, recon_loss=recon_loss)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(xb)
                losses = vae_loss(out["recon"], xb, out["mu"], out["logvar"], beta=beta, recon_loss=recon_loss)

        batch_size = int(xb.shape[0])
        total_loss += float(losses["loss"].detach().cpu()) * batch_size
        total_recon += float(losses["recon"].detach().cpu()) * batch_size
        total_kl += float(losses["kl"].detach().cpu()) * batch_size
        total_samples += batch_size
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    denom = max(total_samples, 1)
    return {
        "loss": total_loss / denom,
        "recon": total_recon / denom,
        "kl": total_kl / denom,
        "samples": total_samples,
        "batches": n_batches,
    }


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_meta = load_json(args.dataset_meta)
    train_ds = NPZPatchDataset(args.dataset_npz, split="train")
    val_ds = NPZPatchDataset(args.dataset_npz, split="val")

    patch_shape = tuple(int(v) for v in dataset_meta["patch_size_dhw"])
    model = AttentionVAE3D(
        in_channels=int(train_ds.patches.shape[1]),
        input_shape_dhw=patch_shape,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
    ).to(device)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "gmr_vae.pt")
    meta_path = os.path.join(args.out_dir, "gmr_vae_meta.json")
    history_path = os.path.join(args.out_dir, "gmr_vae_history.csv")

    best_val = float("inf")
    best_epoch = None
    t0 = time.time()
    print(f"[GMR-VAE] device={device} train={len(train_ds)} val={len(val_ds)} params={count_parameters(model)}")

    for epoch in range(1, args.epochs + 1):
        beta = args.beta_kl * (min(1.0, epoch / args.beta_warmup_epochs) if args.beta_warmup_epochs > 0 else 1.0)
        t_epoch = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, beta, args.recon_loss, args.max_train_batches)
        val_metrics = run_epoch(model, val_loader, None, device, beta, args.recon_loss, args.max_val_batches)
        scheduler.step()

        row = {
            "epoch": epoch,
            "beta": beta,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": train_metrics["loss"],
            "train_recon": train_metrics["recon"],
            "train_kl": train_metrics["kl"],
            "train_samples": train_metrics["samples"],
            "val_loss": val_metrics["loss"],
            "val_recon": val_metrics["recon"],
            "val_kl": val_metrics["kl"],
            "val_samples": val_metrics["samples"],
            "epoch_sec": time.time() - t_epoch,
        }
        append_csv_row(history_path, row, encoding="utf-8")
        print(
            f"[GMR-VAE] epoch {epoch:03d}/{args.epochs:03d} "
            f"train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
            f"recon={row['val_recon']:.6f} kl={row['val_kl']:.6f} ({row['epoch_sec']:.1f}s)"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config_dict(),
                    "train_args": vars(args),
                    "dataset_meta": dataset_meta,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "best_val_recon": val_metrics["recon"],
                    "best_val_kl": val_metrics["kl"],
                },
                ckpt_path,
            )
            save_json(
                {
                    "checkpoint": ckpt_path.replace("\\", "/"),
                    "history_csv": history_path.replace("\\", "/"),
                    "model_config": model.config_dict(),
                    "train_args": vars(args),
                    "dataset_npz": args.dataset_npz.replace("\\", "/"),
                    "dataset_meta_path": args.dataset_meta.replace("\\", "/"),
                    "dataset_meta": dataset_meta,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "best_val_recon": val_metrics["recon"],
                    "best_val_kl": val_metrics["kl"],
                    "runtime_sec_so_far": time.time() - t0,
                },
                meta_path,
            )

    final_meta = load_json(meta_path) if os.path.exists(meta_path) else {}
    final_meta["runtime_sec"] = time.time() - t0
    final_meta["finished"] = True
    save_json(final_meta, meta_path)
    print(f"[GMR-VAE] best_epoch={best_epoch} best_val={best_val:.6f} saved={ckpt_path}")
    return final_meta


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D Attention VAE for GMR.")
    parser.add_argument("--dataset_npz", default=os.path.join("gmr_refactor", "data", "prior_patch_dataset.npz"))
    parser.add_argument("--dataset_meta", default=os.path.join("gmr_refactor", "data", "prior_patch_dataset_meta.json"))
    parser.add_argument("--out_dir", default=os.path.join("gmr_refactor", "checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--beta_kl", type=float, default=1e-4)
    parser.add_argument("--beta_warmup_epochs", type=int, default=10)
    parser.add_argument("--recon_loss", choices=["mse", "l1"], default="mse")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
