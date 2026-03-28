#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spin_mlips.data import ExtXYZDataset, collate_flat_batch, split_train_val
from spin_mlips.descriptors import InvariantDescriptorBuilder
from spin_mlips.model import LocalInvariantPotential, predict_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot training loss and RMSE diagonal plots (E/F/mag_force)."
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/train_from_json"),
        help="Directory containing metrics/checkpoint/config files.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: run-dir/best.pt, fallback run-dir/last.pt).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config path (default: run-dir/config.resolved.json then config.json).",
    )
    p.add_argument(
        "--eval-files",
        nargs="*",
        default=None,
        help="Optional extxyz files for evaluation; overrides config split.",
    )
    p.add_argument(
        "--eval-split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="When --eval-files is not set, choose split from config.",
    )
    p.add_argument(
        "--max-frames-per-file",
        type=int,
        default=None,
        help="Optional cap for frames loaded per extxyz file.",
    )
    p.add_argument(
        "--max-scatter-points",
        type=int,
        default=120000,
        help="Maximum points shown in each diagonal scatter plot.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path (default: run-dir/training_rmse_panels.png).",
    )
    return p.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_config(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return args.config
    p1 = args.run_dir / "config.resolved.json"
    if p1.exists():
        return p1
    p2 = args.run_dir / "config.json"
    if p2.exists():
        return p2
    raise FileNotFoundError("Config not found. Please pass --config explicitly.")


def _find_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    p1 = args.run_dir / "best.pt"
    if p1.exists():
        return p1
    p2 = args.run_dir / "last.pt"
    if p2.exists():
        return p2
    raise FileNotFoundError("Checkpoint not found. Please pass --checkpoint explicitly.")


def _sample_points(x: np.ndarray, y: np.ndarray, max_points: int, rng: np.random.Generator):
    n = x.shape[0]
    if n <= max_points:
        return x, y
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


def _build_eval_dataset(cfg: Dict[str, Any], args: argparse.Namespace):
    include_mag_grad = True
    if args.eval_files:
        files = [Path(x) for x in args.eval_files]
        return ExtXYZDataset(
            files=files,
            include_mag_grad=include_mag_grad,
            max_frames_per_file=args.max_frames_per_file,
        )

    data_cfg = cfg["data"]
    train_files = [Path(x) for x in data_cfg["train_files"]]
    val_files = [Path(x) for x in data_cfg.get("val_files", [])]

    if val_files:
        files = train_files if args.eval_split == "train" else val_files
        max_frames = args.max_frames_per_file
        if max_frames is None:
            if args.eval_split == "train":
                max_frames = data_cfg.get("max_train_frames_per_file")
            else:
                max_frames = data_cfg.get("max_val_frames_per_file")
        return ExtXYZDataset(
            files=files,
            include_mag_grad=include_mag_grad,
            max_frames_per_file=max_frames,
        )

    max_frames = args.max_frames_per_file
    if max_frames is None:
        max_frames = data_cfg.get("max_train_frames_per_file")
    full = ExtXYZDataset(
        files=train_files,
        include_mag_grad=include_mag_grad,
        max_frames_per_file=max_frames,
    )
    train_ds, val_ds = split_train_val(
        full,
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=int(cfg["training"].get("seed", 42)),
    )
    return train_ds if args.eval_split == "train" else val_ds


def _evaluate(
    *,
    model: LocalInvariantPotential,
    descriptor: InvariantDescriptorBuilder,
    loader: DataLoader,
    device: torch.device,
    energy_per_atom: bool,
    center_per_atom: float,
):
    e_true_list: List[np.ndarray] = []
    e_pred_list: List[np.ndarray] = []
    f_true_list: List[np.ndarray] = []
    f_pred_list: List[np.ndarray] = []
    g_true_list: List[np.ndarray] = []
    g_pred_list: List[np.ndarray] = []

    for batch in loader:
        need_mag_grad = "mag_grad_flat" in batch
        with torch.set_grad_enabled(True):
            pred_energy, pred_forces, pred_mag_grad = predict_batch(
                model=model,
                descriptor_builder=descriptor,
                batch=batch,
                device=device,
                create_graph=False,
                need_mag_grad=need_mag_grad,
            )

        n_atoms = batch["n_atoms"].to(device=device, dtype=torch.float32)
        true_energy = batch["energy"].to(device=device, dtype=torch.float32)
        if energy_per_atom:
            e_true = (true_energy / n_atoms).detach().cpu().numpy()
            e_pred = (pred_energy / n_atoms + center_per_atom).detach().cpu().numpy()
        else:
            e_true = true_energy.detach().cpu().numpy()
            e_pred = (pred_energy + center_per_atom * n_atoms).detach().cpu().numpy()
        e_true_list.append(e_true.reshape(-1))
        e_pred_list.append(e_pred.reshape(-1))

        f_true = batch["forces_flat"].detach().cpu().numpy().reshape(-1)
        f_pred = pred_forces.detach().cpu().numpy().reshape(-1)
        f_true_list.append(f_true)
        f_pred_list.append(f_pred)

        if need_mag_grad and pred_mag_grad is not None:
            g_true = batch["mag_grad_flat"].detach().cpu().numpy().reshape(-1)
            g_pred = pred_mag_grad.detach().cpu().numpy().reshape(-1)
            g_true_list.append(g_true)
            g_pred_list.append(g_pred)

    out = {
        "e_true": np.concatenate(e_true_list) if e_true_list else np.zeros(0),
        "e_pred": np.concatenate(e_pred_list) if e_pred_list else np.zeros(0),
        "f_true": np.concatenate(f_true_list) if f_true_list else np.zeros(0),
        "f_pred": np.concatenate(f_pred_list) if f_pred_list else np.zeros(0),
        "g_true": np.concatenate(g_true_list) if g_true_list else np.zeros(0),
        "g_pred": np.concatenate(g_pred_list) if g_pred_list else np.zeros(0),
    }
    return out


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.size == 0 or true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def main() -> None:
    args = parse_args()
    cfg_path = _find_config(args)
    ckpt_path = _find_checkpoint(args)
    cfg = _load_json(cfg_path)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        print("[warn] CUDA requested but unavailable, using CPU.")

    ckpt = torch.load(ckpt_path, map_location=device)
    dcfg = dict(ckpt["descriptor_config"])
    dcfg.pop("mag_ref", None)  # old checkpoint compatibility
    mcfg = ckpt["model_config"]

    descriptor = InvariantDescriptorBuilder(**dcfg).to(device)
    model = LocalInvariantPotential(**mcfg).to(device)
    descriptor.load_state_dict(ckpt["descriptor_state"])
    model.load_state_dict(ckpt["model_state"])
    descriptor.eval()
    model.eval()

    eval_ds = _build_eval_dataset(cfg, args)
    batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.get("training", {}).get("num_workers", 0)),
        collate_fn=collate_flat_batch,
    )

    energy_per_atom = bool(cfg.get("training", {}).get("energy_per_atom", True))
    center_per_atom = float(cfg.get("training", {}).get("energy_center_per_atom", 0.0))
    eval_out = _evaluate(
        model=model,
        descriptor=descriptor,
        loader=loader,
        device=device,
        energy_per_atom=energy_per_atom,
        center_per_atom=center_per_atom,
    )

    e_rmse = _rmse(eval_out["e_pred"], eval_out["e_true"])
    f_rmse = _rmse(eval_out["f_pred"], eval_out["f_true"])
    g_rmse = _rmse(eval_out["g_pred"], eval_out["g_true"])

    metrics_path = args.run_dir / "metrics.jsonl"
    history = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    history.append(json.loads(line))

    rng = np.random.default_rng(42)
    e_x, e_y = _sample_points(eval_out["e_true"], eval_out["e_pred"], args.max_scatter_points, rng)
    f_x, f_y = _sample_points(eval_out["f_true"], eval_out["f_pred"], args.max_scatter_points, rng)
    g_x, g_y = _sample_points(eval_out["g_true"], eval_out["g_pred"], args.max_scatter_points, rng)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    ax = axes[0, 0]
    if history:
        epochs = [row["epoch"] for row in history]
        train_loss = [row["train"]["loss"] for row in history]
        val_loss = [row["val"]["loss"] for row in history]
        ax.plot(epochs, train_loss, label="train_loss", lw=1.8)
        ax.plot(epochs, val_loss, label="val_loss", lw=1.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted Loss")
        ax.set_title("Training Loss Curve")
        ax.grid(alpha=0.25)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "metrics.jsonl not found", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[0, 1]
    if e_x.size > 0:
        lim_min = float(min(e_x.min(), e_y.min()))
        lim_max = float(max(e_x.max(), e_y.max()))
        ax.scatter(e_x, e_y, s=6, alpha=0.35)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.2)
        ax.set_xlabel("Reference Energy" + (" (eV/atom)" if energy_per_atom else " (eV)"))
        ax.set_ylabel("Predicted Energy" + (" (eV/atom)" if energy_per_atom else " (eV)"))
        ax.set_title(f"Energy Diagonal (RMSE={e_rmse:.6f})")
        ax.grid(alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No energy data", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 0]
    if f_x.size > 0:
        lim_min = float(min(f_x.min(), f_y.min()))
        lim_max = float(max(f_x.max(), f_y.max()))
        ax.scatter(f_x, f_y, s=4, alpha=0.2)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.2)
        ax.set_xlabel("Reference Force Components (eV/Å)")
        ax.set_ylabel("Predicted Force Components (eV/Å)")
        ax.set_title(f"Force Diagonal (RMSE={f_rmse:.6f} eV/Å)")
        ax.grid(alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No force data", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 1]
    if g_x.size > 0:
        lim_min = float(min(g_x.min(), g_y.min()))
        lim_max = float(max(g_x.max(), g_y.max()))
        ax.scatter(g_x, g_y, s=4, alpha=0.2)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.2)
        ax.set_xlabel("Reference Magnetic Force Components")
        ax.set_ylabel("Predicted Magnetic Force Components")
        ax.set_title(f"Magnetic Force Diagonal (RMSE={g_rmse:.6f})")
        ax.grid(alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No magnetic force labels", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    output = args.output or (args.run_dir / "training_rmse_panels.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(f"[done] figure saved to: {output}")
    print(
        f"[metrics] energy_rmse={e_rmse:.6f}, "
        f"force_rmse={f_rmse:.6f}, mag_force_rmse={g_rmse:.6f}"
    )


if __name__ == "__main__":
    main()
