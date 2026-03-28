#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spin_mlips.data import (
    ExtXYZDataset,
    collate_flat_batch,
    split_train_val,
    split_train_val_by_blocks,
    split_train_val_grouped,
)
from spin_mlips.descriptors import InvariantDescriptorBuilder
from spin_mlips.model import LocalInvariantPotential, predict_batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _strip_json_comments(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    in_string = False
    quote_char = ""
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_string:
            out.append(c)
            if c == "\\":
                i += 1
                if i < n:
                    out.append(text[i])
            elif c == quote_char:
                in_string = False
        else:
            if c in ("'", '"'):
                in_string = True
                quote_char = c
                out.append(c)
            elif c == "/" and nxt == "/":
                i += 2
                while i < n and text[i] != "\n":
                    i += 1
                continue
            elif c == "/" and nxt == "*":
                i += 2
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 1
            else:
                out.append(c)
        i += 1
    return "".join(out)


def _load_jsonc(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    clean = _strip_json_comments(content)
    return json.loads(clean)


def _must(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required key: {key}")
    return cfg[key]


def _to_path_list(values: list[str]) -> list[Path]:
    return [Path(v) for v in values]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def load_config(config_path: Path) -> Dict[str, Any]:
    cfg = _load_jsonc(config_path)
    data_cfg = _must(cfg, "data")
    model_cfg = _must(cfg, "model")
    train_cfg = _must(cfg, "training")
    loss_cfg = _must(cfg, "loss")
    out_cfg = _must(cfg, "output")
    runtime_cfg = _must(cfg, "runtime")

    out_dir = Path(_must(out_cfg, "output_dir"))
    output = {
        "config_path": config_path,
        "data": {
            "train_files": _to_path_list(_must(data_cfg, "train_files")),
            "val_files": _to_path_list(data_cfg.get("val_files", [])),
            "val_ratio": float(data_cfg.get("val_ratio", 0.1)),
            "max_train_frames_per_file": data_cfg.get("max_train_frames_per_file"),
            "max_val_frames_per_file": data_cfg.get("max_val_frames_per_file"),
            "split_mode": str(data_cfg.get("split_mode", "source_file")),
            "split_group_key": str(data_cfg.get("split_group_key", "source_file")),
            "split_block_size": int(data_cfg.get("split_block_size", 50)),
        },
        "model": {
            "cutoff": float(_must(model_cfg, "cutoff")),
            "num_radial": int(_must(model_cfg, "num_radial")),
            "l_max": int(_must(model_cfg, "l_max")),
            "rho_u_basis": str(model_cfg.get("rho_u_basis", "legendre")),
            "rho_u_degree": int(model_cfg.get("rho_u_degree", 4)),
            "u_norm_mode": str(model_cfg.get("u_norm_mode", "dataset")),
            "mag_ref": float(model_cfg.get("mag_ref", 2.2)),
            "m_stat": (
                float(model_cfg["m_stat"])
                if model_cfg.get("m_stat") is not None
                else None
            ),
            "m_stat_mode": str(model_cfg.get("m_stat_mode", "mean")),
            "m_stat_quantile": float(model_cfg.get("m_stat_quantile", 0.5)),
            "u_center": (
                float(model_cfg["u_center"])
                if model_cfg.get("u_center") is not None
                else None
            ),
            "u_scale": (
                float(model_cfg["u_scale"])
                if model_cfg.get("u_scale") is not None
                else None
            ),
            "include_s2": bool(model_cfg.get("include_s2", True)),
            "include_imm": bool(model_cfg.get("include_imm", False)),
            "hidden_dim": int(_must(model_cfg, "hidden_dim")),
            "depth": int(_must(model_cfg, "depth")),
        },
        "training": {
            "epochs": int(_must(train_cfg, "epochs")),
            "lr": float(_must(train_cfg, "lr")),
            "weight_decay": float(train_cfg.get("weight_decay", 1e-6)),
            "seed": int(train_cfg.get("seed", 42)),
            "batch_size": int(train_cfg.get("batch_size", 1)),
            "num_workers": int(train_cfg.get("num_workers", 0)),
            "energy_per_atom": bool(train_cfg.get("energy_per_atom", True)),
        },
        "loss": {
            "use_mag_loss": bool(loss_cfg.get("use_mag_loss", True)),
            "w_energy": float(loss_cfg.get("w_energy", 1.0)),
            "w_force": float(loss_cfg.get("w_force", 20.0)),
            "w_mag": float(loss_cfg.get("w_mag", 1.0)),
        },
        "output": {
            "output_dir": out_dir,
            "save_last": bool(out_cfg.get("save_last", True)),
            "save_best": bool(out_cfg.get("save_best", True)),
            "save_every_epoch": bool(out_cfg.get("save_every_epoch", False)),
        },
        "runtime": {
            "device": str(runtime_cfg.get("device", "cpu")),
        },
    }

    return output


def build_datasets(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    loss_cfg = cfg["loss"]
    train_files = data_cfg["train_files"]
    if not train_files:
        raise ValueError("data.train_files is empty.")

    include_mag_grad = bool(loss_cfg["use_mag_loss"])
    val_files = data_cfg["val_files"]

    if val_files:
        train_ds = ExtXYZDataset(
            train_files,
            include_mag_grad=include_mag_grad,
            max_frames_per_file=data_cfg["max_train_frames_per_file"],
        )
        val_ds = ExtXYZDataset(
            val_files,
            include_mag_grad=include_mag_grad,
            max_frames_per_file=data_cfg["max_val_frames_per_file"],
        )
    else:
        full = ExtXYZDataset(
            train_files,
            include_mag_grad=include_mag_grad,
            max_frames_per_file=data_cfg["max_train_frames_per_file"],
        )
        split_mode = data_cfg["split_mode"]
        if split_mode == "random":
            train_ds, val_ds = split_train_val(
                full,
                val_ratio=data_cfg["val_ratio"],
                seed=cfg["training"]["seed"],
            )
        elif split_mode == "block":
            train_ds, val_ds = split_train_val_by_blocks(
                full,
                block_size=data_cfg["split_block_size"],
                source_key=data_cfg["split_group_key"],
                val_ratio=data_cfg["val_ratio"],
                seed=cfg["training"]["seed"],
            )
        else:
            if split_mode == "source_file":
                group_key = "source_file"
            elif split_mode == "set":
                group_key = "set"
            elif split_mode in {"config_type", "label"}:
                group_key = "config_type"
            else:
                group_key = data_cfg["split_group_key"]

            train_ds, val_ds = split_train_val_grouped(
                full,
                group_key=group_key,
                val_ratio=data_cfg["val_ratio"],
                seed=cfg["training"]["seed"],
            )
    return train_ds, val_ds


def resolve_energy_center_per_atom(train_ds) -> float:
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty; cannot estimate energy center.")

    sum_energy = 0.0
    sum_atoms = 0
    for i in range(len(train_ds)):
        sample = train_ds[i]
        sum_energy += float(sample["energy"])
        sum_atoms += int(sample["pos"].shape[0])

    if sum_atoms <= 0:
        raise ValueError("Invalid atom count while estimating energy center.")
    return sum_energy / float(sum_atoms)


def resolve_m_stat(train_ds, mode: str = "mean", quantile: float = 0.5) -> float:
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty; cannot estimate m_stat.")

    norms = []
    for i in range(len(train_ds)):
        sample = train_ds[i]
        mag = sample["mag"]
        norms.append(torch.linalg.norm(mag, dim=-1).detach().cpu().numpy())

    values = np.concatenate(norms, axis=0)
    if values.size == 0:
        raise ValueError("No magnetic moments found while estimating m_stat.")

    if mode == "mean":
        stat = float(values.mean())
    elif mode == "median":
        stat = float(np.median(values))
    elif mode == "quantile":
        q = float(np.clip(quantile, 0.0, 1.0))
        stat = float(np.quantile(values, q))
    else:
        raise ValueError("m_stat_mode must be one of: mean, median, quantile.")

    if stat <= 0:
        raise ValueError("Estimated m_stat must be positive.")
    return stat


def compute_losses(
    batch: Dict[str, torch.Tensor],
    pred_energy: torch.Tensor,
    pred_forces: torch.Tensor,
    pred_mag_grad: torch.Tensor | None,
    device: torch.device,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_cfg = cfg["loss"]
    train_cfg = cfg["training"]

    target_energy = batch["energy"].to(device=device, dtype=torch.float32)  # [B]
    target_forces = batch["forces_flat"].to(device=device, dtype=torch.float32)  # [N_total, 3]
    n_atoms = batch["n_atoms"].to(device=device, dtype=torch.float32)  # [B]
    center_pa = float(train_cfg.get("energy_center_per_atom", 0.0))

    if train_cfg["energy_per_atom"]:
        pred_e = pred_energy / n_atoms
        tgt_e = target_energy / n_atoms - center_pa
    else:
        pred_e = pred_energy
        tgt_e = target_energy - center_pa * n_atoms

    mse_e = (pred_e - tgt_e).pow(2).mean()
    mse_f = (pred_forces - target_forces).pow(2).mean()

    if loss_cfg["use_mag_loss"] and ("mag_grad_flat" in batch) and (pred_mag_grad is not None):
        target_mag = batch["mag_grad_flat"].to(device=device, dtype=torch.float32)
        mse_g = (pred_mag_grad - target_mag).pow(2).mean()
    else:
        mse_g = torch.tensor(0.0, device=device)

    total = (
        loss_cfg["w_energy"] * mse_e
        + loss_cfg["w_force"] * mse_f
        + loss_cfg["w_mag"] * mse_g
    )
    rmse_e = torch.sqrt(mse_e)
    rmse_f = torch.sqrt(mse_f)
    rmse_g = torch.sqrt(mse_g)
    metrics = {
        "loss": float(total.detach()),
        "loss_e_mse": float(mse_e.detach()),
        "loss_f_mse": float(mse_f.detach()),
        "loss_g_mse": float(mse_g.detach()),
        "loss_e_rmse": float(rmse_e.detach()),
        "loss_f_rmse": float(rmse_f.detach()),
        "loss_g_rmse": float(rmse_g.detach()),
    }
    return total, metrics


def run_epoch(
    *,
    loader: DataLoader,
    model: LocalInvariantPotential,
    descriptor: InvariantDescriptorBuilder,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
        descriptor.train()
    else:
        model.eval()
        descriptor.eval()

    sums = {
        "loss": 0.0,
        "loss_e_mse": 0.0,
        "loss_f_mse": 0.0,
        "loss_g_mse": 0.0,
        "loss_e_rmse": 0.0,
        "loss_f_rmse": 0.0,
        "loss_g_rmse": 0.0,
    }
    n_batches = 0

    for batch in loader:
        need_mag_grad = cfg["loss"]["use_mag_loss"] and ("mag_grad_flat" in batch)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred_energy, pred_forces, pred_mag_grad = predict_batch(
            model=model,
            descriptor_builder=descriptor,
            batch=batch,
            device=device,
            create_graph=is_train,
            need_mag_grad=need_mag_grad,
        )

        if not is_train:
            pred_energy = pred_energy.detach()
            pred_forces = pred_forces.detach()
            if pred_mag_grad is not None:
                pred_mag_grad = pred_mag_grad.detach()

        loss, metrics = compute_losses(
            batch=batch,
            pred_energy=pred_energy,
            pred_forces=pred_forces,
            pred_mag_grad=pred_mag_grad,
            device=device,
            cfg=cfg,
        )

        if is_train:
            loss.backward()
            optimizer.step()

        for k in sums:
            sums[k] += metrics[k]
        n_batches += 1

    if n_batches == 0:
        return {k: float("nan") for k in sums}
    return {k: sums[k] / n_batches for k in sums}


def main() -> None:
    config_path = Path("train.json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Create it first (see repository train.json template)."
        )

    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])

    output_dir: Path = cfg["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_device = cfg["runtime"]["device"]
    device = torch.device(
        "cuda" if runtime_device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if runtime_device == "cuda" and device.type != "cuda":
        print("[warn] CUDA requested but unavailable, using CPU.")

    train_ds, val_ds = build_datasets(cfg)
    print(
        "[info] dataset split mode: "
        f"{cfg['data']['split_mode']} (train={len(train_ds)}, val={len(val_ds)})"
    )

    if cfg["model"]["u_norm_mode"] in {"dataset", "dual"} and cfg["model"]["m_stat"] is None:
        cfg["model"]["m_stat"] = resolve_m_stat(
            train_ds,
            mode=cfg["model"]["m_stat_mode"],
            quantile=cfg["model"]["m_stat_quantile"],
        )
        print(
            "[info] magnetic norm scale m_stat estimated from training set: "
            f"{cfg['model']['m_stat']:.8f} (mode={cfg['model']['m_stat_mode']})"
        )

    cfg["training"]["energy_center_per_atom"] = resolve_energy_center_per_atom(train_ds)
    print(
        "[info] energy center (per atom) estimated from training set: "
        f"{cfg['training']['energy_center_per_atom']:.8f}"
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_flat_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_flat_batch,
    )

    descriptor = InvariantDescriptorBuilder(
        cutoff=cfg["model"]["cutoff"],
        num_radial=cfg["model"]["num_radial"],
        l_max=cfg["model"]["l_max"],
        rho_u_basis=cfg["model"]["rho_u_basis"],
        rho_u_degree=cfg["model"]["rho_u_degree"],
        u_norm_mode=cfg["model"]["u_norm_mode"],
        mag_ref=cfg["model"]["mag_ref"],
        m_stat=cfg["model"]["m_stat"],
        u_center=cfg["model"]["u_center"],
        u_scale=cfg["model"]["u_scale"],
        include_s2=cfg["model"]["include_s2"],
        include_imm=cfg["model"]["include_imm"],
    ).to(device)
    model = LocalInvariantPotential(
        in_dim=descriptor.descriptor_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        depth=cfg["model"]["depth"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    run_config = _json_ready(cfg)
    run_config["runtime"]["device_resolved"] = str(device)
    with (output_dir / "config.resolved.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=True)

    best_val = float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    log_path = output_dir / "metrics.jsonl"

    with log_path.open("w", encoding="utf-8") as logf:
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            t0 = time.time()
            train_metrics = run_epoch(
                loader=train_loader,
                model=model,
                descriptor=descriptor,
                device=device,
                optimizer=optimizer,
                cfg=cfg,
            )
            val_metrics = run_epoch(
                loader=val_loader,
                model=model,
                descriptor=descriptor,
                device=device,
                optimizer=None,
                cfg=cfg,
            )
            dt = time.time() - t0

            row = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "seconds": dt,
            }
            logf.write(json.dumps(row, ensure_ascii=True) + "\n")
            logf.flush()

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "descriptor_state": descriptor.state_dict(),
                "model_config": {
                    "in_dim": descriptor.descriptor_dim,
                    "hidden_dim": cfg["model"]["hidden_dim"],
                    "depth": cfg["model"]["depth"],
                },
                "descriptor_config": {
                    "cutoff": cfg["model"]["cutoff"],
                    "num_radial": cfg["model"]["num_radial"],
                    "l_max": cfg["model"]["l_max"],
                    "rho_u_basis": cfg["model"]["rho_u_basis"],
                    "rho_u_degree": cfg["model"]["rho_u_degree"],
                    "u_norm_mode": cfg["model"]["u_norm_mode"],
                    "mag_ref": cfg["model"]["mag_ref"],
                    "m_stat": cfg["model"]["m_stat"],
                    "u_center": cfg["model"]["u_center"],
                    "u_scale": cfg["model"]["u_scale"],
                    "include_s2": cfg["model"]["include_s2"],
                    "include_imm": cfg["model"]["include_imm"],
                },
                "optimizer_state": optimizer.state_dict(),
                "run_config": run_config,
                "energy_center_per_atom": cfg["training"]["energy_center_per_atom"],
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

            if cfg["output"]["save_last"]:
                torch.save(ckpt, last_path)
            if cfg["output"]["save_best"] and val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(ckpt, best_path)
            if cfg["output"]["save_every_epoch"]:
                torch.save(ckpt, output_dir / f"epoch_{epoch:04d}.pt")

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"train_e_rmse={train_metrics['loss_e_rmse']:.6f} "
                f"train_f_rmse={train_metrics['loss_f_rmse']:.6f} "
                f"train_g_rmse={train_metrics['loss_g_rmse']:.6f} "
                f"val_e_rmse={val_metrics['loss_e_rmse']:.6f} "
                f"val_f_rmse={val_metrics['loss_f_rmse']:.6f} "
                f"val_g_rmse={val_metrics['loss_g_rmse']:.6f} "
                f"time={dt:.2f}s"
            )

    if cfg["output"]["save_best"]:
        print(f"[done] best checkpoint: {best_path}")
    if cfg["output"]["save_last"]:
        print(f"[done] last checkpoint: {last_path}")
    print(f"[done] metrics log: {log_path}")


if __name__ == "__main__":
    main()
