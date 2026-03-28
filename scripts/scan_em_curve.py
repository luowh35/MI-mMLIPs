#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spin_mlips.data import ExtXYZDataset
from spin_mlips.descriptors import InvariantDescriptorBuilder
from spin_mlips.model import LocalInvariantPotential


def _parse_indices(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan E(m) for one fixed structure from extxyz.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--extxyz", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--m-min", type=float, default=0.0)
    parser.add_argument("--m-max", type=float, default=3.5)
    parser.add_argument("--num-points", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--scan-mode",
        type=str,
        default="global",
        choices=["global", "local", "subset"],
        help="global: all atoms set to |m|=m; local: only one atom; subset: selected atoms.",
    )
    parser.add_argument(
        "--atom-index",
        type=int,
        default=0,
        help="Atom index for local scan mode.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="",
        help="Comma-separated atom indices for subset scan mode.",
    )
    parser.add_argument("--output", type=Path, default=Path("em_curve.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    dcfg = dict(ckpt["descriptor_config"])
    mcfg = ckpt["model_config"]

    descriptor = InvariantDescriptorBuilder(**dcfg).to(device)
    model = LocalInvariantPotential(**mcfg).to(device)
    descriptor.load_state_dict(ckpt["descriptor_state"])
    model.load_state_dict(ckpt["model_state"])
    descriptor.eval()
    model.eval()

    ds = ExtXYZDataset([args.extxyz], include_mag_grad=False)
    sample = ds[args.frame_index]
    pos = sample["pos"].to(device=device, dtype=torch.float32)
    cell = sample["cell"].to(device=device, dtype=torch.float32)
    pbc = sample.get("pbc", torch.ones(3, dtype=torch.bool)).to(device=device, dtype=torch.bool)
    mag0 = sample["mag"].to(device=device, dtype=torch.float32)
    n_atoms = int(mag0.shape[0])

    norms = torch.linalg.norm(mag0, dim=-1, keepdim=True)
    unit = torch.zeros_like(mag0)
    mask = (norms.squeeze(-1) > 1e-12)
    unit[mask] = mag0[mask] / norms[mask]
    unit[~mask, 2] = 1.0

    subset_idx = []
    if args.scan_mode == "local":
        if args.atom_index < 0 or args.atom_index >= n_atoms:
            raise IndexError(f"atom-index out of range: {args.atom_index} for N={n_atoms}")
        subset_idx = [args.atom_index]
    elif args.scan_mode == "subset":
        subset_idx = _parse_indices(args.subset)
        if not subset_idx:
            raise ValueError("subset scan mode requires --subset, e.g. --subset 0,1,2")
        for idx in subset_idx:
            if idx < 0 or idx >= n_atoms:
                raise IndexError(f"subset index out of range: {idx} for N={n_atoms}")

    energy_center_per_atom = float(
        ckpt.get(
            "energy_center_per_atom",
            ckpt.get("run_config", {}).get("training", {}).get("energy_center_per_atom", 0.0),
        )
    )

    m_values = np.linspace(args.m_min, args.m_max, args.num_points)
    rows = []

    for m in m_values:
        if args.scan_mode == "global":
            mag = unit * float(m)
        else:
            mag = mag0.clone()
            for idx in subset_idx:
                mag[idx] = unit[idx] * float(m)
        with torch.no_grad():
            desc = descriptor(pos, mag, cell, pbc=pbc)
            e_i = model(desc)
            energy_centered = e_i.sum().item()
            energy_physical = energy_centered + n_atoms * energy_center_per_atom
        rows.append((float(m), float(energy_centered), float(energy_physical)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write("magnitude,energy_centered,energy_physical\n")
        for m, e_c, e_p in rows:
            f.write(f"{m:.8f},{e_c:.10f},{e_p:.10f}\n")

    print(
        "[done] wrote "
        f"{len(rows)} points to {args.output} "
        f"(scan_mode={args.scan_mode}, N={n_atoms}, center_pa={energy_center_per_atom:.8f})"
    )


if __name__ == "__main__":
    main()
