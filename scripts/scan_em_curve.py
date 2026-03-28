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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan E(m) for one fixed structure from extxyz.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--extxyz", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--m-min", type=float, default=0.0)
    parser.add_argument("--m-max", type=float, default=3.5)
    parser.add_argument("--num-points", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", type=Path, default=Path("em_curve.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    dcfg = dict(ckpt["descriptor_config"])
    # Backward compatibility with old checkpoints.
    dcfg.pop("mag_ref", None)
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
    mag0 = sample["mag"].to(device=device, dtype=torch.float32)

    norms = torch.linalg.norm(mag0, dim=-1, keepdim=True)
    unit = mag0 / (norms + 1e-12)

    m_values = np.linspace(args.m_min, args.m_max, args.num_points)
    rows = []

    for m in m_values:
        mag = unit * float(m)
        with torch.no_grad():
            desc = descriptor(pos, mag, cell)
            energy = model(desc).item()
        rows.append((float(m), float(energy)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write("magnitude,energy\n")
        for m, e in rows:
            f.write(f"{m:.8f},{e:.10f}\n")

    print(f"[done] wrote {len(rows)} points to {args.output}")


if __name__ == "__main__":
    main()
