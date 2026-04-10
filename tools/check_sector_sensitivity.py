#!/usr/bin/env python3
"""Check per-sector descriptor sensitivity to magnetic order.

Takes a structure from the dataset, constructs FM and AFM magnetic
configurations on the same geometry, and compares per-sector descriptor
norms and differences. This verifies that per-sector normalization
prevents amplitude sectors from drowning direction-sensitive sectors.

Usage:
    python check_sector_sensitivity.py <data.xyz> [--mag-element Cr] [--idx 0]
"""

import argparse
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mini_magp.descriptors import (
    compute_all_descriptors, get_mag_sector_dims, get_mag_descriptor_dim,
    get_struct_descriptor_dim,
)
from mini_magp.radial import RadialBasis
from mini_magp.utils import build_neighbor_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to extxyz file")
    parser.add_argument("--mag-element", default="Cr")
    parser.add_argument("--idx", type=int, default=0, help="Structure index")
    parser.add_argument("--r-cutoff", type=float, default=4.7)
    parser.add_argument("--basis-size", type=int, default=12)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("--num-species", type=int, default=2)
    args = parser.parse_args()

    from ase.io import read as ase_read
    atoms = ase_read(args.data, index=args.idx)
    symbols = np.array(atoms.get_chemical_symbols())
    mag_mask = symbols == args.mag_element

    # Build species tensor
    unique_elements = sorted(set(symbols))
    elem_to_idx = {e: i for i, e in enumerate(unique_elements)}
    species = torch.tensor([elem_to_idx[s] for s in symbols], dtype=torch.long)

    positions = torch.tensor(atoms.positions, dtype=torch.float64)
    cell = torch.tensor(np.array(atoms.cell), dtype=torch.float64).unsqueeze(0)
    pbc = torch.tensor(atoms.pbc, dtype=torch.bool)

    # Get original magnetic moments
    m_orig = atoms.arrays.get('magnetic_moment', atoms.arrays.get('magnetic_moments'))
    if m_orig is None:
        print("No magnetic moments found in data")
        return
    m_orig = torch.tensor(m_orig, dtype=torch.float64)
    mag_norm = m_orig[mag_mask].norm(dim=-1).mean().item()
    print(f"Mean |m| for {args.mag_element}: {mag_norm:.3f} μ_B")

    # Construct FM: all magnetic atoms along +z
    m_fm = torch.zeros_like(m_orig)
    m_fm[mag_mask, 2] = mag_norm

    # Construct AFM: checkerboard along z
    m_afm = torch.zeros_like(m_orig)
    mag_indices = torch.where(torch.tensor(mag_mask))[0]
    for i, idx in enumerate(mag_indices):
        m_afm[idx, 2] = mag_norm if i % 2 == 0 else -mag_norm

    # Radial basis
    rb = RadialBasis(args.r_cutoff, args.basis_size, args.n_max, args.num_species)
    rb = rb.double()

    # Neighbor list
    edge_index, r_ij = build_neighbor_list(positions, cell, pbc, args.r_cutoff)
    i_idx, j_idx = edge_index
    dist = r_ij.norm(dim=-1)
    phi = rb(dist, species[i_idx], species[j_idx])

    num_atoms = len(atoms)
    n_max = args.n_max
    sector_dims = get_mag_sector_dims(n_max)
    sector_names = [
        "amplitude",
        "iso_exchange",
        "SIA",
        "SAE",
        "DMI",
        "amp_mixed",
        "neighbor_amp",
        "neighbor_amp_ex",
        "neighbor_amp_mix",
    ]

    configs = {"FM": m_fm, "AFM": m_afm, "original": m_orig}
    results = {}

    for name, m in configs.items():
        desc_struct, desc_mag = compute_all_descriptors(
            edge_index, r_ij, m, phi, num_atoms
        )
        results[name] = desc_mag

    # Per-sector analysis
    print(f"\n{'Sector':<14} {'dim':>4}  {'FM norm²':>12} {'AFM norm²':>12} {'|FM-AFM|²':>12} {'rel_diff%':>10}")
    print("-" * 72)

    offset = 0
    for sname, sdim in zip(sector_names, sector_dims):
        fm_sec = results["FM"][:, offset:offset+sdim]
        afm_sec = results["AFM"][:, offset:offset+sdim]

        fm_norm2 = fm_sec.pow(2).sum().item()
        afm_norm2 = afm_sec.pow(2).sum().item()
        diff_norm2 = (fm_sec - afm_sec).pow(2).sum().item()
        avg_norm2 = (fm_norm2 + afm_norm2) / 2
        rel_diff = diff_norm2 / avg_norm2 * 100 if avg_norm2 > 1e-12 else 0.0

        print(f"{sname:<14} {sdim:>4}  {fm_norm2:>12.4f} {afm_norm2:>12.4f} {diff_norm2:>12.4f} {rel_diff:>9.1f}%")
        offset += sdim

    # After per-sector normalization
    print(f"\n--- After per-sector normalization (simulated) ---")
    print(f"{'Sector':<14} {'dim':>4}  {'FM norm²':>12} {'AFM norm²':>12} {'|FM-AFM|²':>12} {'rel_diff%':>10}")
    print("-" * 72)

    # Compute scaler from all configs combined
    all_desc = torch.cat([results["FM"], results["AFM"], results["original"]], dim=0)
    offset = 0
    total_fm_norm2 = 0.0
    total_afm_norm2 = 0.0
    total_diff_norm2 = 0.0
    for sname, sdim in zip(sector_names, sector_dims):
        sec_all = all_desc[:, offset:offset+sdim]
        shift = sec_all.mean(dim=0)
        scale = 1.0 / sec_all.std(dim=0, unbiased=False).clamp(min=1e-6)

        fm_sec = (results["FM"][:, offset:offset+sdim] - shift) * scale
        afm_sec = (results["AFM"][:, offset:offset+sdim] - shift) * scale

        fm_norm2 = fm_sec.pow(2).sum().item()
        afm_norm2 = afm_sec.pow(2).sum().item()
        diff_norm2 = (fm_sec - afm_sec).pow(2).sum().item()
        avg_norm2 = (fm_norm2 + afm_norm2) / 2
        rel_diff = diff_norm2 / avg_norm2 * 100 if avg_norm2 > 1e-12 else 0.0

        total_fm_norm2 += fm_norm2
        total_afm_norm2 += afm_norm2
        total_diff_norm2 += diff_norm2

        print(f"{sname:<14} {sdim:>4}  {fm_norm2:>12.2f} {afm_norm2:>12.2f} {diff_norm2:>12.2f} {rel_diff:>9.1f}%")
        offset += sdim

    total_avg = (total_fm_norm2 + total_afm_norm2) / 2
    total_rel = total_diff_norm2 / total_avg * 100 if total_avg > 1e-12 else 0.0
    print(f"{'TOTAL':<14} {sum(sector_dims):>4}  {total_fm_norm2:>12.2f} {total_afm_norm2:>12.2f} {total_diff_norm2:>12.2f} {total_rel:>9.1f}%")


if __name__ == "__main__":
    main()
