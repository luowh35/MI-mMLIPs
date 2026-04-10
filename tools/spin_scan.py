#!/usr/bin/env python3
"""Scan magnetic configuration space on a fixed structure.

Usage:
    python spin_scan.py <checkpoint.pt> <structure.xyz> [--index 0]

Produces spin_scan.dat and spin_scan.png with three scans:
  1. FM -> AFM (sublattice rotation)
  2. Global rotation (SO(3) invariance test)
  3. Spiral spin texture
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read as ase_read


def predict_energy(model, positions, species_t, mag_np, cell, pbc, species_map):
    from mini_magp.model import compute_forces_and_fields
    mag = torch.tensor(mag_np, dtype=torch.float32)
    with torch.no_grad(), torch.enable_grad():
        energy, _, _ = compute_forces_and_fields(
            model, positions, species_t, mag, cell=cell, pbc=pbc, compute_heff=False)
    return energy.item()


def main():
    parser = argparse.ArgumentParser(description="Scan magnetic configuration space")
    parser.add_argument("model", help="Path to checkpoint .pt")
    parser.add_argument("structure", help="Path to extxyz file")
    parser.add_argument("--index", type=int, default=0, help="Structure index")
    parser.add_argument("--magnetic-element", default="Cr", help="Magnetic element symbol")
    args = parser.parse_args()

    from mini_magp.model import MagPot
    from mini_magp.data import ensure_vector_moments

    device = torch.device("cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    hparams = ckpt["hparams"]
    species_map = ckpt["species_map"]
    model = MagPot(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    atoms = ase_read(args.structure, index=args.index)
    symbols = np.array(atoms.get_chemical_symbols())
    species = np.array([species_map[s] for s in symbols], dtype=np.int64)
    positions = torch.tensor(atoms.positions, dtype=torch.float32)
    species_t = torch.tensor(species, dtype=torch.long)
    cell = torch.tensor(np.array(atoms.cell), dtype=torch.float32).unsqueeze(0)
    pbc = torch.tensor(atoms.pbc, dtype=torch.bool)

    mag_elem = args.magnetic_element
    cr_mask = symbols == mag_elem
    cr_indices = np.where(cr_mask)[0]
    n_atoms = len(atoms)
    n_cr = cr_mask.sum()

    m_key = "magnetic_moment" if "magnetic_moment" in atoms.arrays else "magnetic_moments"
    m_orig = atoms.arrays.get(m_key)
    if m_orig is not None:
        m_orig = ensure_vector_moments(m_orig, n_atoms)
        mean_norm = np.mean(np.linalg.norm(m_orig[cr_mask], axis=1))
    else:
        mean_norm = 3.0
    print(f"Structure: {n_atoms} atoms, {n_cr} {mag_elem}, |m| mean={mean_norm:.3f}")

    def pred(mag_np):
        return predict_energy(model, positions, species_t, mag_np, cell, pbc, species_map)

    # Scan 1: FM -> AFM (sublattice rotation)
    sub_A = cr_indices[::2]
    sub_B = cr_indices[1::2]
    angles1 = np.linspace(0, 180, 37)
    e1 = []
    for theta_deg in angles1:
        theta = np.radians(theta_deg)
        mag = np.zeros((n_atoms, 3), dtype=np.float32)
        mag[sub_A, 2] = mean_norm
        mag[sub_B, 0] = mean_norm * np.sin(theta)
        mag[sub_B, 2] = mean_norm * np.cos(theta)
        e1.append(pred(mag))
    print(f"Scan 1: E(FM)={e1[0]:.4f}, E(AFM)={e1[-1]:.4f}, diff={e1[-1]-e1[0]:.4f} eV")

    # Scan 2: Global rotation
    angles2 = np.linspace(0, 360, 37)
    e2 = []
    for phi_deg in angles2:
        phi = np.radians(phi_deg)
        mag = np.zeros((n_atoms, 3), dtype=np.float32)
        mag[cr_mask, 0] = mean_norm * np.sin(phi)
        mag[cr_mask, 2] = mean_norm * np.cos(phi)
        e2.append(pred(mag))
    spread = max(e2) - min(e2)
    print(f"Scan 2: SO(3) spread = {spread*1000:.1f} meV")

    # Scan 3: Spiral
    cr_x = atoms.positions[cr_mask, 0]
    cr_order = np.argsort(cr_x)
    angles3 = np.linspace(0, 360, 19)
    e3 = []
    for twist_deg in angles3:
        mag = np.zeros((n_atoms, 3), dtype=np.float32)
        for rank, idx in enumerate(cr_order):
            phi = np.radians(twist_deg * rank / n_cr)
            mag[cr_indices[idx], 0] = mean_norm * np.sin(phi)
            mag[cr_indices[idx], 2] = mean_norm * np.cos(phi)
        e3.append(pred(mag))
    print(f"Scan 3: E(0)={e3[0]:.4f}, E(180)={e3[9]:.4f}")

    # Write data
    with open("spin_scan.dat", "w") as f:
        f.write("# Scan 1: FM->AFM sublattice rotation\n# angle(deg) energy(eV)\n")
        for a, e in zip(angles1, e1):
            f.write(f"{a:.1f} {e:.6f}\n")
        f.write("\n\n# Scan 2: Global rotation\n# angle(deg) energy(eV)\n")
        for a, e in zip(angles2, e2):
            f.write(f"{a:.1f} {e:.6f}\n")
        f.write("\n\n# Scan 3: Spiral\n# twist(deg) energy(eV)\n")
        for a, e in zip(angles3, e3):
            f.write(f"{a:.1f} {e:.6f}\n")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(angles1, e1, "o-", ms=3)
    axes[0].set_xlabel("Sublattice angle (deg)")
    axes[0].set_ylabel("Energy (eV)")
    axes[0].set_title("FM(0) -> AFM(180)")

    axes[1].plot(angles2, e2, "o-", ms=3, color="C1")
    axes[1].set_xlabel("Global rotation (deg)")
    axes[1].set_ylabel("Energy (eV)")
    axes[1].set_title(f"SO(3) test (spread={spread*1000:.1f} meV)")

    axes[2].plot(angles3, e3, "o-", ms=3, color="C2")
    axes[2].set_xlabel("Total twist (deg)")
    axes[2].set_ylabel("Energy (eV)")
    axes[2].set_title("Spiral")

    plt.tight_layout()
    plt.savefig("spin_scan.png", dpi=100, bbox_inches="tight")
    print("Saved spin_scan.dat and spin_scan.png")


if __name__ == "__main__":
    main()
