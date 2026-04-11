#!/usr/bin/env python3
"""Scan magnetic configuration space on a fixed structure.

Usage:
    python spin_scan.py <checkpoint.pt> <structure.xyz> [--index 0]

Produces spin_scan.dat and spin_scan.png with four scans:
  1. FM -> two-sublattice AFM canting
  2. Joint global rotation of positions + spins (SO(3)_diag invariance test)
  3. Fixed-lattice uniform-spin orientation scans in zx, yz, and xy planes
  4. Commensurate q dot r spin spirals along lattice directions
"""

import argparse
from collections import deque
import torch
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read as ase_read


def predict_energy(model, positions, species_t, mag_np, cell, pbc):
    from mini_magp.model import compute_forces_and_fields
    mag = torch.tensor(mag_np, dtype=torch.float32)
    with torch.no_grad(), torch.enable_grad():
        energy, _, _ = compute_forces_and_fields(
            model, positions, species_t, mag, cell=cell, pbc=pbc, compute_heff=False)
    return energy.item()


def rotation_y(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def make_uniform_moments(n_atoms, mag_indices, mean_norm, direction):
    mag = np.zeros((n_atoms, 3), dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    mag[mag_indices] = mean_norm * direction
    return mag


def make_plane_moments(n_atoms, mag_indices, mean_norm, angle_rad, plane):
    if plane == "zx":
        direction = [np.sin(angle_rad), 0.0, np.cos(angle_rad)]
    elif plane == "yz":
        direction = [0.0, np.sin(angle_rad), np.cos(angle_rad)]
    elif plane == "xy":
        direction = [np.cos(angle_rad), np.sin(angle_rad), 0.0]
    else:
        raise ValueError(f"Unknown spin-orientation plane: {plane}")
    return make_uniform_moments(n_atoms, mag_indices, mean_norm, direction)


def make_spiral_moments(n_atoms, mag_indices, mean_norm, phases):
    mag = np.zeros((n_atoms, 3), dtype=np.float32)
    mag[mag_indices, 0] = mean_norm * np.sin(phases)
    mag[mag_indices, 2] = mean_norm * np.cos(phases)
    return mag


def split_magnetic_sublattices(atoms, mag_indices):
    """Bipartition the nearest-neighbor magnetic graph when possible."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    if len(mag_indices) < 2:
        return mag_indices, mag_indices[:0], "single magnetic sublattice"

    mag_atoms = Atoms(
        symbols=["X"] * len(mag_indices),
        positions=atoms.positions[mag_indices],
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    i_all, j_all, d_all = neighbor_list("ijd", mag_atoms, 8.0)
    pair_mask = i_all < j_all
    i_all, j_all, d_all = i_all[pair_mask], j_all[pair_mask], d_all[pair_mask]
    if len(d_all) == 0:
        return mag_indices[::2], mag_indices[1::2], "index parity fallback"

    nn_cutoff = float(d_all.min()) * 1.25
    edges = [
        (int(i), int(j))
        for i, j, d in zip(i_all, j_all, d_all)
        if float(d) <= nn_cutoff
    ]
    graph = [[] for _ in range(len(mag_indices))]
    for i, j in edges:
        graph[i].append(j)
        graph[j].append(i)

    colors = [None] * len(mag_indices)
    is_bipartite = True
    for start in range(len(mag_indices)):
        if colors[start] is not None:
            continue
        colors[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nbr in graph[node]:
                if colors[nbr] is None:
                    colors[nbr] = 1 - colors[node]
                    queue.append(nbr)
                elif colors[nbr] == colors[node]:
                    is_bipartite = False

    if not is_bipartite or 0 not in colors or 1 not in colors:
        return mag_indices[::2], mag_indices[1::2], "index parity fallback"

    colors = np.array(colors)
    return (
        mag_indices[colors == 0],
        mag_indices[colors == 1],
        f"nearest-neighbor graph bipartition, cutoff={nn_cutoff:.3f} A",
    )


def relative_mev_per_mag_atom(energies, n_mag):
    energies = np.asarray(energies)
    return (energies - energies[0]) * 1000.0 / max(n_mag, 1)


def relative_to_reference_mev_per_mag_atom(energies, reference, n_mag):
    energies = np.asarray(energies)
    return (energies - reference) * 1000.0 / max(n_mag, 1)


def main():
    parser = argparse.ArgumentParser(description="Scan magnetic configuration space")
    parser.add_argument("model", help="Path to checkpoint .pt")
    parser.add_argument("structure", help="Path to extxyz file")
    parser.add_argument("--index", type=int, default=0, help="Structure index")
    parser.add_argument("--magnetic-element", default="Cr", help="Magnetic element symbol")
    args = parser.parse_args()

    from mini_magp.model import MagPot, infer_mag_head_mode_from_state_dict
    from mini_magp.data import ensure_vector_moments

    device = torch.device("cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    hparams = dict(ckpt["hparams"])
    hparams.setdefault(
        "mag_head_mode",
        infer_mag_head_mode_from_state_dict(ckpt["state_dict"]),
    )
    species_map = ckpt["species_map"]
    model = MagPot(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.enable_scaler_if_available()
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

    def pred(mag_np, pos_t=positions, cell_t=cell):
        return predict_energy(model, pos_t, species_t, mag_np, cell_t, pbc)

    # Scan 1: FM -> AFM canting using a physical magnetic sublattice split
    # when the nearest-neighbor magnetic graph is bipartite. The old index-parity
    # split can produce misleading curves because atom order is not a sublattice.
    sub_A, sub_B, split_note = split_magnetic_sublattices(atoms, cr_indices)
    print(f"Scan 1 sublattices: {len(sub_A)} + {len(sub_B)} ({split_note})")
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

    # Scan 2: Joint global rotation.
    #
    # SO(3)_diag invariance means E({R r_i}, {R m_i}, R cell) = E({r_i}, {m_i}, cell).
    # Rotating spins only at fixed lattice is a physical anisotropy scan, not an
    # invariance test, because the spins change relative to the crystal axes.
    angles2 = np.linspace(0, 360, 37)
    e2 = []
    mag0 = make_uniform_moments(n_atoms, cr_indices, mean_norm, [0.0, 0.0, 1.0])
    e_fm_z = pred(mag0)
    positions_np = atoms.positions.astype(np.float32)
    cell_np = np.array(atoms.cell, dtype=np.float32)
    for phi_deg in angles2:
        phi = np.radians(phi_deg)
        R = rotation_y(phi)

        pos_rot = torch.tensor(positions_np @ R.T, dtype=torch.float32)
        cell_rot = torch.tensor(cell_np @ R.T, dtype=torch.float32).unsqueeze(0)
        mag_rot = mag0 @ R.T
        e2.append(pred(mag_rot, pos_t=pos_rot, cell_t=cell_rot))
    spread = max(e2) - min(e2)
    print(f"Scan 2: SO(3)_diag joint-rotation spread = {spread*1000:.4f} meV")

    # Scan 3: Fixed-lattice uniform spin orientation.
    #
    # These curves measure magnetocrystalline anisotropy-like behavior. They
    # should not be invariant because the lattice is held fixed while only the
    # spin direction changes. The zx/yz scans start from +z; the xy scan is a
    # pure in-plane rotation and is plotted relative to +z for easy comparison.
    angles3 = np.linspace(0, 360, 37)
    anisotropy_scans = {"zx": [], "yz": [], "xy": []}
    for angle_deg in angles3:
        angle = np.radians(angle_deg)
        for plane in anisotropy_scans:
            anisotropy_scans[plane].append(
                pred(make_plane_moments(n_atoms, cr_indices, mean_norm, angle, plane))
            )
    anisotropy_span = {
        plane: (max(vals) - min(vals)) * 1000.0
        for plane, vals in anisotropy_scans.items()
    }
    print(
        "Scan 3: fixed-lattice uniform-spin spans "
        + ", ".join(f"{p}={s:.1f} meV" for p, s in anisotropy_span.items())
    )

    # Scan 4: Commensurate q dot r spin spirals.
    #
    # For integer winding w and fractional coordinate f, phi_i = 2*pi*w*f_i is
    # periodic in the simulation cell. This is still a model diagnostic, not a
    # replacement for a DFT generalized-Bloch spin-spiral calculation, but it is
    # tied to lattice coordinates and is much stricter than x-sorted atom order.
    frac = atoms.get_scaled_positions(wrap=True)[cr_indices]
    windings = np.arange(0, 7, dtype=int)
    spiral_coords = {
        "q_a": frac[:, 0],
        "q_b": frac[:, 1],
        "q_a+b": np.mod(frac[:, 0] + frac[:, 1], 1.0),
    }
    spiral_scans = {name: [] for name in spiral_coords}
    for winding in windings:
        for name, coord in spiral_coords.items():
            phases = 2.0 * np.pi * winding * coord
            spiral_scans[name].append(
                pred(make_spiral_moments(n_atoms, cr_indices, mean_norm, phases))
            )
    print(
        "Scan 4: q dot r spiral E(w=1)-E(FM) "
        + ", ".join(
            f"{name}={(vals[1] - vals[0]) * 1000.0 / max(n_cr, 1):.3f} meV/{mag_elem}"
            for name, vals in spiral_scans.items()
        )
    )

    # Write data
    with open("spin_scan.dat", "w") as f:
        f.write(f"# Scan 1: FM->AFM canting ({split_note})\n# angle(deg) energy(eV)\n")
        for a, e in zip(angles1, e1):
            f.write(f"{a:.1f} {e:.6f}\n")
        f.write("\n\n# Scan 2: Joint global rotation (positions+cell+spins)\n# angle(deg) energy(eV)\n")
        for a, e in zip(angles2, e2):
            f.write(f"{a:.1f} {e:.6f}\n")
        f.write("\n\n# Scan 3: Fixed-lattice uniform-spin orientation\n")
        f.write("# angle(deg) energy_zx(eV) energy_yz(eV) energy_xy(eV)\n")
        for idx, a in enumerate(angles3):
            f.write(
                f"{a:.1f} "
                f"{anisotropy_scans['zx'][idx]:.6f} "
                f"{anisotropy_scans['yz'][idx]:.6f} "
                f"{anisotropy_scans['xy'][idx]:.6f}\n"
            )
        f.write("\n\n# Scan 4: Commensurate q dot r spin spirals\n")
        f.write("# winding energy_q_a(eV) energy_q_b(eV) energy_q_a+b(eV)\n")
        for idx, winding in enumerate(windings):
            f.write(
                f"{int(winding)} "
                f"{spiral_scans['q_a'][idx]:.6f} "
                f"{spiral_scans['q_b'][idx]:.6f} "
                f"{spiral_scans['q_a+b'][idx]:.6f}\n"
            )

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8))
    ax = axes.ravel()

    ax[0].plot(angles1, relative_mev_per_mag_atom(e1, n_cr), "o-", ms=3)
    ax[0].set_xlabel("B-sublattice canting angle (deg)")
    ax[0].set_ylabel("Delta E (meV / magnetic atom)")
    ax[0].set_title("Two-sublattice canting")

    ax[1].plot(angles2, relative_mev_per_mag_atom(e2, n_cr), "o-", ms=3, color="C1")
    ax[1].set_xlabel("Joint rotation angle (deg)")
    ax[1].set_ylabel("Delta E (meV / magnetic atom)")
    ax[1].set_title(f"SO(3)_diag check, spread={spread*1000:.3f} meV")

    for plane, vals in anisotropy_scans.items():
        ax[2].plot(
            angles3,
            relative_to_reference_mev_per_mag_atom(vals, e_fm_z, n_cr),
            "o-",
            ms=3,
            label=plane,
        )
    ax[2].set_xlabel("Uniform spin direction angle (deg)")
    ax[2].set_ylabel("Delta E (meV / magnetic atom)")
    ax[2].set_title("Fixed-lattice anisotropy")
    ax[2].legend(frameon=False, fontsize=8)

    for name, vals in spiral_scans.items():
        ax[3].plot(windings, relative_mev_per_mag_atom(vals, n_cr), "o-", ms=3, label=name)
    ax[3].set_xlabel("Integer winding in simulation cell")
    ax[3].set_ylabel("Delta E (meV / magnetic atom)")
    ax[3].set_title("Commensurate q dot r spirals")
    ax[3].legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig("spin_scan.png", dpi=100, bbox_inches="tight")
    print("Saved spin_scan.dat and spin_scan.png")


if __name__ == "__main__":
    main()
