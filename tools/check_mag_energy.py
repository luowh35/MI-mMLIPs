#!/usr/bin/env python3
"""Check DFT energy vs magnetic order for same-geometry structures.

Usage:
    python check_mag_energy.py <data.xyz>

Finds groups of structures sharing the same atomic positions but different
magnetic configurations, and compares their DFT energies.
"""

import argparse
import numpy as np
from ase.io import read as ase_read


def parse_energies(path):
    """Parse energies from xyz comment lines (ASE sometimes misses them)."""
    energies = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                n_atoms = int(line.strip())
            except ValueError:
                continue
            comment = f.readline().strip()
            e = None
            for part in comment.split():
                if part.startswith('energy='):
                    e = float(part.split('=')[1])
            energies.append(e)
            for _ in range(n_atoms):
                f.readline()
    return np.array(energies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to extxyz file")
    parser.add_argument("--mag-element", default="Cr")
    parser.add_argument("--tol", type=float, default=0.01,
                        help="Position tolerance for same-geometry detection")
    args = parser.parse_args()

    atoms_list = ase_read(args.data, index=':')
    energies = parse_energies(args.data)
    n_atoms = len(atoms_list[0])

    # Compute order parameter for each structure
    orders = []
    for atoms in atoms_list:
        symbols = np.array(atoms.get_chemical_symbols())
        mask = symbols == args.mag_element
        m = atoms.arrays.get('magnetic_moment', atoms.arrays.get('magnetic_moments'))
        if m is None or m.ndim == 1:
            orders.append(0.0)
            continue
        cr_m = m[mask]
        cr_norms = np.linalg.norm(cr_m, axis=1)
        mean_norm = np.mean(cr_norms)
        if mean_norm < 0.01:
            orders.append(0.0)
            continue
        orders.append(np.linalg.norm(cr_m.mean(axis=0)) / mean_norm)
    orders = np.array(orders)

    # Group by geometry (same positions within tolerance)
    used = set()
    groups = []
    for i in range(len(atoms_list)):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, len(atoms_list)):
            if j in used:
                continue
            if np.linalg.norm(atoms_list[j].positions - atoms_list[i].positions) < args.tol:
                group.append(j)
                used.add(j)
        if len(group) > 1:
            groups.append(group)

    print(f"Found {len(groups)} geometry groups with multiple mag configs\n")

    for gi, group in enumerate(groups):
        group_data = [(idx, orders[idx], energies[idx]) for idx in group]
        group_data.sort(key=lambda x: x[1])

        o_min = group_data[0][1]
        o_max = group_data[-1][1]
        if o_max - o_min < 0.1:
            continue

        print(f"=== Geometry group {gi+1} ({len(group)} structures) ===")
        print(f"  {'idx':>5} {'order':>7} {'E_DFT':>12} {'E/atom':>10}")
        for idx, order, e in group_data:
            print(f"  {idx:>5} {order:>7.3f} {e:>12.4f} {e/n_atoms:>10.6f}")

        e_high_order = [e for _, o, e in group_data if o > 0.8 * o_max]
        e_low_order = [e for _, o, e in group_data if o < 0.2 * o_max + 0.2]
        if e_high_order and e_low_order:
            de = np.mean(e_high_order) - np.mean(e_low_order)
            print(f"  E(ordered) - E(disordered) = {de:.4f} eV = {de/n_atoms*1000:.1f} meV/atom")
        print()


if __name__ == "__main__":
    main()
