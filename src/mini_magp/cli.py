"""
CLI entry point for mini-magp.

Usage:
    mini-magp train <data.xyz> [options]
    mini-magp train <data.xyz> --config params.json
    mini-magp predict <model.pt> <data.xyz> [options]
    mini-magp default-config [output.json]
"""

import argparse
import json
import os
import subprocess
import sys

import torch
import numpy as np

from .data import ensure_vector_moments


# All training parameters with defaults, used for config file generation
TRAIN_DEFAULTS = {
    "data": None,
    "val": None,
    "val_split": 0.1,
    "epochs": 500,
    "batch_size": 4,
    "lr": 1e-3,
    "r_cutoff": 6.0,
    "basis_size": 12,
    "n_max": 8,
    "num_species": 1,
    "hidden_dim": 64,
    "num_layers": 2,
    "hidden_dim_mag": None,
    "num_layers_mag": None,
    "mag_head_mode": "sector",
    "lambda_e": 1.0,
    "lambda_f": 10.0,
    "lambda_h": 10.0,
    "auto_weight": True,
    "early_stop_patience": 100,
    "device": "cuda",
    "output": "best.pt",
    "output_dir": ".",
    "predict_interval": 10,
    "species_map": None,
    "magnetic_species": None,
    "resume": None,
}


def _load_config(path: str) -> dict:
    """Load JSON config file."""
    with open(path) as f:
        cfg = json.load(f)
    # Validate keys
    unknown = set(cfg) - set(TRAIN_DEFAULTS)
    if unknown:
        print(f"Warning: unknown config keys ignored: {unknown}")
    return cfg


def _fit_scaler_batched(model, train_ds, batch_size: int):
    """Fit descriptor scaler without collating the full dataset at once."""
    from torch.utils.data import DataLoader

    from .data import collate_magnetic
    from .descriptors import compute_all_descriptors
    from .utils import build_neighbor_topology, rij_from_topology

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_magnetic,
    )

    struct_sum = torch.zeros(model.struct_dim, dtype=torch.float64)
    struct_sumsq = torch.zeros(model.struct_dim, dtype=torch.float64)
    mag_sum = torch.zeros(model.mag_dim, dtype=torch.float64)
    mag_sumsq = torch.zeros(model.mag_dim, dtype=torch.float64)
    n_atoms_total = 0
    compositions = []
    energies = []

    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            positions = batch_data["positions"]
            species = batch_data["species"]
            magnetic_moments = batch_data["magnetic_moments"]
            batch = batch_data.get("batch")
            cell = batch_data.get("cell")
            pbc = batch_data.get("pbc")

            edge_index, shifts = build_neighbor_topology(
                positions, cell, pbc, model.r_cutoff, batch
            )
            r_ij = rij_from_topology(positions, edge_index, shifts)
            i_idx, j_idx = edge_index
            dist = r_ij.norm(dim=-1)
            phi = model.radial_basis(dist, species[i_idx], species[j_idx])
            desc_struct, desc_mag = compute_all_descriptors(
                edge_index, r_ij, magnetic_moments, phi, positions.shape[0]
            )

            desc_struct64 = desc_struct.to(torch.float64)
            desc_mag64 = desc_mag.to(torch.float64)
            struct_sum += desc_struct64.sum(dim=0)
            struct_sumsq += desc_struct64.square().sum(dim=0)
            mag_sum += desc_mag64.sum(dim=0)
            mag_sumsq += desc_mag64.square().sum(dim=0)
            n_atoms_total += int(desc_struct.shape[0])

            if "energy" in batch_data:
                num_structures = int(batch.max().item()) + 1
                comp = torch.zeros(
                    num_structures, model.num_species, dtype=torch.float64
                )
                for sp in range(model.num_species):
                    mask_sp = (species == sp).to(torch.float64)
                    comp[:, sp].scatter_add_(0, batch, mask_sp)
                compositions.append(comp)
                energies.append(batch_data["energy"].to(torch.float64))

    if n_atoms_total == 0:
        raise ValueError("Cannot fit scaler on an empty training dataset.")

    struct_mean = struct_sum / n_atoms_total
    struct_var = (struct_sumsq / n_atoms_total - struct_mean.square()).clamp(min=1e-12)
    mag_mean = mag_sum / n_atoms_total
    mag_var = (mag_sumsq / n_atoms_total - mag_mean.square()).clamp(min=1e-12)

    model.struct_shift.copy_(struct_mean.to(model.struct_shift))
    model.struct_scale.copy_((1.0 / struct_var.sqrt()).to(model.struct_scale))
    model.mag_shift.copy_(mag_mean.to(model.mag_shift))
    model.mag_scale.copy_((1.0 / mag_var.sqrt()).to(model.mag_scale))

    if compositions and energies:
        A = torch.cat(compositions, dim=0)
        y = torch.cat(energies, dim=0)
        result = torch.linalg.lstsq(A, y)
        model.atomic_energy_shift.copy_(
            result.solution.to(model.atomic_energy_shift)
        )

    model._scaler_fitted = True


def main():
    parser = argparse.ArgumentParser(
        prog="mini-magp",
        description="Magnetic machine learning potential",
    )
    sub = parser.add_subparsers(dest="command")

    # --- train ---
    p_train = sub.add_parser("train", help="Train a MagPot model")
    p_train.add_argument("data", nargs="?", default=None,
                         help="Path to extxyz training file")
    p_train.add_argument("--config", default=None,
                         help="JSON config file (CLI args override config values)")
    p_train.add_argument("--val", default=None, help="Validation extxyz file")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument("--r-cutoff", type=float, default=None)
    p_train.add_argument("--basis-size", type=int, default=None)
    p_train.add_argument("--n-max", type=int, default=None)
    p_train.add_argument("--num-species", type=int, default=None)
    p_train.add_argument("--hidden-dim", type=int, default=None)
    p_train.add_argument("--num-layers", type=int, default=None)
    p_train.add_argument("--mag-head-mode", choices=["sector", "monolithic"],
                         default=None,
                         help="Magnetic energy head layout (default: sector)")
    p_train.add_argument("--lambda-e", type=float, default=None)
    p_train.add_argument("--lambda-f", type=float, default=None)
    p_train.add_argument("--lambda-h", type=float, default=None)
    p_train.add_argument("--device", default=None)
    p_train.add_argument("--output", default=None, help="Output model path")
    p_train.add_argument("--output-dir", default=None,
                         help="Directory for output files (loss.out, etc.)")
    p_train.add_argument("--predict-interval", type=int, default=None,
                         help="Write prediction files every N epochs (default: 10)")
    p_train.add_argument("--species-map", default=None,
                         help='JSON string, e.g. \'{"Fe":0,"Co":1}\'')
    p_train.add_argument("--magnetic-species", default=None,
                         help='JSON list of magnetic species, e.g. \'["Cr"]\'.'
                              ' Only these species contribute to H_eff loss.')
    p_train.add_argument("--resume", default=None,
                         help="Path to checkpoint .pt file to resume training from")
    # --- predict ---
    p_pred = sub.add_parser("predict", help="Predict on structures")
    p_pred.add_argument("model", help="Path to checkpoint .pt file")
    p_pred.add_argument("data", help="Path to extxyz input file")
    p_pred.add_argument("--device", default="cuda")
    p_pred.add_argument("--output", default="predictions.xyz", help="Output extxyz")

    # --- export ---
    p_export = sub.add_parser("export",
                              help="Export model to TorchScript for LAMMPS")
    p_export.add_argument("model", help="Path to checkpoint .pt file")
    p_export.add_argument("output", nargs="?", default=None,
                          help="Output TorchScript path (default: model_lammps.pt)")

    # --- test ---
    p_test = sub.add_parser(
        "test",
        help="Run magnetic diagnostic scans and write plots",
    )
    p_test.add_argument("model", help="Path to checkpoint .pt file")
    p_test.add_argument("structure", help="Path to extxyz or LAMMPS spin data file")
    p_test.add_argument("--index", type=int, default=0,
                        help="Structure index for extxyz input")
    p_test.add_argument("--magnetic-element", default="Cr",
                        help="Magnetic element symbol")
    p_test.add_argument("--output-dir", default="mini_magp_test",
                        help="Directory for diagnostic outputs")
    p_test.add_argument("--format", choices=["auto", "extxyz", "lammps"],
                        default="auto",
                        help="Input format for sector diagnostics")
    p_test.add_argument("--type-map", default=None,
                        help="LAMMPS type symbols, e.g. 'Cr,I'")
    p_test.add_argument("--device", default="cpu",
                        help="Device for sector diagnostics")
    p_test.add_argument("--skip-spin-scan", action="store_true",
                        help="Skip spin_scan plot generation")
    p_test.add_argument("--skip-sector", action="store_true",
                        help="Skip sector diagnostics plot generation")

    # --- default-config ---
    p_cfg = sub.add_parser("default-config",
                           help="Generate a default config JSON file")
    p_cfg.add_argument("output", nargs="?", default="config.json",
                       help="Output path (default: config.json)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        _run_train(args)
    elif args.command == "predict":
        _run_predict(args)
    elif args.command == "export":
        _run_export(args)
    elif args.command == "test":
        _run_test(args)
    elif args.command == "default-config":
        _run_default_config(args)


def _resolve_train_args(args) -> argparse.Namespace:
    """Merge: TRAIN_DEFAULTS <- config file <- CLI args."""
    merged = dict(TRAIN_DEFAULTS)

    # Layer 2: config file
    if args.config:
        cfg = _load_config(args.config)
        for k, v in cfg.items():
            if k in merged:
                merged[k] = v

    # Layer 3: CLI args override (only non-None values)
    cli_map = {
        "data": args.data,
        "val": args.val,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "r_cutoff": args.r_cutoff,
        "basis_size": args.basis_size,
        "n_max": args.n_max,
        "num_species": args.num_species,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "hidden_dim_mag": getattr(args, "hidden_dim_mag", None),
        "num_layers_mag": getattr(args, "num_layers_mag", None),
        "mag_head_mode": getattr(args, "mag_head_mode", None),
        "lambda_e": args.lambda_e,
        "lambda_f": args.lambda_f,
        "lambda_h": args.lambda_h,
        "device": args.device,
        "output": args.output,
        "output_dir": args.output_dir,
        "predict_interval": args.predict_interval,
        "species_map": args.species_map,
        "magnetic_species": args.magnetic_species,
        "resume": getattr(args, "resume", None),
    }
    for k, v in cli_map.items():
        if v is not None:
            merged[k] = v

    if merged["data"] is None:
        print("Error: training data path is required (positional arg or 'data' in config)")
        sys.exit(1)

    # Convert species_map from JSON string if needed
    if isinstance(merged["species_map"], str):
        merged["species_map"] = json.loads(merged["species_map"])

    # Convert magnetic_species from JSON string if needed
    if isinstance(merged["magnetic_species"], str):
        merged["magnetic_species"] = json.loads(merged["magnetic_species"])

    return argparse.Namespace(**merged)


def _run_train(args):
    from .model import MagPot
    from .data import MagneticDataset
    from .train import Trainer

    # Merge defaults <- config <- CLI
    args = _resolve_train_args(args)

    species_map = args.species_map

    print(f"Loading training data: {args.data}")
    train_ds = MagneticDataset.from_extxyz(args.data, species_map=species_map)

    # Auto-detect species map from training data if not provided
    if species_map is None:
        from ase.io import read as ase_read
        atoms_list = ase_read(args.data, index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        all_symbols = set()
        for atoms in atoms_list:
            all_symbols.update(atoms.get_chemical_symbols())
        species_map = {s: i for i, s in enumerate(sorted(all_symbols))}
        num_species = len(species_map)
    else:
        num_species = len(species_map)

    if args.num_species != 1:
        num_species = args.num_species

    val_ds = None
    if args.val:
        print(f"Loading validation data: {args.val}")
        val_ds = MagneticDataset.from_extxyz(args.val, species_map=species_map)
    elif args.val_split > 0 and len(train_ds) >= 10:
        # Auto-split
        from torch.utils.data import random_split
        n_val = max(1, int(len(train_ds) * args.val_split))
        n_train = len(train_ds) - n_val
        train_sub, val_sub = random_split(train_ds, [n_train, n_val])
        all_structures = train_ds.structures
        val_ds = MagneticDataset([all_structures[i] for i in val_sub.indices])
        train_ds = MagneticDataset([all_structures[i] for i in train_sub.indices])
        print(f"Auto-split: {n_train} train, {n_val} val")
    else:
        print(f"No validation set. Training on all {len(train_ds)} structures.")

    if args.resume:
        from .model import infer_mag_head_mode_from_state_dict
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(resume_ckpt, dict) and "state_dict" in resume_ckpt:
            args.mag_head_mode = resume_ckpt.get("hparams", {}).get(
                "mag_head_mode",
                infer_mag_head_mode_from_state_dict(resume_ckpt["state_dict"]),
            )

    hparams = {
        "r_cutoff": args.r_cutoff,
        "basis_size": args.basis_size,
        "n_max": args.n_max,
        "num_species": num_species,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "hidden_dim_mag": args.hidden_dim_mag,
        "num_layers_mag": args.num_layers_mag,
        "mag_head_mode": args.mag_head_mode,
    }

    model = MagPot(**hparams)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Fit descriptor scaler without building one huge all-training-set
    # neighbor graph. This matters for large cutoffs such as r_cutoff=8.
    print(f"Fitting descriptor scaler in batches of {args.batch_size}...")
    _fit_scaler_batched(model, train_ds, args.batch_size)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    magnetic_species = args.magnetic_species
    if magnetic_species:
        print(f"Magnetic species: {magnetic_species}")

    trainer = Trainer(
        model, train_ds, val_ds,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_e=args.lambda_e,
        lambda_f=args.lambda_f,
        lambda_h=args.lambda_h,
        auto_weight=args.auto_weight,
        early_stop_patience=args.early_stop_patience,
        device=args.device,
        output_path=args.output,
        output_dir=args.output_dir,
        predict_interval=args.predict_interval,
        hparams=hparams,
        species_map=species_map,
        magnetic_species=magnetic_species,
    )

    start_epoch = 0
    resume_path = getattr(args, "resume", None)
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        start_epoch = trainer.load_checkpoint(resume_path)
        print(f"Resuming from epoch {start_epoch}")

    trainer.train(num_epochs=args.epochs, start_epoch=start_epoch)
    print(f"Training complete. Model saved to {args.output}")


def _run_predict(args):
    from .model import MagPot, compute_forces_and_fields, infer_mag_head_mode_from_state_dict
    from ase.io import read as ase_read, write as ase_write

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading model: {args.model}")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "hparams" in ckpt:
        hparams = dict(ckpt["hparams"])
        species_map = ckpt["species_map"]
        magnetic_species = ckpt.get("magnetic_species")
        state_dict = ckpt["state_dict"]
        hparams.setdefault(
            "mag_head_mode",
            infer_mag_head_mode_from_state_dict(state_dict),
        )
    else:
        raise ValueError(
            "Checkpoint missing 'hparams'. Was it saved with an older version? "
            "Re-train or manually provide architecture args."
        )

    model = MagPot(**hparams)
    model.load_state_dict(state_dict)
    model.enable_scaler_if_available()
    model.to(device)
    model.eval()

    # Load structures
    print(f"Loading structures: {args.data}")
    atoms_list = ase_read(args.data, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    # Build magnetic species index set
    if magnetic_species:
        mag_indices = {species_map[s] for s in magnetic_species if s in species_map}
    else:
        mag_indices = None

    results = []
    for i, atoms in enumerate(atoms_list):
        symbols = atoms.get_chemical_symbols()
        species = np.array([species_map[s] for s in symbols], dtype=np.int64)

        positions = torch.tensor(atoms.positions, dtype=torch.float32, device=device)
        species_t = torch.tensor(species, dtype=torch.long, device=device)

        # Support both "magnetic_moments" and "magnetic_moment" keys
        raw = atoms.arrays.get("magnetic_moments") or atoms.arrays.get("magnetic_moment")
        if raw is not None:
            mag = torch.tensor(
                ensure_vector_moments(raw, len(atoms)),
                dtype=torch.float32, device=device,
            )
        else:
            mag = torch.zeros(len(atoms), 3, dtype=torch.float32, device=device)

        cell = torch.tensor(
            np.array(atoms.cell), dtype=torch.float32, device=device
        ).unsqueeze(0)
        pbc = torch.tensor(atoms.pbc, dtype=torch.bool, device=device)

        # Build magnetic mask
        if mag_indices is not None:
            magnetic_mask = torch.zeros(len(atoms), dtype=torch.bool, device=device)
            for idx in mag_indices:
                magnetic_mask |= (species_t == idx)
        else:
            magnetic_mask = None

        with torch.no_grad(), torch.enable_grad():
            energy, forces, h_eff = compute_forces_and_fields(
                model, positions, species_t, mag,
                cell=cell, pbc=pbc, compute_heff=True,
                magnetic_mask=magnetic_mask,
            )

        atoms.info["energy"] = energy.item()
        atoms.arrays["forces"] = forces.detach().cpu().numpy()
        if h_eff is not None:
            # h_eff is [num_magnetic, 3] when mask used; expand to full array
            if magnetic_mask is not None:
                full_heff = np.zeros((len(atoms), 3), dtype=np.float32)
                full_heff[magnetic_mask.cpu().numpy()] = h_eff.detach().cpu().numpy()
                atoms.arrays["effective_field"] = full_heff
            else:
                atoms.arrays["effective_field"] = h_eff.detach().cpu().numpy()
        results.append(atoms)

        if (i + 1) % 100 == 0:
            print(f"  Predicted {i + 1}/{len(atoms_list)} structures")

    ase_write(args.output, results)
    print(f"Predictions written to {args.output} ({len(results)} structures)")


def _run_export(args):
    from .export import export_model

    output = args.output
    if output is None:
        output = os.path.splitext(args.model)[0] + "_lammps.pt"

    export_model(args.model, output)


def _run_tool(script_name: str, cmd_args: list[str], cwd: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script_path = os.path.join(repo_root, "tools", script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Diagnostic tool not found: {script_path}")
    cmd = [sys.executable, script_path, *cmd_args]
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", os.path.join(cwd, ".matplotlib"))
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def _run_test(args):
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.abspath(args.model)
    structure_path = os.path.abspath(args.structure)

    print(f"Diagnostic output directory: {output_dir}", flush=True)

    if not args.skip_spin_scan:
        if args.format == "lammps" or _path_suffix(structure_path) in {".data", ".lmp"}:
            print("Skipping spin_scan: it currently expects an extxyz structure.")
        else:
            _run_tool(
                "spin_scan.py",
                [
                    model_path,
                    structure_path,
                    "--index", str(args.index),
                    "--magnetic-element", args.magnetic_element,
                ],
                cwd=output_dir,
            )
            print(f"Saved {os.path.join(output_dir, 'spin_scan.dat')}")
            print(f"Saved {os.path.join(output_dir, 'spin_scan.png')}")

    if not args.skip_sector:
        sector_csv = os.path.join(output_dir, "spin_sector_diagnostics.csv")
        sector_png = os.path.join(output_dir, "spin_sector_diagnostics.png")
        cmd_args = [
            model_path,
            structure_path,
            "--format", args.format,
            "--index", str(args.index),
            "--magnetic-element", args.magnetic_element,
            "--csv", sector_csv,
            "--plot", sector_png,
            "--device", args.device,
        ]
        if args.type_map:
            cmd_args += ["--type-map", args.type_map]
        _run_tool("diagnose_spin_sector_energies.py", cmd_args, cwd=output_dir)
        print(f"Saved {sector_csv}")
        print(f"Saved {sector_png}")

    print("mini-magp test complete.")


def _path_suffix(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _run_default_config(args):
    """Write a default config JSON file with all training parameters."""
    with open(args.output, "w") as f:
        json.dump(TRAIN_DEFAULTS, f, indent=4)
    print(f"Default config written to {args.output}")


if __name__ == "__main__":
    main()
