"""
Training loop for MagPot.

Supports multi-target loss: energy + forces + effective magnetic field.

Output files (following GPUMD/NEP convention):
    loss.out              - per-epoch RMSE (appended every epoch)
    energy_train.out      - predicted vs reference energy per structure
    force_train.out       - predicted vs reference force per atom (6 columns)
    heff_train.out        - predicted vs reference H_eff per atom (6 columns)
    energy_test.out       - same for validation set
    force_test.out
    heff_test.out
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Optional, Dict, List
import sys
import os

from .model import MagPot, compute_forces_and_fields
from .data import MagneticDataset, collate_magnetic


class Trainer:
    """Training manager for MagPot."""

    def __init__(
        self,
        model: MagPot,
        train_dataset: MagneticDataset,
        val_dataset: Optional[MagneticDataset] = None,
        batch_size: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        lambda_e: float = 1.0,
        lambda_f: float = 10.0,
        lambda_h: float = 10.0,
        auto_weight: bool = True,
        early_stop_patience: int = 100,
        device: str = "cuda",
        output_path: str = "best.pt",
        output_dir: str = ".",
        predict_interval: int = 10,
        hparams: Optional[Dict] = None,
        species_map: Optional[Dict] = None,
        magnetic_species: Optional[List[str]] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.output_path = (
            output_path if os.path.isabs(output_path)
            else os.path.join(output_dir, output_path)
        )
        self.output_dir = output_dir
        self.predict_interval = predict_interval
        # If hparams not provided, extract from model so checkpoint is always valid
        if hparams is not None:
            self.hparams = hparams
        else:
            self.hparams = {
                "r_cutoff": model.r_cutoff,
                "basis_size": model.radial_basis.basis_size,
                "n_max": model.n_max,
                "num_species": model.num_species,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "hidden_dim_mag": model.hidden_dim_mag,
                "num_layers_mag": model.num_layers_mag,
            }
        self.species_map = species_map or {}
        self.magnetic_species = magnetic_species
        # Precompute set of magnetic species indices for fast mask construction
        if magnetic_species is not None and self.species_map:
            self._mag_indices = {self.species_map[s] for s in magnetic_species
                                 if s in self.species_map}
        else:
            self._mag_indices = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_magnetic,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_magnetic,
            )

        self.optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self._lr = lr
        self._steps_per_epoch = len(self.train_loader)
        # Scheduler will be created in train() once we know num_epochs
        self.scheduler = None

        self.lambda_e = lambda_e
        self.lambda_f = lambda_f
        self.lambda_h = lambda_h
        self.auto_weight = auto_weight
        self.early_stop_patience = early_stop_patience

        if auto_weight:
            # Compute label scales from training set for loss normalization.
            e_vals, f_vals, h_vals = [], [], []
            for s in train_dataset:
                if "energy" in s:
                    n_atoms = s["positions"].shape[0]
                    e_vals.append(s["energy"].item() / n_atoms)
                if "forces" in s:
                    f_vals.append(s["forces"].numpy().ravel())
                if "effective_field" in s:
                    heff = s["effective_field"].numpy()
                    # Only use magnetic atoms' H_eff for scale (non-magnetic are zero)
                    if self._mag_indices is not None and "species" in s:
                        sp = s["species"].numpy()
                        mask = np.zeros(len(sp), dtype=bool)
                        for idx in self._mag_indices:
                            mask |= (sp == idx)
                        heff = heff[mask]
                    h_vals.append(heff.ravel())
            self._scale_e = max(float(np.std(e_vals)), 1e-8) if e_vals else 1.0
            self._scale_h = max(float(np.std(np.concatenate(h_vals))), 1e-8) if h_vals else 1.0

            # Learnable uncertainty weights for energy and H_eff.
            # Forces always use fixed weight (lambda_f).
            self.log_var_e = nn.Parameter(torch.zeros(1, device=self.device))
            self.log_var_h = nn.Parameter(torch.zeros(1, device=self.device))
            self.optimizer.add_param_group({
                "params": [self.log_var_e, self.log_var_h],
                "lr": lr, "weight_decay": 0.0,
            })

    def _to_device(self, batch: Dict) -> Dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _build_magnetic_mask(self, species: torch.Tensor) -> Optional[torch.Tensor]:
        """Build boolean mask selecting magnetic atoms from species tensor."""
        if self._mag_indices is None:
            return None
        mask = torch.zeros(species.shape[0], dtype=torch.bool, device=species.device)
        for idx in self._mag_indices:
            mask |= (species == idx)
        return mask

    def _compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        batch = self._to_device(batch)

        has_forces = "forces" in batch
        has_heff = "effective_field" in batch
        has_energy = "energy" in batch

        magnetic_mask = self._build_magnetic_mask(batch["species"])

        energy, forces, h_eff = compute_forces_and_fields(
            self.model,
            batch["positions"],
            batch["species"],
            batch["magnetic_moments"],
            cell=batch.get("cell"),
            pbc=batch.get("pbc"),
            batch=batch.get("batch"),
            compute_heff=has_heff,
            magnetic_mask=magnetic_mask,
        )

        losses = {}
        total = torch.tensor(0.0, device=self.device)

        if has_energy:
            # Per-atom energy MSE
            num_structures = energy.shape[0]
            batch_idx = batch["batch"]
            atoms_per_struct = torch.zeros(num_structures, device=self.device)
            atoms_per_struct.scatter_add_(
                0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float)
            )
            e_per_atom_pred = energy / atoms_per_struct
            e_per_atom_ref = batch["energy"] / atoms_per_struct
            loss_e = nn.functional.mse_loss(e_per_atom_pred, e_per_atom_ref)
            losses["energy"] = loss_e
            if self.auto_weight:
                loss_e_norm = loss_e / (self._scale_e ** 2)
                total = total + self.lambda_e * (
                    0.5 * torch.exp(-self.log_var_e) * loss_e_norm + 0.5 * self.log_var_e
                )
            else:
                total = total + self.lambda_e * loss_e

        if has_forces:
            loss_f = nn.functional.mse_loss(forces, batch["forces"])
            losses["forces"] = loss_f
            total = total + self.lambda_f * loss_f

        if has_heff and h_eff is not None:
            # h_eff is [num_magnetic, 3] when mask is used, [num_atoms, 3] otherwise
            if magnetic_mask is not None:
                ref_heff = batch["effective_field"][magnetic_mask]
            else:
                ref_heff = batch["effective_field"]
            loss_h = nn.functional.mse_loss(h_eff, ref_heff)
            losses["heff"] = loss_h
            if self.auto_weight:
                loss_h_norm = loss_h / (self._scale_h ** 2)
                total = total + self.lambda_h * (
                    0.5 * torch.exp(-self.log_var_h) * loss_h_norm + 0.5 * self.log_var_h
                )
            else:
                total = total + self.lambda_h * loss_h

        losses["total"] = total
        # Raw weighted MSE (without log_var regularization) for logging
        raw_total = torch.tensor(0.0, device=self.device)
        if "energy" in losses:
            raw_total = raw_total + self.lambda_e * losses["energy"]
        if "forces" in losses:
            raw_total = raw_total + self.lambda_f * losses["forces"]
        if "heff" in losses:
            raw_total = raw_total + self.lambda_h * losses["heff"]
        losses["raw_total"] = raw_total
        return losses

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {}
        n_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            losses = self._compute_loss(batch)
            losses["total"].backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / n_batches for k, v in epoch_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        val_losses = {}
        n_batches = 0

        for batch in self.val_loader:
            # Need grad for autograd force computation
            with torch.enable_grad():
                losses = self._compute_loss(batch)
            for k, v in losses.items():
                val_losses[k] = val_losses.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / n_batches for k, v in val_losses.items()}

    def _predict_dataset(self, dataset: MagneticDataset) -> Dict[str, np.ndarray]:
        """Run model on entire dataset, collect predictions and references.

        Returns dict with keys:
            energy_pred, energy_ref: [num_structures] per-atom energy
            force_pred, force_ref: [total_atoms, 3]
            heff_pred, heff_ref: [total_atoms, 3] (if available)
        """
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.train_loader.batch_size,
                            shuffle=False, collate_fn=collate_magnetic)

        e_pred, e_ref = [], []
        f_pred, f_ref = [], []
        h_pred, h_ref = [], []

        for batch in loader:
            batch = self._to_device(batch)
            has_heff = "effective_field" in batch
            magnetic_mask = self._build_magnetic_mask(batch["species"])

            with torch.enable_grad():
                energy, forces, h_eff = compute_forces_and_fields(
                    self.model,
                    batch["positions"], batch["species"],
                    batch["magnetic_moments"],
                    cell=batch.get("cell"), pbc=batch.get("pbc"),
                    batch=batch.get("batch"), compute_heff=has_heff,
                    magnetic_mask=magnetic_mask,
                )

            batch_idx = batch["batch"]
            atoms_per_struct = torch.zeros(
                energy.shape[0], device=self.device, dtype=energy.dtype
            )
            atoms_per_struct.scatter_add_(
                0, batch_idx, torch.ones_like(batch_idx, dtype=energy.dtype)
            )

            # Per-atom energy for each structure in the batch
            e_pred.extend((energy / atoms_per_struct).detach().cpu().tolist())
            if "energy" in batch:
                e_ref.extend((batch["energy"] / atoms_per_struct).detach().cpu().tolist())

            f_pred.append(forces.detach().cpu().numpy())
            if "forces" in batch:
                f_ref.append(batch["forces"].cpu().numpy())

            if has_heff and h_eff is not None:
                h_pred.append(h_eff.detach().cpu().numpy())
                # h_eff is [num_magnetic, 3] when mask used
                if magnetic_mask is not None:
                    h_ref.append(batch["effective_field"][magnetic_mask].cpu().numpy())
                else:
                    h_ref.append(batch["effective_field"].cpu().numpy())

        result = {}
        if e_pred:
            result["energy_pred"] = np.array(e_pred)
        if e_ref:
            result["energy_ref"] = np.array(e_ref)
        if f_pred:
            result["force_pred"] = np.concatenate(f_pred, axis=0)
        if f_ref:
            result["force_ref"] = np.concatenate(f_ref, axis=0)
        if h_pred:
            result["heff_pred"] = np.concatenate(h_pred, axis=0)
        if h_ref:
            result["heff_ref"] = np.concatenate(h_ref, axis=0)
        return result

    def _write_predictions(self, preds: Dict[str, np.ndarray], suffix: str):
        """Write prediction files in GPUMD-style format.

        Args:
            preds: output from _predict_dataset
            suffix: 'train' or 'test'
        """
        d = self.output_dir

        # energy_{suffix}.out: pred ref (per structure, per-atom energy)
        if "energy_pred" in preds and "energy_ref" in preds:
            path = os.path.join(d, f"energy_{suffix}.out")
            with open(path, "w") as f:
                for p, r in zip(preds["energy_pred"], preds["energy_ref"]):
                    f.write(f"{p:g} {r:g}\n")

        # force_{suffix}.out: pred_x pred_y pred_z ref_x ref_y ref_z (per atom)
        if "force_pred" in preds and "force_ref" in preds:
            path = os.path.join(d, f"force_{suffix}.out")
            fp, fr = preds["force_pred"], preds["force_ref"]
            with open(path, "w") as f:
                for i in range(fp.shape[0]):
                    f.write(f"{fp[i,0]:g} {fp[i,1]:g} {fp[i,2]:g} "
                            f"{fr[i,0]:g} {fr[i,1]:g} {fr[i,2]:g}\n")

        # heff_{suffix}.out: pred_x pred_y pred_z ref_x ref_y ref_z (per atom)
        if "heff_pred" in preds and "heff_ref" in preds:
            path = os.path.join(d, f"heff_{suffix}.out")
            hp, hr = preds["heff_pred"], preds["heff_ref"]
            with open(path, "w") as f:
                for i in range(hp.shape[0]):
                    f.write(f"{hp[i,0]:g} {hp[i,1]:g} {hp[i,2]:g} "
                            f"{hr[i,0]:g} {hr[i,1]:g} {hr[i,2]:g}\n")

    def _compute_rmse(self, preds: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute RMSE from prediction arrays."""
        rmse = {}
        if "energy_pred" in preds and "energy_ref" in preds:
            rmse["energy"] = float(np.sqrt(np.mean(
                (preds["energy_pred"] - preds["energy_ref"]) ** 2)))
        if "force_pred" in preds and "force_ref" in preds:
            rmse["force"] = float(np.sqrt(np.mean(
                (preds["force_pred"] - preds["force_ref"]) ** 2)))
        if "heff_pred" in preds and "heff_ref" in preds:
            rmse["heff"] = float(np.sqrt(np.mean(
                (preds["heff_pred"] - preds["heff_ref"]) ** 2)))
        return rmse

    def _save_checkpoint(self, path: str, epoch: int = 0, best_val_loss: float = float("inf")):
        """Save full checkpoint with training state for resume."""
        ckpt = {
            "hparams": self.hparams,
            "species_map": self.species_map,
            "magnetic_species": self.magnetic_species,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }
        if self.auto_weight:
            ckpt["log_var_e"] = self.log_var_e.data.clone()
            ckpt["log_var_h"] = self.log_var_h.data.clone()
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and restore training state. Returns start epoch."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        # Scheduler is recreated in train() and fast-forwarded based on start_epoch
        if self.auto_weight:
            if "log_var_e" in ckpt:
                self.log_var_e.data.copy_(ckpt["log_var_e"])
            if "log_var_h" in ckpt:
                self.log_var_h.data.copy_(ckpt["log_var_h"])
        start_epoch = ckpt.get("epoch", 0)
        self._resume_best_val = ckpt.get("best_val_loss", float("inf"))
        return start_epoch

    def train(self, num_epochs: int = 500, log_interval: int = 10, start_epoch: int = 0):
        """Run full training loop.

        Args:
            num_epochs: total epochs to train
            log_interval: print console log every N epochs
            start_epoch: resume from this epoch (0 = fresh start)

        Output files:
            loss.out - appended every epoch (GPUMD-style)
            energy_train.out, force_train.out, heff_train.out - every predict_interval
            energy_test.out, force_test.out, heff_test.out - every predict_interval (if val set)
        """
        # Create OneCycleLR scheduler
        # pct_start=0.3 → 30% warmup, 70% cosine annealing
        total_steps = num_epochs * self._steps_per_epoch
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self._lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,       # initial_lr = max_lr / 25
            final_div_factor=1e4,  # final_lr = initial_lr / 10000
        )

        # On resume, fast-forward the scheduler to the correct step
        if start_epoch > 0:
            skip_steps = start_epoch * self._steps_per_epoch
            for _ in range(skip_steps):
                self.scheduler.step()

        best_val_loss = getattr(self, "_resume_best_val", float("inf"))
        no_improve_count = 0
        early_stop_patience = self.early_stop_patience
        latest_path = os.path.splitext(self.output_path)[0] + "_latest.pt"
        loss_path = os.path.join(self.output_dir, "loss.out")

        # Write loss.out header only on fresh start
        if start_epoch == 0:
            with open(loss_path, "w") as f:
                f.write("# epoch  loss_total  rmse_e_train  rmse_f_train  rmse_h_train")
                if self.val_loader is not None:
                    f.write("  rmse_e_test  rmse_f_test  rmse_h_test")
                f.write("\n")

        for epoch in range(start_epoch + 1, num_epochs + 1):
            train_losses = self.train_epoch()
            val_losses = self.validate()

            # --- loss.out: append every epoch ---
            with open(loss_path, "a") as f:
                e_rmse = train_losses.get("energy", 0.0) ** 0.5
                f_rmse = train_losses.get("forces", 0.0) ** 0.5
                h_rmse = train_losses.get("heff", 0.0) ** 0.5
                f.write(f"%-8d%-13.5f%-13.5f%-13.5f%-13.5f" % (
                    epoch, train_losses["raw_total"], e_rmse, f_rmse, h_rmse))
                if val_losses:
                    ve_rmse = val_losses.get("energy", 0.0) ** 0.5
                    vf_rmse = val_losses.get("forces", 0.0) ** 0.5
                    vh_rmse = val_losses.get("heff", 0.0) ** 0.5
                    f.write(f"%-13.5f%-13.5f%-13.5f" % (ve_rmse, vf_rmse, vh_rmse))
                f.write("\n")

            # --- Console log ---
            if epoch % log_interval == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                msg = f"Epoch {epoch:4d} | lr={lr:.2e}"
                msg += f" | train_loss={train_losses['raw_total']:.6f}"
                if "energy" in train_losses:
                    msg += f" | E_rmse={train_losses['energy']**0.5:.4e}"
                if "forces" in train_losses:
                    msg += f" | F_rmse={train_losses['forces']**0.5:.4e}"
                if "heff" in train_losses:
                    msg += f" | H_rmse={train_losses['heff']**0.5:.4e}"
                if val_losses:
                    msg += f" | val={val_losses['raw_total']:.6f}"
                if self.auto_weight:
                    w_e = self.lambda_e * 0.5 * torch.exp(-self.log_var_e).item()
                    w_h = self.lambda_h * 0.5 * torch.exp(-self.log_var_h).item()
                    msg += f" | w_E={w_e:.2f} w_F={self.lambda_f:.2f}(fix) w_H={w_h:.2f}"
                print(msg)
                sys.stdout.flush()

            # --- Prediction files: every predict_interval ---
            if epoch % self.predict_interval == 0 or epoch == 1:
                with torch.no_grad():
                    train_preds = self._predict_dataset(self.train_dataset)
                    self._write_predictions(train_preds, "train")
                    if self.val_dataset is not None:
                        val_preds = self._predict_dataset(self.val_dataset)
                        self._write_predictions(val_preds, "test")

            # Save best model (use raw_total for comparison, not uncertainty-weighted total)
            if val_losses and val_losses["raw_total"] < best_val_loss:
                best_val_loss = val_losses["raw_total"]
                no_improve_count = 0
                self._save_checkpoint(self.output_path, epoch, best_val_loss)
            elif val_losses:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                    break
            else:
                # No validation set: save best by training loss, no early stopping
                if train_losses["raw_total"] < best_val_loss:
                    best_val_loss = train_losses["raw_total"]
                    self._save_checkpoint(self.output_path, epoch, best_val_loss)
                self._save_checkpoint(latest_path, epoch, best_val_loss)
