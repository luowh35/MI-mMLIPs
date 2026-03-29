from __future__ import annotations

import time
from typing import Dict, Tuple

import torch
import torch.nn as nn


class LocalInvariantPotential(nn.Module):
    """E = sum_i MLP(d_i)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, depth: int = 3) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if depth < 2:
            raise ValueError("depth must be >= 2.")

        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Return per-atom energies [N]."""
        return self.mlp(descriptors).squeeze(-1)


def _to_device_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device, dtype=torch.float32)


def predict_batch(
    model: nn.Module,
    descriptor_builder: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    create_graph: bool = True,
    need_force_grad: bool = True,
    need_mag_grad: bool = True,
    profile: Dict[str, float] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Predict energies, forces, mag_grad for a flat-batched dict."""
    pos_flat = _to_device_tensor(batch["pos_flat"], device).detach().requires_grad_(True)
    mag_flat = _to_device_tensor(batch["mag_flat"], device).detach().requires_grad_(True)
    cell = _to_device_tensor(batch["cell"], device)
    if "pbc" in batch:
        pbc = batch["pbc"].to(device=device, dtype=torch.bool)
    else:
        pbc = torch.ones((cell.shape[0], 3), device=device, dtype=torch.bool)
    n_atoms = batch["n_atoms"].to(device)
    batch_idx = batch["batch_idx"].to(device)
    B = n_atoms.shape[0]

    desc_t0 = time.perf_counter()
    desc_acc_before = profile.get("descriptor_s", 0.0) if profile is not None else 0.0
    if profile is None:
        descriptors = descriptor_builder.forward_batch(
            pos_flat,
            mag_flat,
            cell,
            n_atoms,
            pbc=pbc,
        )
    else:
        try:
            descriptors = descriptor_builder.forward_batch(
                pos_flat,
                mag_flat,
                cell,
                n_atoms,
                pbc=pbc,
                profile=profile,
            )
        except TypeError:
            descriptors = descriptor_builder.forward_batch(
                pos_flat,
                mag_flat,
                cell,
                n_atoms,
                pbc=pbc,
            )
    if profile is not None:
        # Prefer descriptor-side timing; fallback to outer timing for generic builders.
        if profile.get("descriptor_s", 0.0) <= desc_acc_before:
            profile["descriptor_s"] = profile.get("descriptor_s", 0.0) + (
                time.perf_counter() - desc_t0
            )

    model_t0 = time.perf_counter()
    e_i = model(descriptors)  # [N_total]

    # per-frame energies via scatter_add
    energies = torch.zeros(B, device=device, dtype=e_i.dtype)
    energies.scatter_add_(0, batch_idx, e_i)
    if profile is not None:
        profile["model_s"] = profile.get("model_s", 0.0) + (time.perf_counter() - model_t0)

    # forces and mag_grad via autograd on the total energy
    total_energy = energies.sum()
    if need_force_grad:
        force_t0 = time.perf_counter()
        forces = -torch.autograd.grad(
            total_energy,
            pos_flat,
            create_graph=create_graph,
            retain_graph=need_mag_grad or create_graph,
        )[0]
        if profile is not None:
            profile["grad_force_s"] = profile.get("grad_force_s", 0.0) + (
                time.perf_counter() - force_t0
            )
    else:
        forces = torch.zeros_like(pos_flat)

    if need_mag_grad:
        mag_t0 = time.perf_counter()
        mag_grad = -torch.autograd.grad(
            total_energy,
            mag_flat,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        if profile is not None:
            profile["grad_mag_s"] = profile.get("grad_mag_s", 0.0) + (time.perf_counter() - mag_t0)
    else:
        mag_grad = None

    return energies, forces, mag_grad


def predict_energy_forces_maggrad(
    model: nn.Module,
    descriptor_builder: nn.Module,
    sample: Dict[str, torch.Tensor],
    device: torch.device,
    create_graph: bool = True,
    need_mag_grad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Single-frame compatibility wrapper around predict_batch."""
    batch = {
        "pos_flat": sample["pos"],
        "mag_flat": sample["mag"],
        "cell": sample["cell"].unsqueeze(0),
        "pbc": sample.get("pbc", torch.ones(3, dtype=torch.bool)).unsqueeze(0),
        "n_atoms": torch.tensor([sample["pos"].shape[0]], dtype=torch.long),
        "batch_idx": torch.zeros(sample["pos"].shape[0], dtype=torch.long),
    }
    energies, forces, mag_grad = predict_batch(
        model, descriptor_builder, batch, device,
        create_graph=create_graph, need_mag_grad=need_mag_grad,
    )
    return energies[0], forces, mag_grad


def score_magnetic_candidates(
    model: nn.Module,
    descriptor_builder: nn.Module,
    pos: torch.Tensor,
    cell: torch.Tensor,
    mag_candidates: torch.Tensor,
    device: torch.device,
    pbc: torch.Tensor | None = None,
    need_mag_grad: bool = False,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    Score many magnetic states for one fixed structure.

    Inputs:
    - pos: [N, 3]
    - cell: [3, 3]
    - pbc: [3] or None
    - mag_candidates: [K, N, 3]

    Returns:
    - energies: [K] (sum_i E_i per candidate)
    - mag_grads: [K, N, 3] or None
    """
    if mag_candidates.ndim != 3:
        raise ValueError("mag_candidates must have shape [K, N, 3].")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape [N, 3].")
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape [3, 3].")
    if mag_candidates.shape[1] != pos.shape[0] or mag_candidates.shape[2] != 3:
        raise ValueError("mag_candidates shape must match fixed structure atom count [K, N, 3].")

    pos = _to_device_tensor(pos, device)
    cell = _to_device_tensor(cell, device)
    if pbc is None:
        pbc = torch.ones(3, device=device, dtype=torch.bool)
    else:
        pbc = pbc.to(device=device, dtype=torch.bool)
    mag_candidates = _to_device_tensor(mag_candidates, device)

    k = int(mag_candidates.shape[0])
    n = int(pos.shape[0])

    pos_flat = pos.unsqueeze(0).expand(k, n, 3).reshape(k * n, 3).detach()
    mag_flat = mag_candidates.reshape(k * n, 3).detach().requires_grad_(need_mag_grad)
    cell_batch = cell.unsqueeze(0).expand(k, 3, 3)
    pbc_batch = pbc.unsqueeze(0).expand(k, 3)
    n_atoms = torch.full((k,), n, device=device, dtype=torch.long)
    batch_idx = torch.arange(k, device=device, dtype=torch.long).repeat_interleave(n)

    descriptors = descriptor_builder.forward_batch(
        pos_flat, mag_flat, cell_batch, n_atoms, pbc=pbc_batch
    )
    e_i = model(descriptors)

    energies = torch.zeros(k, device=device, dtype=e_i.dtype)
    energies.scatter_add_(0, batch_idx, e_i)

    if not need_mag_grad:
        return energies, None

    total_energy = energies.sum()
    mag_grad_flat = -torch.autograd.grad(
        total_energy,
        mag_flat,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    mag_grads = mag_grad_flat.reshape(k, n, 3)
    return energies, mag_grads
