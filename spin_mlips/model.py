from __future__ import annotations

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
    need_mag_grad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Predict energies, forces, mag_grad for a flat-batched dict."""
    pos_flat = _to_device_tensor(batch["pos_flat"], device).detach().requires_grad_(True)
    mag_flat = _to_device_tensor(batch["mag_flat"], device).detach().requires_grad_(True)
    cell = _to_device_tensor(batch["cell"], device)
    n_atoms = batch["n_atoms"].to(device)
    batch_idx = batch["batch_idx"].to(device)
    B = n_atoms.shape[0]

    descriptors = descriptor_builder.forward_batch(pos_flat, mag_flat, cell, n_atoms)
    e_i = model(descriptors)  # [N_total]

    # per-frame energies via scatter_add
    energies = torch.zeros(B, device=device, dtype=e_i.dtype)
    energies.scatter_add_(0, batch_idx, e_i)

    # forces and mag_grad via autograd on the total energy
    total_energy = energies.sum()
    forces = -torch.autograd.grad(
        total_energy,
        pos_flat,
        create_graph=create_graph,
        retain_graph=need_mag_grad or create_graph,
    )[0]

    if need_mag_grad:
        mag_grad = -torch.autograd.grad(
            total_energy,
            mag_flat,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
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
        "n_atoms": torch.tensor([sample["pos"].shape[0]], dtype=torch.long),
        "batch_idx": torch.zeros(sample["pos"].shape[0], dtype=torch.long),
    }
    energies, forces, mag_grad = predict_batch(
        model, descriptor_builder, batch, device,
        create_graph=create_graph, need_mag_grad=need_mag_grad,
    )
    return energies[0], forces, mag_grad
