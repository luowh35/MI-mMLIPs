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
        e_i = self.mlp(descriptors).squeeze(-1)
        return e_i.sum()


def _to_device_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device, dtype=torch.float32)


def predict_energy_forces_maggrad(
    model: nn.Module,
    descriptor_builder: nn.Module,
    sample: Dict[str, torch.Tensor],
    device: torch.device,
    create_graph: bool = True,
    need_mag_grad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    pos = _to_device_tensor(sample["pos"], device).detach().requires_grad_(True)
    mag = _to_device_tensor(sample["mag"], device).detach().requires_grad_(True)
    cell = _to_device_tensor(sample["cell"], device)

    descriptors = descriptor_builder(pos, mag, cell)
    energy = model(descriptors)

    forces = -torch.autograd.grad(
        energy,
        pos,
        create_graph=create_graph,
        retain_graph=need_mag_grad or create_graph,
    )[0]

    if need_mag_grad:
        mag_grad = -torch.autograd.grad(
            energy,
            mag,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
    else:
        mag_grad = None

    return energy, forces, mag_grad
