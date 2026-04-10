"""
Radial basis functions and cutoff functions for MagPot.

Implements Chebyshev polynomial basis with cosine cutoff,
following the NEP approach with learnable linear combination coefficients.
"""

import torch
import torch.nn as nn
from torch import Tensor


def cosine_cutoff(r: Tensor, r_cutoff: float) -> Tensor:
    """Smooth cosine cutoff function.

    f_c(r) = 0.5 * cos(pi * r / r_c) + 0.5  for r < r_c, else 0.
    """
    return torch.where(
        r < r_cutoff,
        0.5 * torch.cos(torch.pi * r / r_cutoff) + 0.5,
        torch.zeros_like(r),
    )


def chebyshev_basis(r: Tensor, r_cutoff: float, basis_size: int) -> Tensor:
    """Chebyshev polynomial basis multiplied by cutoff.

    f_n(r) = 0.5 * (T_n(x) + 1) * f_c(r)
    where x = 2 * (r / r_c)^2 - 1

    Args:
        r: distances, shape [num_edges]
        r_cutoff: cutoff radius
        basis_size: number of basis functions (n = 0, 1, ..., basis_size-1)

    Returns:
        Basis values, shape [num_edges, basis_size]
    """
    fc = cosine_cutoff(r, r_cutoff)
    x = 2.0 * (r / r_cutoff).pow(2) - 1.0

    # Chebyshev recurrence: T_0=1, T_1=x, T_{n+1}=2x*T_n - T_{n-1}
    # Build as list to avoid inplace ops on tensors in the computation graph
    polys = []
    if basis_size >= 1:
        polys.append(torch.ones_like(x))  # T_0
    if basis_size >= 2:
        polys.append(x)  # T_1
    for n in range(2, basis_size):
        polys.append(2.0 * x * polys[n - 1] - polys[n - 2])

    basis = torch.stack(polys, dim=-1)  # [num_edges, basis_size]

    # f_n = 0.5 * (T_n + 1) * f_c
    basis = 0.5 * (basis + 1.0) * fc.unsqueeze(-1)
    return basis


class RadialBasis(nn.Module):
    """Learnable radial basis: phi_n(r) = sum_k c_{nk} f_k(r).

    The raw Chebyshev basis {f_k} is fixed; the linear combination
    coefficients c are trainable, indexed by (species_i, species_j, n, k).
    """

    def __init__(
        self,
        r_cutoff: float = 6.0,
        basis_size: int = 12,
        n_max: int = 8,
        num_species: int = 1,
    ):
        super().__init__()
        self.r_cutoff = r_cutoff
        self.basis_size = basis_size
        self.n_max = n_max
        self.num_species = num_species

        # Learnable coefficients: [num_species^2, n_max, basis_size]
        self.coefficients = nn.Parameter(
            torch.randn(num_species * num_species, n_max, basis_size) * 0.1
        )

    def forward(self, r: Tensor, species_i: Tensor, species_j: Tensor) -> Tensor:
        """Compute radial basis values.

        Args:
            r: pairwise distances, shape [num_edges]
            species_i: species index of center atoms, shape [num_edges]
            species_j: species index of neighbor atoms, shape [num_edges]

        Returns:
            phi_n values, shape [num_edges, n_max]
        """
        # Raw Chebyshev basis: [num_edges, basis_size]
        f_k = chebyshev_basis(r, self.r_cutoff, self.basis_size)

        # Species pair index
        pair_idx = species_i * self.num_species + species_j  # [num_edges]

        # Gather coefficients for each edge: [num_edges, n_max, basis_size]
        c = self.coefficients[pair_idx]

        # phi_n = sum_k c_{nk} * f_k  =>  [num_edges, n_max]
        phi = torch.einsum("enk,ek->en", c, f_k)
        return phi
