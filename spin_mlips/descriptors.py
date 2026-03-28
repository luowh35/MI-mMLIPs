from __future__ import annotations

import torch
import torch.nn as nn


class InvariantDescriptorBuilder(nn.Module):
    """
    Build non-SOC O(3)xO(3) invariant local descriptors.

    Channels:
    - rho_u: [u_i, u_i^2, u_i^3]
    - rho_r
    - A_rr
    - rho_m2
    - rho_im
    - A_mm
    - A_imm (optional)
    """
    MAG_REF = 2.2

    def __init__(
        self,
        cutoff: float = 4.5,
        num_radial: int = 8,
        l_max: int = 2,
        include_imm: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if num_radial <= 0:
            raise ValueError("num_radial must be positive.")
        if l_max < 0:
            raise ValueError("l_max must be >= 0.")

        self.cutoff = float(cutoff)
        self.num_radial = int(num_radial)
        self.l_max = int(l_max)
        self.include_imm = bool(include_imm)
        self.eps = float(eps)
        self._u_ref = self.MAG_REF * self.MAG_REF  # reference u for normalization

        centers = torch.linspace(0.0, self.cutoff, steps=self.num_radial)
        if self.num_radial > 1:
            spacing = float(centers[1] - centers[0])
        else:
            spacing = self.cutoff
        beta = torch.full_like(centers, 1.0 / max(spacing * spacing, self.eps))

        self.register_buffer("centers", centers)
        self.register_buffer("beta", beta)

    @property
    def pair_angular_dim(self) -> int:
        return self.num_radial * self.num_radial * (self.l_max + 1)

    @property
    def descriptor_dim(self) -> int:
        base = (
            3  # rho_u
            + self.num_radial  # rho_r
            + self.pair_angular_dim  # A_rr
            + self.num_radial  # rho_m2
            + self.num_radial  # rho_im
            + self.pair_angular_dim  # A_mm
        )
        if self.include_imm:
            base += self.pair_angular_dim
        return base

    def cutoff_fn(self, r: torch.Tensor) -> torch.Tensor:
        x = (torch.pi * r / self.cutoff).clamp(0.0, torch.pi)
        y = 0.5 * (torch.cos(x) + 1.0)
        return y * (r < self.cutoff).to(r.dtype)

    def radial_basis(self, r: torch.Tensor) -> torch.Tensor:
        # Gaussian basis f_n(r) * f_c(r)
        diff = r.unsqueeze(-1) - self.centers.unsqueeze(0)
        radial = torch.exp(-self.beta.unsqueeze(0) * diff * diff)
        return radial * self.cutoff_fn(r).unsqueeze(-1)

    def legendre_basis(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-1.0, 1.0)
        polys = [torch.ones_like(x)]
        if self.l_max >= 1:
            polys.append(x)
        for l in range(1, self.l_max):
            p_l = polys[-1]
            p_lm1 = polys[-2]
            p_lp1 = ((2 * l + 1) * x * p_l - l * p_lm1) / (l + 1)
            polys.append(p_lp1)
        return torch.stack(polys, dim=-1)

    def minimum_image(self, disp: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        inv_cell = torch.linalg.inv(cell)
        frac = disp @ inv_cell
        # Iterative rounding for robustness with skewed cells
        for _ in range(3):
            shift = torch.round(frac)
            frac = frac - shift
            if shift.abs().max() < 0.5:
                break
        return frac @ cell

    def forward(self, pos: torch.Tensor, mag: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        n_atoms = pos.shape[0]
        eye = torch.eye(n_atoms, device=pos.device, dtype=torch.bool)

        # r_ij = R_j - R_i
        disp = pos.unsqueeze(0) - pos.unsqueeze(1)
        rij = self.minimum_image(disp, cell)
        dist = torch.linalg.norm(rij, dim=-1)
        neighbor_mask = (dist < self.cutoff) & (~eye)

        zeros_r = pos.new_zeros(self.num_radial)
        zeros_a = pos.new_zeros(self.pair_angular_dim)

        all_features = []
        for i in range(n_atoms):
            idx = torch.where(neighbor_mask[i])[0]
            m_i = mag[i]
            u_i = (m_i * m_i).sum()
            u_norm = u_i / self._u_ref
            rho_u = torch.stack([u_norm, u_norm * u_norm, u_norm * u_norm * u_norm], dim=0)

            if idx.numel() == 0:
                rho_r = zeros_r
                rho_m2 = zeros_r
                rho_im = zeros_r
                a_rr = zeros_a
                a_mm = zeros_a
                a_imm = zeros_a if self.include_imm else None
            else:
                r_vec = rij[i, idx]
                r_len = dist[i, idx]
                m_j = mag[idx]
                u_j = (m_j * m_j).sum(dim=-1)
                s_ij = (m_j * m_i.unsqueeze(0)).sum(dim=-1)

                f = self.radial_basis(r_len)
                rho_r = f.sum(dim=0)
                rho_m2 = (f * u_j.unsqueeze(-1)).sum(dim=0)
                rho_im = (f * s_ij.unsqueeze(-1)).sum(dim=0)

                if idx.numel() >= 2:
                    tri = torch.triu_indices(idx.numel(), idx.numel(), offset=1, device=pos.device)
                    j_local, k_local = tri[0], tri[1]

                    r_j = r_vec[j_local]
                    r_k = r_vec[k_local]
                    cos_theta = (r_j * r_k).sum(dim=-1) / (
                        r_len[j_local] * r_len[k_local] + self.eps
                    )
                    p_l = self.legendre_basis(cos_theta)

                    f_j = f[j_local]
                    f_k = f[k_local]
                    # Symmetrize (n,m): accumulate both (j,k) and (k,j) contributions
                    pair_jk = f_j.unsqueeze(-1) * f_k.unsqueeze(1)  # [P, N, N]
                    pair_kj = f_k.unsqueeze(-1) * f_j.unsqueeze(1)  # [P, N, N]
                    pair = pair_jk + pair_kj  # symmetric in (n, m)
                    geom_tensor = pair.unsqueeze(-1) * p_l.unsqueeze(1).unsqueeze(1)

                    a_rr = geom_tensor.sum(dim=0).reshape(-1)

                    q_jk = (m_j[j_local] * m_j[k_local]).sum(dim=-1)
                    a_mm = (geom_tensor * q_jk.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(
                        dim=0
                    ).reshape(-1)

                    if self.include_imm:
                        p_ijk = s_ij[j_local] * s_ij[k_local]
                        a_imm = (
                            geom_tensor
                            * p_ijk.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        ).sum(dim=0).reshape(-1)
                    else:
                        a_imm = None
                else:
                    a_rr = zeros_a
                    a_mm = zeros_a
                    a_imm = zeros_a if self.include_imm else None

            parts = [rho_u, rho_r, a_rr, rho_m2, rho_im, a_mm]
            if self.include_imm:
                assert a_imm is not None
                parts.append(a_imm)
            all_features.append(torch.cat(parts, dim=0))

        return torch.stack(all_features, dim=0)

    def forward_batch(
        self,
        pos_flat: torch.Tensor,
        mag_flat: torch.Tensor,
        cell: torch.Tensor,
        n_atoms: torch.Tensor,
    ) -> torch.Tensor:
        """Batch-aware forward: split flat tensors per frame, compute descriptors, cat back."""
        pos_splits = torch.split(pos_flat, n_atoms.tolist())
        mag_splits = torch.split(mag_flat, n_atoms.tolist())
        descs = []
        for pos_i, mag_i, cell_i in zip(pos_splits, mag_splits, cell):
            descs.append(self.forward(pos_i, mag_i, cell_i))
        return torch.cat(descs, dim=0)
