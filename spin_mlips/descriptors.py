from __future__ import annotations

import torch
import torch.nn as nn

from .neighbor import build_neighbor_list


class InvariantDescriptorBuilder(nn.Module):
    """
    Build non-SOC O(3)xO(3) invariant local descriptors.

    Channels:
    - rho_u: onsite longitudinal channel (power or Legendre basis)
    - rho_r
    - A_rr
    - rho_uj
    - rho_s
    - rho_s2 (optional)
    - A_jk
    - A_imm (optional)
    """
    MAG_REF = 2.2

    def __init__(
        self,
        cutoff: float = 4.5,
        num_radial: int = 8,
        l_max: int = 2,
        rho_u_basis: str = "power",
        rho_u_degree: int = 3,
        u_norm_mode: str = "fixed",
        mag_ref: float = MAG_REF,
        m_stat: float | None = None,
        u_center: float | None = None,
        u_scale: float | None = None,
        neighbor_search: str = "cell_list",
        cell_bin_size: float | None = None,
        include_s2: bool = False,
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
        if rho_u_basis not in {"power", "legendre"}:
            raise ValueError("rho_u_basis must be one of: power, legendre.")
        if rho_u_degree <= 0:
            raise ValueError("rho_u_degree must be >= 1.")
        if u_norm_mode not in {"fixed", "dataset", "dual"}:
            raise ValueError("u_norm_mode must be one of: fixed, dataset, dual.")
        if mag_ref <= 0:
            raise ValueError("mag_ref must be positive.")
        if m_stat is not None and m_stat <= 0:
            raise ValueError("m_stat must be positive when provided.")
        if u_scale is not None and u_scale <= 0:
            raise ValueError("u_scale must be positive when provided.")
        if neighbor_search not in {"cell_list", "naive"}:
            raise ValueError(
                "neighbor_search must be one of: cell_list, naive "
                "(accepted for backward-compatible checkpoint loading; "
                "descriptor now always uses lightweight O(N^2) half-list neighbor builder)."
            )
        if cell_bin_size is not None and cell_bin_size <= 0:
            raise ValueError("cell_bin_size must be positive when provided.")

        self.cutoff = float(cutoff)
        self.num_radial = int(num_radial)
        self.l_max = int(l_max)
        self.rho_u_basis = str(rho_u_basis)
        self.rho_u_degree = int(rho_u_degree)
        self.u_norm_mode = str(u_norm_mode)
        self.mag_ref = float(mag_ref)
        self.m_stat = float(m_stat) if m_stat is not None else None
        self.u_center = float(u_center) if u_center is not None else None
        self.u_scale = float(u_scale) if u_scale is not None else None
        # Keep legacy arguments for checkpoint compatibility; neighbor building is O(N^2) half-list.
        self.neighbor_search = str(neighbor_search)
        self.cell_bin_size = float(cell_bin_size) if cell_bin_size is not None else self.cutoff
        self.include_s2 = bool(include_s2)
        self.include_imm = bool(include_imm)
        self.eps = float(eps)

        self._u_ref = self._resolve_u_ref()
        if self.u_center is None:
            self._u_center = self._u_ref
        else:
            self._u_center = self.u_center
        if self.u_scale is None:
            self._u_scale = max(self._u_ref, self.eps)
        else:
            self._u_scale = self.u_scale

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
    def rho_u_dim(self) -> int:
        extra = 2 if self.u_norm_mode == "dual" else 0
        return self.rho_u_degree + extra

    @property
    def geometry_dim(self) -> int:
        return self.num_radial + self.pair_angular_dim

    @property
    def magnetic_dim(self) -> int:
        dim = self.rho_u_dim + self.num_radial + self.num_radial + self.pair_angular_dim
        if self.include_s2:
            dim += self.num_radial
        if self.include_imm:
            dim += self.pair_angular_dim
        return dim

    @property
    def descriptor_dim(self) -> int:
        base = (
            self.rho_u_dim  # rho_u
            + self.num_radial  # rho_r
            + self.pair_angular_dim  # A_rr
            + self.num_radial  # rho_uj
            + self.num_radial  # rho_s
            + self.pair_angular_dim  # A_jk
        )
        if self.include_s2:
            base += self.num_radial
        if self.include_imm:
            base += self.pair_angular_dim
        return base

    def _resolve_u_ref(self) -> float:
        if self.u_norm_mode == "fixed":
            return max(self.mag_ref * self.mag_ref, self.eps)

        if self.m_stat is not None:
            return max(self.m_stat * self.m_stat, self.eps)
        return max(self.mag_ref * self.mag_ref, self.eps)

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

    def rho_u(self, u_i: torch.Tensor) -> torch.Tensor:
        u_norm = u_i / self._u_ref

        if self.rho_u_basis == "power":
            basis_terms = [u_norm.pow(p) for p in range(1, self.rho_u_degree + 1)]
        else:
            x = ((u_i - self._u_center) / (self._u_scale + self.eps)).clamp(-1.0, 1.0)
            polys = [torch.ones_like(x), x]
            for l in range(1, self.rho_u_degree):
                p_l = polys[-1]
                p_lm1 = polys[-2]
                p_lp1 = ((2 * l + 1) * x * p_l - l * p_lm1) / (l + 1)
                polys.append(p_lp1)
            basis_terms = polys[1 : self.rho_u_degree + 1]

        if self.u_norm_mode == "dual":
            basis_terms = [u_i, u_norm] + basis_terms
        return torch.stack(basis_terms, dim=0)

    def forward_with_blocks(
        self,
        pos: torch.Tensor,
        mag: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return:
        - descriptor: [N, descriptor_dim] (legacy channel order)
        - geometry_block: [N, geometry_dim] = [rho_r, A_rr]
        - magnetic_block: [N, magnetic_dim] = [rho_u, rho_uj, rho_s, rho_s2?, A_jk, A_imm?]
        """
        n_atoms = int(pos.shape[0])
        zeros_r = pos.new_zeros(self.num_radial)
        zeros_a = pos.new_zeros(self.pair_angular_dim)

        if pbc is None:
            pbc = torch.ones(3, device=pos.device, dtype=torch.bool)
        else:
            pbc = pbc.to(device=pos.device, dtype=torch.bool)

        nb = build_neighbor_list(
            pos=pos,
            cell=cell,
            pbc=pbc,
            cutoff=self.cutoff,
            eps=self.eps,
        )
        neigh_idx = nb["neighbors"]
        edge_index = nb["edge_index"]
        edge_vec = nb["edge_vec"]
        edge_dist = nb["edge_dist"]

        neigh_vec: list[list[torch.Tensor]] = [[] for _ in range(n_atoms)]
        neigh_len: list[list[torch.Tensor]] = [[] for _ in range(n_atoms)]
        for e in range(edge_index.shape[1]):
            i = int(edge_index[0, e].item())
            j = int(edge_index[1, e].item())
            rij = edge_vec[e]
            r = edge_dist[e]
            neigh_vec[i].append(rij)
            neigh_len[i].append(r)
            neigh_vec[j].append(-rij)
            neigh_len[j].append(r)

        all_desc = []
        all_geom = []
        all_mag = []

        for i in range(n_atoms):
            idx_i = neigh_idx[i]
            m_i = mag[i]
            u_i = (m_i * m_i).sum()
            rho_u = self.rho_u(u_i)

            if not idx_i:
                rho_r = zeros_r
                rho_uj = zeros_r
                rho_s = zeros_r
                rho_s2 = zeros_r if self.include_s2 else None
                a_rr = zeros_a
                a_jk = zeros_a
                a_imm = zeros_a if self.include_imm else None
            else:
                idx = torch.tensor(idx_i, device=pos.device, dtype=torch.long)
                r_vec = torch.stack(neigh_vec[i], dim=0)
                r_len = torch.stack(neigh_len[i], dim=0)
                m_j = mag[idx]
                u_j = (m_j * m_j).sum(dim=-1)
                s_ij = (m_j * m_i.unsqueeze(0)).sum(dim=-1)
                s2_ij = s_ij * s_ij

                f = self.radial_basis(r_len)
                rho_r = f.sum(dim=0)
                rho_uj = (f * u_j.unsqueeze(-1)).sum(dim=0)
                rho_s = (f * s_ij.unsqueeze(-1)).sum(dim=0)
                rho_s2 = (f * s2_ij.unsqueeze(-1)).sum(dim=0) if self.include_s2 else None

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
                    pair_jk = f_j.unsqueeze(-1) * f_k.unsqueeze(1)
                    pair_kj = f_k.unsqueeze(-1) * f_j.unsqueeze(1)
                    pair = pair_jk + pair_kj
                    geom_tensor = pair.unsqueeze(-1) * p_l.unsqueeze(1).unsqueeze(1)

                    a_rr = geom_tensor.sum(dim=0).reshape(-1)

                    q_jk = (m_j[j_local] * m_j[k_local]).sum(dim=-1)
                    a_jk = (geom_tensor * q_jk.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(
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
                    a_jk = zeros_a
                    a_imm = zeros_a if self.include_imm else None

            desc_parts = [rho_u, rho_r, a_rr, rho_uj, rho_s]
            mag_parts = [rho_u, rho_uj, rho_s]
            if self.include_s2:
                assert rho_s2 is not None
                desc_parts.append(rho_s2)
                mag_parts.append(rho_s2)
            desc_parts.append(a_jk)
            mag_parts.append(a_jk)
            if self.include_imm:
                assert a_imm is not None
                desc_parts.append(a_imm)
                mag_parts.append(a_imm)

            all_desc.append(torch.cat(desc_parts, dim=0))
            all_geom.append(torch.cat([rho_r, a_rr], dim=0))
            all_mag.append(torch.cat(mag_parts, dim=0))

        return (
            torch.stack(all_desc, dim=0),
            torch.stack(all_geom, dim=0),
            torch.stack(all_mag, dim=0),
        )

    def forward(
        self,
        pos: torch.Tensor,
        mag: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        descriptor, _, _ = self.forward_with_blocks(pos=pos, mag=mag, cell=cell, pbc=pbc)
        return descriptor

    def forward_batch(
        self,
        pos_flat: torch.Tensor,
        mag_flat: torch.Tensor,
        cell: torch.Tensor,
        n_atoms: torch.Tensor,
        pbc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Batch-aware forward: split flat tensors per frame, compute descriptors, cat back."""
        pos_splits = torch.split(pos_flat, n_atoms.tolist())
        mag_splits = torch.split(mag_flat, n_atoms.tolist())
        if pbc is None:
            pbc = torch.ones((cell.shape[0], 3), device=cell.device, dtype=torch.bool)
        descs = []
        for pos_i, mag_i, cell_i, pbc_i in zip(pos_splits, mag_splits, cell, pbc):
            descs.append(self.forward(pos_i, mag_i, cell_i, pbc=pbc_i))
        return torch.cat(descs, dim=0)

    def forward_batch_with_blocks(
        self,
        pos_flat: torch.Tensor,
        mag_flat: torch.Tensor,
        cell: torch.Tensor,
        n_atoms: torch.Tensor,
        pbc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_splits = torch.split(pos_flat, n_atoms.tolist())
        mag_splits = torch.split(mag_flat, n_atoms.tolist())
        if pbc is None:
            pbc = torch.ones((cell.shape[0], 3), device=cell.device, dtype=torch.bool)
        descs = []
        geoms = []
        mags = []
        for pos_i, mag_i, cell_i, pbc_i in zip(pos_splits, mag_splits, cell, pbc):
            d_i, g_i, m_i = self.forward_with_blocks(pos=pos_i, mag=mag_i, cell=cell_i, pbc=pbc_i)
            descs.append(d_i)
            geoms.append(g_i)
            mags.append(m_i)
        return torch.cat(descs, dim=0), torch.cat(geoms, dim=0), torch.cat(mags, dim=0)
