from __future__ import annotations

import time

import torch
import torch.nn as nn

from .neighbor import build_neighbor_list, build_neighbor_list_batch, half_to_full_edges


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
        u_norm_mode: str = "dataset",
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
        if u_norm_mode not in {"dataset", "dual"}:
            raise ValueError("u_norm_mode must be one of: dataset, dual.")
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

    def rho_u_batch(self, u_all: torch.Tensor) -> torch.Tensor:
        """Vectorized rho_u over all atoms: u_all [N] -> [N, rho_u_dim]."""
        u_norm = u_all / self._u_ref

        if self.rho_u_basis == "power":
            basis_terms = [u_norm.pow(p) for p in range(1, self.rho_u_degree + 1)]
        else:
            x = ((u_all - self._u_center) / (self._u_scale + self.eps)).clamp(-1.0, 1.0)
            polys = [torch.ones_like(x), x]
            for l in range(1, self.rho_u_degree):
                p_l = polys[-1]
                p_lm1 = polys[-2]
                p_lp1 = ((2 * l + 1) * x * p_l - l * p_lm1) / (l + 1)
                polys.append(p_lp1)
            basis_terms = polys[1 : self.rho_u_degree + 1]

        if self.u_norm_mode == "dual":
            basis_terms = [u_all, u_norm] + basis_terms
        return torch.stack(basis_terms, dim=-1)

    def _build_triplet_indices(
        self,
        full_src: torch.Tensor,
        n_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build triplet indices for angular channels.

        For each center atom i, find all unique neighbor pairs (j, k) with j < k
        in the edge list order.

        Returns:
            triplet_center: [T] center atom index for each triplet
            edge_j: [T] index into full edge list for the j-th neighbor
            edge_k: [T] index into full edge list for the k-th neighbor
        """
        device = full_src.device
        counts = torch.bincount(full_src, minlength=n_atoms)  # [N]
        if not bool((counts >= 2).any().item()):
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty

        offsets = torch.zeros(n_atoms + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(counts, 0)
        starts = offsets[:-1]  # [N]

        max_neigh = int(counts.max().item())
        local_j, local_k = torch.triu_indices(max_neigh, max_neigh, offset=1, device=device)
        if local_j.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty

        # For each center i, keep local pairs (a,b) where b < count_i.
        valid = local_k.unsqueeze(0) < counts.unsqueeze(1)  # [N, Pmax]
        centers = torch.arange(n_atoms, device=device).unsqueeze(1).expand_as(valid)[valid]

        edge_j = (starts.unsqueeze(1) + local_j.unsqueeze(0))[valid]
        edge_k = (starts.unsqueeze(1) + local_k.unsqueeze(0))[valid]
        return centers, edge_j, edge_k

    def _forward_vectorized(
        self,
        pos: torch.Tensor,
        mag: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        nb: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fully vectorized descriptor computation using scatter_add and triplet indexing.
        Replaces the per-atom Python loop in forward_with_blocks.
        """
        n_atoms = int(pos.shape[0])
        device = pos.device

        edge_index = nb["edge_index"]
        edge_vec = nb["edge_vec"]
        edge_dist = nb["edge_dist"]

        # --- Expand to full (bidirectional) edge list ---
        full_src, full_dst, full_vec, full_dist = half_to_full_edges(
            edge_index, edge_vec, edge_dist
        )

        # Sort by source atom for consistent ordering (needed for triplet building)
        sort_idx = torch.argsort(full_src, stable=True)
        full_src = full_src[sort_idx]
        full_dst = full_dst[sort_idx]
        full_vec = full_vec[sort_idx]
        full_dist = full_dist[sort_idx]

        n_edges = full_src.shape[0]

        # --- Radial basis for all edges ---
        f_all = self.radial_basis(full_dist)  # [2E, num_radial]

        # --- rho_u: onsite ---
        u_all = (mag * mag).sum(dim=-1)  # [N]
        rho_u = self.rho_u_batch(u_all)  # [N, rho_u_dim]

        # --- rho_r: sum of radial basis over neighbors ---
        rho_r = torch.zeros(n_atoms, self.num_radial, device=device, dtype=pos.dtype)
        if n_edges > 0:
            rho_r.index_add_(0, full_src, f_all)

        # --- Magnetic dot products per edge ---
        m_dst = mag[full_dst]  # [2E, 3]
        m_src = mag[full_src]  # [2E, 3]
        u_dst = (m_dst * m_dst).sum(dim=-1)  # [2E]
        s_edge = (m_dst * m_src).sum(dim=-1)  # [2E] = m_j . m_i
        s2_edge = s_edge * s_edge  # [2E]

        # --- rho_uj: sum f(r) * u_j ---
        rho_uj = torch.zeros(n_atoms, self.num_radial, device=device, dtype=pos.dtype)
        if n_edges > 0:
            rho_uj.index_add_(0, full_src, f_all * u_dst.unsqueeze(-1))

        # --- rho_s: sum f(r) * s_ij ---
        rho_s = torch.zeros(n_atoms, self.num_radial, device=device, dtype=pos.dtype)
        if n_edges > 0:
            rho_s.index_add_(0, full_src, f_all * s_edge.unsqueeze(-1))

        # --- rho_s2 (optional) ---
        rho_s2 = None
        if self.include_s2:
            rho_s2 = torch.zeros(n_atoms, self.num_radial, device=device, dtype=pos.dtype)
            if n_edges > 0:
                rho_s2.index_add_(0, full_src, f_all * s2_edge.unsqueeze(-1))

        # --- Angular channels: build triplet indices ---
        R = self.num_radial
        L = self.l_max + 1
        a_rr = torch.zeros(n_atoms, R * R * L, device=device, dtype=pos.dtype)
        a_jk = torch.zeros(n_atoms, R * R * L, device=device, dtype=pos.dtype)
        a_imm = torch.zeros(n_atoms, R * R * L, device=device, dtype=pos.dtype) if self.include_imm else None

        if n_edges > 0:
            triplet_center, ej, ek = self._build_triplet_indices(full_src, n_atoms)

            if triplet_center.numel() > 0:
                # Compute cos(theta) for each triplet
                vec_j = full_vec[ej]  # [T, 3]
                vec_k = full_vec[ek]  # [T, 3]
                dist_j = full_dist[ej]  # [T]
                dist_k = full_dist[ek]  # [T]

                cos_theta = (vec_j * vec_k).sum(dim=-1) / (dist_j * dist_k + self.eps)
                p_l = self.legendre_basis(cos_theta)  # [T, L]

                f_j = f_all[ej]  # [T, R]
                f_k = f_all[ek]  # [T, R]

                # pair_flat: [T, R*R], then outer with P_l => [T, R*R*L]
                pair_flat = (
                    f_j.unsqueeze(-1) * f_k.unsqueeze(1)
                    + f_k.unsqueeze(-1) * f_j.unsqueeze(1)
                ).reshape(f_j.shape[0], -1)
                geom_flat = torch.einsum("tp,tl->tpl", pair_flat, p_l).reshape(pair_flat.shape[0], -1)

                # A_rr
                a_rr.index_add_(0, triplet_center, geom_flat)

                # A_jk: weight by m_j . m_k
                dst_j = full_dst[ej]
                dst_k = full_dst[ek]
                q_jk = (mag[dst_j] * mag[dst_k]).sum(dim=-1)  # [T]
                a_jk.index_add_(0, triplet_center, geom_flat * q_jk.unsqueeze(-1))

                # A_imm: weight by s_ij * s_ik
                if self.include_imm:
                    s_j = s_edge[ej]  # [T]
                    s_k = s_edge[ek]  # [T]
                    p_ijk = s_j * s_k
                    a_imm.index_add_(0, triplet_center, geom_flat * p_ijk.unsqueeze(-1))

        # --- Assemble outputs ---
        desc_parts = [rho_u, rho_r, a_rr, rho_uj, rho_s]
        mag_parts = [rho_u, rho_uj, rho_s]
        if self.include_s2:
            desc_parts.append(rho_s2)
            mag_parts.append(rho_s2)
        desc_parts.append(a_jk)
        mag_parts.append(a_jk)
        if self.include_imm:
            desc_parts.append(a_imm)
            mag_parts.append(a_imm)

        descriptor = torch.cat(desc_parts, dim=-1)       # [N, descriptor_dim]
        geometry_block = torch.cat([rho_r, a_rr], dim=-1) # [N, geometry_dim]
        magnetic_block = torch.cat(mag_parts, dim=-1)     # [N, magnetic_dim]

        return descriptor, geometry_block, magnetic_block

    def forward_with_blocks(
        self,
        pos: torch.Tensor,
        mag: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor | None = None,
        profile: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return:
        - descriptor: [N, descriptor_dim] (legacy channel order)
        - geometry_block: [N, geometry_dim] = [rho_r, A_rr]
        - magnetic_block: [N, magnetic_dim] = [rho_u, rho_uj, rho_s, rho_s2?, A_jk, A_imm?]
        """
        if pbc is None:
            pbc = torch.ones(3, device=pos.device, dtype=torch.bool)
        else:
            pbc = pbc.to(device=pos.device, dtype=torch.bool)

        neighbor_t0 = time.perf_counter()
        nb = build_neighbor_list(
            pos=pos,
            cell=cell,
            pbc=pbc,
            cutoff=self.cutoff,
            eps=self.eps,
        )
        if profile is not None:
            profile["neighbor_s"] = (
                profile.get("neighbor_s", 0.0) + (time.perf_counter() - neighbor_t0)
            )

        kernel_t0 = time.perf_counter()
        result = self._forward_vectorized(pos, mag, cell, pbc, nb)
        if profile is not None:
            profile["descriptor_kernel_s"] = (
                profile.get("descriptor_kernel_s", 0.0) + (time.perf_counter() - kernel_t0)
            )

        return result

    def forward(
        self,
        pos: torch.Tensor,
        mag: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor | None = None,
        profile: dict[str, float] | None = None,
    ) -> torch.Tensor:
        descriptor, _, _ = self.forward_with_blocks(
            pos=pos,
            mag=mag,
            cell=cell,
            pbc=pbc,
            profile=profile,
        )
        return descriptor

    def forward_batch(
        self,
        pos_flat: torch.Tensor,
        mag_flat: torch.Tensor,
        cell: torch.Tensor,
        n_atoms: torch.Tensor,
        pbc: torch.Tensor | None = None,
        profile: dict[str, float] | None = None,
        use_true_batch: bool = True,
    ) -> torch.Tensor:
        """
        Batch-aware forward.

        Args:
            use_true_batch: If True, use vectorized batch processing (faster on GPU).
                           If False, use per-frame loop (original behavior).
        """
        if use_true_batch and cell.shape[0] > 1:
            # Use true batch processing
            return self._forward_batch_vectorized(
                pos_flat, mag_flat, cell, n_atoms, pbc, profile
            )
        else:
            # Fall back to per-frame loop
            pos_splits = torch.split(pos_flat, n_atoms.tolist())
            mag_splits = torch.split(mag_flat, n_atoms.tolist())
            if pbc is None:
                pbc = torch.ones((cell.shape[0], 3), device=cell.device, dtype=torch.bool)
            descs = []
            for pos_i, mag_i, cell_i, pbc_i in zip(pos_splits, mag_splits, cell, pbc):
                frame_t0 = time.perf_counter()
                descs.append(self.forward(pos_i, mag_i, cell_i, pbc=pbc_i, profile=profile))
                if profile is not None:
                    profile["descriptor_s"] = (
                        profile.get("descriptor_s", 0.0) + (time.perf_counter() - frame_t0)
                    )
            return torch.cat(descs, dim=0)

    def _forward_batch_vectorized(
        self,
        pos_flat: torch.Tensor,
        mag_flat: torch.Tensor,
        cell: torch.Tensor,
        n_atoms: torch.Tensor,
        pbc: torch.Tensor | None = None,
        profile: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """
        True vectorized batch processing - all structures processed together.
        """
        if pbc is None:
            pbc = torch.ones((cell.shape[0], 3), device=cell.device, dtype=torch.bool)

        device = pos_flat.device
        B = n_atoms.shape[0]
        N_total = pos_flat.shape[0]

        # Build batch neighbor list
        neighbor_t0 = time.perf_counter()
        nb = build_neighbor_list_batch(
            pos_flat=pos_flat,
            cell=cell,
            pbc=pbc,
            n_atoms=n_atoms,
            cutoff=self.cutoff,
            eps=self.eps,
        )
        if profile is not None:
            profile["neighbor_s"] = (
                profile.get("neighbor_s", 0.0) + (time.perf_counter() - neighbor_t0)
            )

        kernel_t0 = time.perf_counter()

        edge_index = nb["edge_index"]
        edge_vec = nb["edge_vec"]
        edge_dist = nb["edge_dist"]

        # Expand to full edges
        full_src, full_dst, full_vec, full_dist = half_to_full_edges(
            edge_index, edge_vec, edge_dist
        )

        # Sort by source for consistent ordering
        sort_idx = torch.argsort(full_src, stable=True)
        full_src = full_src[sort_idx]
        full_dst = full_dst[sort_idx]
        full_vec = full_vec[sort_idx]
        full_dist = full_dist[sort_idx]

        n_edges = full_src.shape[0]

        # Radial basis
        f_all = self.radial_basis(full_dist)  # [2E, num_radial]

        # rho_u: onsite
        u_all = (mag_flat * mag_flat).sum(dim=-1)  # [N_total]
        rho_u = self.rho_u_batch(u_all)  # [N_total, rho_u_dim]

        # rho_r
        rho_r = torch.zeros(N_total, self.num_radial, device=device, dtype=pos_flat.dtype)
        if n_edges > 0:
            rho_r.index_add_(0, full_src, f_all)

        # Magnetic dot products
        m_dst = mag_flat[full_dst]  # [2E, 3]
        m_src = mag_flat[full_src]  # [2E, 3]
        u_dst = (m_dst * m_dst).sum(dim=-1)  # [2E]
        s_edge = (m_dst * m_src).sum(dim=-1)  # [2E]
        s2_edge = s_edge * s_edge  # [2E]

        # rho_uj
        rho_uj = torch.zeros(N_total, self.num_radial, device=device, dtype=pos_flat.dtype)
        if n_edges > 0:
            rho_uj.index_add_(0, full_src, f_all * u_dst.unsqueeze(-1))

        # rho_s
        rho_s = torch.zeros(N_total, self.num_radial, device=device, dtype=pos_flat.dtype)
        if n_edges > 0:
            rho_s.index_add_(0, full_src, f_all * s_edge.unsqueeze(-1))

        # rho_s2 (optional)
        rho_s2 = None
        if self.include_s2:
            rho_s2 = torch.zeros(N_total, self.num_radial, device=device, dtype=pos_flat.dtype)
            if n_edges > 0:
                rho_s2.index_add_(0, full_src, f_all * s2_edge.unsqueeze(-1))

        # Angular channels
        R = self.num_radial
        L = self.l_max + 1
        a_rr = torch.zeros(N_total, R * R * L, device=device, dtype=pos_flat.dtype)
        a_jk = torch.zeros(N_total, R * R * L, device=device, dtype=pos_flat.dtype)
        a_imm = torch.zeros(N_total, R * R * L, device=device, dtype=pos_flat.dtype) if self.include_imm else None

        if n_edges > 0:
            triplet_center, ej, ek = self._build_triplet_indices(full_src, N_total)

            if triplet_center.numel() > 0:
                vec_j = full_vec[ej]
                vec_k = full_vec[ek]
                dist_j = full_dist[ej]
                dist_k = full_dist[ek]

                cos_theta = (vec_j * vec_k).sum(dim=-1) / (dist_j * dist_k + self.eps)
                p_l = self.legendre_basis(cos_theta)

                f_j = f_all[ej]
                f_k = f_all[ek]

                pair_flat = (
                    f_j.unsqueeze(-1) * f_k.unsqueeze(1)
                    + f_k.unsqueeze(-1) * f_j.unsqueeze(1)
                ).reshape(f_j.shape[0], -1)
                geom_flat = torch.einsum("tp,tl->tpl", pair_flat, p_l).reshape(pair_flat.shape[0], -1)

                a_rr.index_add_(0, triplet_center, geom_flat)

                dst_j = full_dst[ej]
                dst_k = full_dst[ek]
                q_jk = (mag_flat[dst_j] * mag_flat[dst_k]).sum(dim=-1)
                a_jk.index_add_(0, triplet_center, geom_flat * q_jk.unsqueeze(-1))

                if self.include_imm:
                    s_j = s_edge[ej]
                    s_k = s_edge[ek]
                    p_ijk = s_j * s_k
                    a_imm.index_add_(0, triplet_center, geom_flat * p_ijk.unsqueeze(-1))

        # Assemble descriptor
        desc_parts = [rho_u, rho_r, a_rr, rho_uj, rho_s]
        if self.include_s2:
            desc_parts.append(rho_s2)
        desc_parts.append(a_jk)
        if self.include_imm:
            desc_parts.append(a_imm)

        descriptor = torch.cat(desc_parts, dim=-1)

        if profile is not None:
            profile["descriptor_kernel_s"] = (
                profile.get("descriptor_kernel_s", 0.0) + (time.perf_counter() - kernel_t0)
            )

        return descriptor

    def forward_batch_with_blocks(
        self,
        pos_flat: torch.Tensor,
        mag_flat: torch.Tensor,
        cell: torch.Tensor,
        n_atoms: torch.Tensor,
        pbc: torch.Tensor | None = None,
        profile: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_splits = torch.split(pos_flat, n_atoms.tolist())
        mag_splits = torch.split(mag_flat, n_atoms.tolist())
        if pbc is None:
            pbc = torch.ones((cell.shape[0], 3), device=cell.device, dtype=torch.bool)
        descs = []
        geoms = []
        mags = []
        for pos_i, mag_i, cell_i, pbc_i in zip(pos_splits, mag_splits, cell, pbc):
            d_i, g_i, m_i = self.forward_with_blocks(
                pos=pos_i,
                mag=mag_i,
                cell=cell_i,
                pbc=pbc_i,
                profile=profile,
            )
            descs.append(d_i)
            geoms.append(g_i)
            mags.append(m_i)
        return torch.cat(descs, dim=0), torch.cat(geoms, dim=0), torch.cat(mags, dim=0)
