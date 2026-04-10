"""
Magnetic descriptor computation for MagPot.

Implements the 6 descriptor sectors from the minimal generator framework:
1. Amplitude (u_i, u_i^2, u_i^3)
2. Isotropic exchange (m_i · M_n)
3. Single-ion anisotropy (m_i · Q_n · m_i)
4. Symmetric anisotropic exchange (m_i · Q_m · M_n)
5. DMI (A_nm · (m_i × M_m))
6. Amplitude-mixed (u_i × sectors 2,3,5)

All operations preserve the PyTorch computation graph for autograd.
"""

import torch
from torch import Tensor
from typing import Dict, Tuple


def _scatter_add(src: Tensor, index: Tensor, dim: int, dim_size: int) -> Tensor:
    """Non-inplace scatter_add that works with autograd.

    Args:
        src: source tensor
        index: index tensor (1D, will be broadcast)
        dim: dimension to scatter along (must be 0)
        dim_size: size of output along scatter dimension
    """
    # Expand index to match src shape
    idx = index
    for _ in range(src.dim() - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    # Use non-inplace scatter_add via zeros + scatter_add
    out = src.new_zeros([dim_size] + list(src.shape[1:]))
    return out.scatter_add(dim, idx, src)


def _compute_A_nm(
    i_idx: Tensor,
    r_ij: Tensor,
    phi: Tensor,
    num_atoms: int,
    n_max: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Compute the full axial vector channel A_{nm} via direct neighbor-pair sum.

    A_{nm}(i) = Σ_{j<k} φ_n(r_ij) φ_m(r_ik) (r_ij × r_ik)

    This is NOT antisymmetric in (n,m) and retains both symmetric and
    antisymmetric parts, unlike the P_n × P_m approximation.

    Implementation uses vectorized pair construction over all edges sharing
    the same center atom, fully compatible with autograd.

    Args:
        i_idx: center atom index per edge, [num_edges]
        r_ij: displacement vectors, [num_edges, 3]
        phi: radial basis values, [num_edges, n_max]
        num_atoms: total number of atoms
        n_max: number of radial basis functions
        device, dtype: tensor device and dtype

    Returns:
        A: [num_atoms, n_max, n_max, 3]
    """
    num_edges = r_ij.shape[0]

    # Build all (edge_j, edge_k) pairs that share the same center atom,
    # with edge_j < edge_k (within each atom's sorted block) to get j<k.
    #
    # Strategy: sort edges by center atom, then for each atom's block of
    # size K, enumerate K*(K-1)/2 pairs. We do this without Python loops
    # by precomputing pair indices for all atoms at once.

    sorted_order = torch.argsort(i_idx)
    i_sorted = i_idx[sorted_order]
    r_sorted = r_ij[sorted_order]        # [num_edges, 3]
    phi_sorted = phi[sorted_order]        # [num_edges, n_max]

    # Count neighbors per atom and compute offsets
    counts = torch.zeros(num_atoms, dtype=torch.long, device=device)
    counts.scatter_add_(0, i_sorted, torch.ones_like(i_sorted))
    offsets = torch.zeros(num_atoms + 1, dtype=torch.long, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    # Vectorized pair-index construction — no Python loop over atoms.
    #
    # For each atom a with K neighbors at sorted positions [start, start+K),
    # we need all upper-triangle pairs (local_j < local_k) within that block.
    # Total pairs = sum_a K*(K-1)/2.
    #
    # Strategy: for each edge e (sorted), it participates as the "j" side in
    # pairs with all later edges in the same atom's block, and as the "k" side
    # with all earlier edges.  We build this with repeat_interleave.

    # Number of pairs each edge contributes as the left element (j):
    # edge at position p within atom a's block of size K contributes (K - local_pos - 1) pairs
    # where local_pos = p - offsets[a].
    # Equivalently: for edge at sorted position p, its atom is i_sorted[p],
    # and its local position within the block is p - offsets[i_sorted[p]].
    local_pos = torch.arange(num_edges, device=device) - offsets[i_sorted]
    K_per_edge = counts[i_sorted]  # block size for each edge's atom
    pairs_as_j = (K_per_edge - local_pos - 1).clamp(min=0)  # [num_edges]

    total_pairs = pairs_as_j.sum().item()
    if total_pairs == 0:
        return r_ij.new_zeros(num_atoms, n_max, n_max, 3)

    # ej[p] = sorted edge index of the left element
    ej = torch.repeat_interleave(
        torch.arange(num_edges, device=device), pairs_as_j
    )  # [total_pairs]

    # For each ej, ek runs over the subsequent edges in the same block.
    # cumulative offset within the pairs contributed by each j-edge:
    cum = torch.zeros(num_edges + 1, dtype=torch.long, device=device)
    torch.cumsum(pairs_as_j, dim=0, out=cum[1:])

    # For pair index p, its offset within ej's run is p - cum[ej[p]]
    pair_arange = torch.arange(total_pairs, device=device)
    within_run = pair_arange - cum[ej]  # 0-based offset within each j's run
    ek = ej + within_run + 1  # ek > ej, same atom block

    # Gather pair data (these ops preserve the computation graph)
    r_j = r_sorted[ej]          # [num_pairs, 3]
    r_k = r_sorted[ek]          # [num_pairs, 3]
    phi_j = phi_sorted[ej]      # [num_pairs, n_max]
    phi_k = phi_sorted[ek]      # [num_pairs, n_max]
    center = i_sorted[ej]       # [num_pairs], center atom index

    # cross product: r_j × r_k, [num_pairs, 3]
    cross_jk = torch.cross(r_j, r_k, dim=-1)

    # φ_n(r_j) * φ_m(r_k) * (r_j × r_k)_α
    # -> [num_pairs, n_max, n_max, 3]
    contrib = (
        phi_j.unsqueeze(2).unsqueeze(3)    # [P, n, 1, 1]
        * phi_k.unsqueeze(1).unsqueeze(3)  # [P, 1, m, 1]
        * cross_jk.unsqueeze(1).unsqueeze(2)  # [P, 1, 1, 3]
    )

    # Scatter-add contributions to center atoms
    A = _scatter_add(contrib, center, 0, num_atoms)  # [num_atoms, n, m, 3]

    return A


def descriptor_radial(phi: Tensor, i_idx: Tensor, num_atoms: int) -> Tensor:
    """NEP radial descriptor (L=0): q_n = Σ_j φ_n(r_ij).

    Pure structural descriptor — no magnetic moment dependence.

    Args:
        phi: [num_edges, n_max], radial basis values
        i_idx: [num_edges], center atom index
        num_atoms: total number of atoms

    Returns:
        [num_atoms, n_max]
    """
    return _scatter_add(phi, i_idx, 0, num_atoms)


def descriptor_angular_l1(
    r_ij: Tensor, phi: Tensor, i_idx: Tensor, num_atoms: int
) -> Tensor:
    """NEP angular descriptor (L=1): q_nm = P_n · P_m.

    P_n(i) = Σ_j φ_n(r_ij) r̂_ij  (vector channel)
    q_nm = P_n · P_m              (scalar, upper triangle n<=m)

    Pure structural descriptor — no magnetic moment dependence.

    Args:
        r_ij: [num_edges, 3], displacement vectors
        phi: [num_edges, n_max], radial basis values
        i_idx: [num_edges], center atom index
        num_atoms: total number of atoms

    Returns:
        [num_atoms, n_max*(n_max+1)//2]
    """
    n_max = phi.shape[1]
    dist = r_ij.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    r_hat = r_ij / dist  # [num_edges, 3]

    # P_n(i) = Σ_j φ_n(r_ij) r̂_ij
    weighted_rhat = phi.unsqueeze(-1) * r_hat.unsqueeze(1)  # [E, n_max, 3]
    P = _scatter_add(weighted_rhat, i_idx, 0, num_atoms)    # [N, n_max, 3]

    # q_nm = P_n · P_m, upper triangle
    PP = torch.einsum("ina,ima->inm", P, P)  # [N, n_max, n_max]
    n_idx, m_idx = torch.triu_indices(n_max, n_max)
    return PP[:, n_idx, m_idx]  # [N, n_max*(n_max+1)//2]


def descriptor_angular_l2(
    r_ij: Tensor, phi: Tensor, i_idx: Tensor, num_atoms: int
) -> Tensor:
    """NEP angular descriptor (L=2): q_nm = Tr(Q_n · Q_m^T).

    Q_n(i) = Σ_j φ_n(r_ij) [r̂_ij ⊗ r̂_ij - I/3]  (traceless 2nd-rank tensor)
    q_nm = Σ_{αβ} Q_n_{αβ} Q_m_{αβ}                 (upper triangle n<=m)

    Pure structural descriptor — no magnetic moment dependence.

    Args:
        r_ij: [num_edges, 3], displacement vectors
        phi: [num_edges, n_max], radial basis values
        i_idx: [num_edges], center atom index
        num_atoms: total number of atoms

    Returns:
        [num_atoms, n_max*(n_max+1)//2]
    """
    n_max = phi.shape[1]
    device = r_ij.device
    dtype = r_ij.dtype

    dist = r_ij.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    r_hat = r_ij / dist  # [E, 3]

    # Traceless outer product: r̂⊗r̂ - I/3
    outer = r_hat.unsqueeze(-1) * r_hat.unsqueeze(-2)  # [E, 3, 3]
    eye3 = torch.eye(3, device=device, dtype=dtype)
    traceless = outer - eye3 / 3.0  # [E, 3, 3]

    # Q_n(i) = Σ_j φ_n(r_ij) * traceless_j
    weighted_Q = phi.unsqueeze(-1).unsqueeze(-1) * traceless.unsqueeze(1)  # [E, n, 3, 3]
    Q_struct = _scatter_add(weighted_Q, i_idx, 0, num_atoms)  # [N, n, 3, 3]

    # q_nm = Tr(Q_n · Q_m^T) = Σ_{αβ} Q_n_{αβ} Q_m_{αβ}
    # Flatten spatial dims for dot product
    Q_flat = Q_struct.reshape(num_atoms, n_max, 9)  # [N, n, 9]
    QQ = torch.einsum("ina,ima->inm", Q_flat, Q_flat)  # [N, n, n]
    n_idx, m_idx = torch.triu_indices(n_max, n_max)
    return QQ[:, n_idx, m_idx]  # [N, n_max*(n_max+1)//2]


def compute_covariants(
    edge_index: Tensor,
    r_ij: Tensor,
    magnetic_moments: Tensor,
    phi: Tensor,
    num_atoms: int,
) -> Dict[str, Tensor]:
    """Compute covariant intermediate quantities from the mother basis.

    Args:
        edge_index: [2, num_edges], (i, j) pairs
        r_ij: [num_edges, 3], displacement vectors
        magnetic_moments: [num_atoms, 3]
        phi: [num_edges, n_max], radial basis values
        num_atoms: total number of atoms

    Returns:
        Dictionary with covariant tensors:
        - 'u': [num_atoms], magnetic moment magnitude squared
        - 'M': [num_atoms, n_max, 3], neighbor magnetic moment channel
        - 'Q': [num_atoms, n_max, 3, 3], geometric 2nd-rank tensor
        - 'A': [num_atoms, n_max, n_max, 3], axial vector channel
              A_{nm}(i) = Σ_{j<k} φ_n(r_ij) φ_m(r_ik) (r_ij × r_ik)
    """
    i_idx = edge_index[0]  # center atoms
    j_idx = edge_index[1]  # neighbor atoms
    n_max = phi.shape[1]
    device = r_ij.device
    dtype = r_ij.dtype

    m = magnetic_moments  # [num_atoms, 3]

    # u_i = |m_i|^2
    u = (m * m).sum(dim=-1)  # [num_atoms]

    # Neighbor magnetic moments
    m_j = m[j_idx]  # [num_edges, 3]

    # Unit vectors
    d_ij = r_ij.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [num_edges, 1]
    r_hat = r_ij / d_ij  # [num_edges, 3]

    # M_n^(1)(i) = Σ_j phi_n(r_ij) * m_j
    weighted_m = phi.unsqueeze(-1) * m_j.unsqueeze(1)  # [num_edges, n_max, 3]
    M = _scatter_add(weighted_m, i_idx, 0, num_atoms)

    # Q_n^(2)(i) = Σ_j phi_n(r_ij) * [r̂_ij ⊗ r̂_ij - (1/3)I]
    outer = r_hat.unsqueeze(-1) * r_hat.unsqueeze(-2)  # [num_edges, 3, 3]
    eye3 = torch.eye(3, device=device, dtype=dtype)
    traceless = outer - eye3 / 3.0  # [num_edges, 3, 3]
    weighted_Q = phi.unsqueeze(-1).unsqueeze(-1) * traceless.unsqueeze(1)  # [E, n, 3, 3]
    Q = _scatter_add(weighted_Q, i_idx, 0, num_atoms)

    # A_{nm}(i) = Σ_{j<k} φ_n(r_ij) φ_m(r_ik) (r_ij × r_ik)
    # Direct double sum over neighbor pairs per center atom.
    A = _compute_A_nm(i_idx, r_ij, phi, num_atoms, n_max, device, dtype)

    return {"u": u, "M": M, "Q": Q, "A": A}


def descriptor_amplitude(u: Tensor) -> Tensor:
    """Sector 1: Amplitude descriptors {u_i, u_i^2, u_i^3}.

    Args:
        u: [num_atoms], magnetic moment magnitude squared

    Returns:
        [num_atoms, 3]
    """
    return torch.stack([u, u.pow(2), u.pow(3)], dim=-1)


def descriptor_isotropic_exchange(m: Tensor, M: Tensor) -> Tensor:
    """Sector 2: Isotropic exchange descriptors m_i · M_n^(1).

    Args:
        m: [num_atoms, 3], magnetic moments
        M: [num_atoms, n_max, 3], neighbor magnetic moment channel

    Returns:
        [num_atoms, n_max]
    """
    # m_i · M_n = Σ_α m_α M_{n,α}
    return (m.unsqueeze(1) * M).sum(dim=-1)


def descriptor_sia(m: Tensor, Q: Tensor) -> Tensor:
    """Sector 3: Single-ion anisotropy descriptors m_i · Q_n · m_i.

    d_n^SIA = Σ_{α,β} m_α Q_{n,αβ} m_β

    Args:
        m: [num_atoms, 3]
        Q: [num_atoms, n_max, 3, 3]

    Returns:
        [num_atoms, n_max]
    """
    # Q_n · m_i: [num_atoms, n_max, 3]
    Qm = torch.einsum("anij,aj->ani", Q, m)
    # m_i · (Q_n · m_i): [num_atoms, n_max]
    return (m.unsqueeze(1) * Qm).sum(dim=-1)


def descriptor_sae(m: Tensor, Q: Tensor, M: Tensor) -> Tensor:
    """Sector 4: Symmetric anisotropic exchange descriptors m_i · Q_m · M_n.

    d_{nm}^SAE = Σ_{α,β} (m_i)_α (Q_m)_{αβ} (M_n)_β

    Args:
        m: [num_atoms, 3]
        Q: [num_atoms, n_max, 3, 3]
        M: [num_atoms, n_max, 3]

    Returns:
        [num_atoms, n_max * n_max]
    """
    n_max = M.shape[1]
    # Q_m · M_n: need to contract over spatial index
    # m_i · Q_m: [num_atoms, n_max_m, 3]
    mQ = torch.einsum("ai,amij->amj", m, Q)  # [num_atoms, n_max, 3]
    # (m_i · Q_m) · M_n = mQ_m · M_n
    # [num_atoms, n_max_m, 3] x [num_atoms, n_max_n, 3] -> [num_atoms, n_max_m, n_max_n]
    d_sae = torch.einsum("amj,anj->amn", mQ, M)
    return d_sae.reshape(d_sae.shape[0], -1)


def descriptor_dmi(m: Tensor, M: Tensor, A: Tensor) -> Tensor:
    """Sector 5: DMI descriptors A_{nm} · (m_i × M_p).

    d_{nmp}^DMI = A_{nm} · (m_i × M_p)

    where A_{nm}(i) = Σ_{j<k} φ_n(r_ij) φ_m(r_ik) (r_ij × r_ik)
    is the full axial vector channel (not antisymmetrized in n,m).

    The second index m of A is contracted with M's index p (m=p) to keep
    the descriptor count at n_max^2 rather than n_max^3.

    Args:
        m: [num_atoms, 3]
        M: [num_atoms, n_max, 3]
        A: [num_atoms, n_max, n_max, 3], full axial vector channel

    Returns:
        [num_atoms, n_max * n_max]
    """
    # m_i × M_p: [num_atoms, n_max, 3]
    m_cross_M = torch.cross(
        m.unsqueeze(1).expand_as(M), M, dim=-1
    )

    # A_{nm} · (m_i × M_m): contract m-index of A with M, and spatial dim
    # A: [atoms, n, m, 3], m_cross_M: [atoms, m, 3] -> [atoms, n, m]
    d_dmi = torch.einsum("inmd,imd->inm", A, m_cross_M)
    return d_dmi.reshape(d_dmi.shape[0], -1)


def descriptor_amplitude_mixed(
    u: Tensor, d_iso: Tensor, d_sia: Tensor, d_dmi: Tensor
) -> Tensor:
    """Sector 6: Amplitude-mixed descriptors.

    u_i * d_iso, u_i * d_sia, u_i * d_dmi

    Args:
        u: [num_atoms]
        d_iso: [num_atoms, n_max]
        d_sia: [num_atoms, n_max]
        d_dmi: [num_atoms, n_max^2]

    Returns:
        [num_atoms, n_max + n_max + n_max^2]
    """
    u_col = u.unsqueeze(-1)
    return torch.cat([u_col * d_iso, u_col * d_sia, u_col * d_dmi], dim=-1)


def compute_all_descriptors(
    edge_index: Tensor,
    r_ij: Tensor,
    magnetic_moments: Tensor,
    phi: Tensor,
    num_atoms: int,
) -> Tuple[Tensor, Tensor]:
    """Compute structural and magnetic descriptor vectors for all atoms.

    Args:
        edge_index: [2, num_edges]
        r_ij: [num_edges, 3]
        magnetic_moments: [num_atoms, 3]
        phi: [num_edges, n_max]
        num_atoms: total number of atoms

    Returns:
        (desc_struct, desc_mag):
            desc_struct: [num_atoms, struct_dim]  — no m dependence
            desc_mag:    [num_atoms, mag_dim]     — depends on m
    """
    i_idx = edge_index[0]

    # --- Structural descriptors (no m dependence) ---
    d_radial = descriptor_radial(phi, i_idx, num_atoms)
    d_ang_l1 = descriptor_angular_l1(r_ij, phi, i_idx, num_atoms)
    d_ang_l2 = descriptor_angular_l2(r_ij, phi, i_idx, num_atoms)
    desc_struct = torch.cat([d_radial, d_ang_l1, d_ang_l2], dim=-1)

    # --- Magnetic descriptors ---
    cov = compute_covariants(edge_index, r_ij, magnetic_moments, phi, num_atoms)

    m = magnetic_moments
    u = cov["u"]
    M = cov["M"]
    Q = cov["Q"]
    A = cov["A"]

    d_amp = descriptor_amplitude(u)
    d_iso = descriptor_isotropic_exchange(m, M)
    d_sia = descriptor_sia(m, Q)
    d_sae = descriptor_sae(m, Q, M)
    d_dmi = descriptor_dmi(m, M, A)
    d_mix = descriptor_amplitude_mixed(u, d_iso, d_sia, d_dmi)
    desc_mag = torch.cat([d_amp, d_iso, d_sia, d_sae, d_dmi, d_mix], dim=-1)

    return desc_struct, desc_mag


def get_descriptor_dim(n_max: int) -> int:
    """Calculate total descriptor dimension for given n_max."""
    return get_struct_descriptor_dim(n_max) + get_mag_descriptor_dim(n_max)


def get_struct_descriptor_dim(n_max: int) -> int:
    """Structural descriptor dimension (no m dependence)."""
    d_radial = n_max
    d_ang = n_max * (n_max + 1) // 2
    return d_radial + d_ang + d_ang  # L=0 + L=1 + L=2


def get_mag_descriptor_dim(n_max: int) -> int:
    """Magnetic descriptor dimension (depends on m)."""
    d_amp = 3
    d_iso = n_max
    d_sia = n_max
    d_sae = n_max * n_max
    d_dmi = n_max * n_max
    d_mix = n_max + n_max + n_max * n_max
    return d_amp + d_iso + d_sia + d_sae + d_dmi + d_mix
