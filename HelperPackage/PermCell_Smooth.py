import numpy as np
import torch
from typing import Optional, Dict, Set, Tuple, List, Union
from tqdm.notebook import tqdm
from scipy.stats import chi2, norm
import pandas as pd

def _pick_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def _estimate_bandwidth(P, device, sample_size=2000, q=0.05):
    """
    Fast 5th-percentile distance estimate from a random subset (like your np.quantile on upper triangle).
    P: (N,2) torch.float32 on device
    """
    N = P.shape[0]
    idx = torch.randperm(N, device=device)[:min(sample_size, N)]
    Ps = P.index_select(0, idx)  # (S,2)
    # pairwise distances in chunks to limit memory
    S = Ps.shape[0]
    # Compute full SxS once (S is small), then upper triangle
    D = torch.cdist(Ps, Ps)  # float32
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)
    v = D[mask]
    # torch.quantile is fine on float32
    bw = torch.quantile(v, q).item()
    return max(bw, 1e-8)

def gaussian_smooth_all_torch(
    data: np.ndarray,
    positions: np.ndarray,
    bandwidth: float,
    *,
    k: int | None = 64,
    radius: float | None = None,
    chunk_size: int = 2048,
    device: str | None = None
) -> np.ndarray:
    """
    Gaussian smoothing with Torch + MPS (or CUDA/CPU fallback).

    Modes:
      - k-NN sparse (default): set k (e.g., 32–256). Ignores 'radius' unless provided (then also masks by radius).
      - Dense chunked: set k=None to compute exact dense smoothing in row chunks.

    Args
    ----
    data: (N,F) float array
    positions: (N,2) float array
    bandwidth: float; if -1, auto-estimate via 5th percentile of distances on a random subset
    k: int or None. If int -> k-NN sparse mode. If None -> dense mode.
    radius: optional float to additionally drop neighbors farther than radius (only in k-NN mode)
    chunk_size: number of query rows per chunk (dense or k-NN)
    device: 'mps' | 'cuda' | 'cpu' or None (auto)

    Returns
    -------
    smoothed_data: (N,F) float32 numpy array
    """
    dev = _pick_device(device)
    # Torch tensors (float32, contiguous)
    data=data.astype(float)
 #   print(data)
    P = torch.as_tensor(positions, dtype=torch.float32, device=dev).contiguous()
    X = torch.as_tensor(data, dtype=torch.float32, device=dev).contiguous()
    N, F = X.shape

    # bandwidth
    bw = _estimate_bandwidth(P, dev) if bandwidth == -1 else float(bandwidth)
    print(f"bandwidth used: {bw}")

    if k is not None:
        # --------- k-NN sparse mode (chunked over query rows) ----------
        k_eff = min(max(1, int(k)), N)  # clamp
        out = torch.empty((N, F), dtype=torch.float32, device=dev)
        two_bw2 = 2.0 * (bw ** 2)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            Q = P[start:end]                                  # (M,2)
            D = torch.cdist(Q, P)                             # (M,N)
            # smallest k distances (include self automatically)
            vals, idxs = torch.topk(D, k=k_eff, largest=False, sorted=False)  # (M,k)
            if radius is not None:
                # mask out neighbors beyond radius
                mask = (vals <= radius)
            else:
                mask = None

            # Gaussian weights (constant prefactor cancels in normalization)
            W = torch.exp(- (vals * vals) / two_bw2)          # (M,k)

            if mask is not None:
                W = W * mask

            # Normalize rows (avoid div by 0)
            row_sums = W.sum(dim=1, keepdim=True).clamp_min(1e-12)
            Wn = W / row_sums                                 # (M,k)

            # Gather neighbor features: X[idxs] -> (M,k,F)
            Xnbr = X.index_select(0, idxs.reshape(-1)).reshape(idxs.shape[0], idxs.shape[1], F)
            # Weighted sum over k neighbors: (M,k,1) * (M,k,F) -> sum_k -> (M,F)
            smoothed = (Wn.unsqueeze(-1) * Xnbr).sum(dim=1)
            out[start:end] = smoothed

        return out.cpu().numpy()

    else:
        # --------- Dense exact mode (chunked rows; no N×N kept) ----------
        out = torch.empty((N, F), dtype=torch.float32, device=dev)
        two_bw2 = 2.0 * (bw ** 2)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            Q = P[start:end]                            # (M,2)
            D = torch.cdist(Q, P)                       # (M,N)
            W = torch.exp(- (D * D) / two_bw2)          # (M,N)
            row_sums = W.sum(dim=1, keepdim=True).clamp_min(1e-12)
            # (M,N) @ (N,F) -> (M,F)
            smoothed = (W @ X) / row_sums
            out[start:end] = smoothed

        return out.cpu().numpy()


# ===========================
# Weighted set utilities (with optional normalization + sparse)
# ===========================
from collections.abc import Mapping
from scipy import sparse
from typing import Dict, Set
def _to_weight_pairs(obj):
    if obj is None:
        return []
    if isinstance(obj, Mapping) and set(obj.keys()) & {"up", "down"}:
        pairs = []
        for g in (obj.get("up", []) or []):   pairs.append((g, +1.0))
        for g in (obj.get("down", []) or []): pairs.append((g, -1.0))
        return pairs
    if isinstance(obj, Mapping):
        return [(k, float(v)) for k, v in obj.items()]
    if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], (list, tuple)) and len(obj[0]) == 2:
        return [(str(k), float(v)) for (k, v) in obj]
    if isinstance(obj, (set, list, tuple)):
        return [(str(k), 1.0) for k in obj]
    if isinstance(obj, str):
        return [(obj, 1.0)]
    raise TypeError(f"Unsupported marker-set format: {type(obj)} -> {obj}")

def normalize_weighted_marker_sets(
    var_names: np.ndarray,
    marker_sets: Dict[str, object],
    tolerate_prefixes: bool = True,
    normalize_set_weights: Optional[str] = None,  # None, "l2", "l1"
    use_sparse_W: bool = False
):
    """
    Build a weight matrix W (S x G) aligned to var_names (length G).
    If use_sparse_W=True, returns CSR matrix.
    Normalization (per set):
      - "l2": divide by L2 norm of nonzero weights
      - "l1": divide by sum of absolute weights
    Returns:
      set_names: List[str]
      W: np.ndarray or sparse.csr_matrix (S, G) float32
      sizes: np.ndarray (S,) int, number of nonzero weights in each set
    """
    G = len(var_names)
    vlow = np.char.lower(var_names.astype(str))
    set_names, sizes = [], []

    if use_sparse_W:
        data, indices, indptr = [], [], [0]

    rows_dense = []  # used only if not sparse

    for sname, payload in marker_sets.items():
        pairs = _to_weight_pairs(payload)
        # collect hits first to avoid touching big arrays repeatedly
        hits_idx, hits_w = [], []
        for g, wt in pairs:
            g_low = str(g).lower()
            hit = (vlow == g_low)
            if tolerate_prefixes:
                hit |= np.char.startswith(vlow, g_low + "(") | np.char.startswith(vlow, g_low + "-")
            idx = np.where(hit)[0]
            if idx.size:
                # apply the same weight to all matched aliases
                hits_idx.extend(idx.tolist())
                hits_w.extend([float(wt)] * idx.size)

        if not hits_idx:
            continue

        # combine duplicates (same feature matched multiple times)
        if len(hits_idx) != len(set(hits_idx)):
            df = pd.DataFrame({"i": hits_idx, "w": hits_w})
            agg = df.groupby("i", sort=False)["w"].sum()
            hits_idx = agg.index.to_numpy(dtype=int)
            hits_w = agg.to_numpy(dtype=float)

        # normalization
        if normalize_set_weights in {"l2", "l1"}:
            denom = np.linalg.norm(hits_w) if normalize_set_weights == "l2" else np.sum(np.abs(hits_w))
            denom = float(denom) if denom != 0.0 else 1.0
            hits_w = (np.asarray(hits_w, dtype=np.float64) / denom).tolist()

        set_names.append(sname)
        sizes.append(len(hits_idx))

        if use_sparse_W:
            indices.extend(hits_idx)
            data.extend(hits_w)
            indptr.append(len(indices))
        else:
            w = np.zeros(G, dtype=np.float32)
            w[np.asarray(hits_idx, dtype=int)] = np.asarray(hits_w, dtype=np.float32)
            rows_dense.append(w)

    if not set_names:
        raise ValueError("No marker sets overlap the features (after weighting).")

    sizes = np.asarray(sizes, dtype=int)
    if use_sparse_W:
        W = sparse.csr_matrix((np.asarray(data, dtype=np.float32),
                               np.asarray(indices, dtype=np.int32),
                               np.asarray(indptr, dtype=np.int32)),
                              shape=(len(set_names), G), dtype=np.float32)
    else:
        W = np.vstack(rows_dense).astype(np.float32)

    return set_names, W, sizes

from scipy import sparse
from scipy import sparse

# =======================
# Utility helpers (API kept)
# =======================
def get_layer_matrix(adata, layer: str):
    """Return matrix from AnnData layer or X (float32)."""
    X = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
    return np.asarray(X, dtype=np.float32)

def intersect_sets(var_names, marker_sets: Dict[str, Set[str]]):
    """Intersect sets with current feature space; return names, mask [S,G], sizes [S]."""
    var_names = np.asarray(var_names)
    names, masks, sizes = [], [], []
    for nm, genes in marker_sets.items():
        m = np.isin(var_names, list(genes))
        if m.any():
            names.append(nm)
            masks.append(m)
            sizes.append(int(m.sum()))
    if not names:
        raise ValueError("No marker/gene sets overlap the features (var_names).")
    Gmask = np.vstack(masks).astype(bool)
    return names, Gmask, np.asarray(sizes, int)

def bh_fdr(p):
    """Benjamini–Hochberg FDR (vectorized)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    n = len(p)
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return out



def sipsic_like_scores_v3(
    adata,
    marker_sets: Dict[str, object],
    layer: str = "scaled",
    n_perm: int = 2000,
    seed: int = 0,
    exclude_set: bool = True,
    two_sided: bool = False,
    abs_variant: bool = True,
    exact_max_combinations: int = 50_000,
    progress: bool = True,
    normalize_set_weights: Optional[str] = None,   # None / "l2" / "l1"
    use_sparse_W: bool = False,
    # speed knobs:
    prefer_permutation: bool = True,
    perm_batch: int = 1024,
):
    rng = np.random.default_rng(seed)
    X = get_layer_matrix(adata, layer).astype(np.float32)   # (N,G)
    genes = np.asarray(adata.var_names)

    set_names, W, sizes = normalize_weighted_marker_sets(
        genes, marker_sets,
        tolerate_prefixes=True,
        normalize_set_weights=normalize_set_weights,
        use_sparse_W=use_sparse_W
    )
    N, G = X.shape
    S = len(set_names)

    # Row-centering
    Xc = X - X.mean(axis=1, keepdims=True)
    if abs_variant:
        Xc_abs = np.abs(Xc)

    # Observed scores
    if use_sparse_W:
        obs = Xc @ W.T                              # (N,S)
        obs_abs = (Xc_abs @ (abs(W)).T) if abs_variant else None
    else:
        WT = W.T
        obs = Xc @ WT
        obs_abs = (Xc_abs @ np.abs(WT)) if abs_variant else None

    Z = np.zeros((N, S), np.float32)
    Pmat = np.ones((N, S), np.float64)   # rename to avoid collisions
    if abs_variant:
        Z_abs = np.zeros((N, S), np.float32)
        P_abs = np.ones((N, S), np.float64)

    def p_from_Z(z): return (2.0 * norm.sf(np.abs(z))) if two_sided else norm.sf(z)

    def update_stream(mu, m2, n_seen, batch):
        # batch: (N, B)
        B = batch.shape[1]
        if B == 0: return mu, m2, n_seen
        bmean = batch.mean(axis=1)
        bm2 = ((batch - bmean[:, None])**2).sum(axis=1)
        new_n = n_seen + B
        delta = bmean - mu
        new_mu = mu + delta * (B / new_n)
        new_m2 = m2 + bm2 + (delta**2) * n_seen * B / new_n
        return new_mu, new_m2, new_n

    it = range(S)
    if progress:
        it = tqdm(it, desc="Scoring weighted sets (perm/null)", leave=False)

    if use_sparse_W:
        W_csr = W  # (S,G) CSR
        indptr, indices, dataW = W_csr.indptr, W_csr.indices, W_csr.data

    for s in it:
        m = int(sizes[s])

        # Extract non-zero weights for set s
        if use_sparse_W:
            start, end = indptr[s], indptr[s+1]
            feat_idx = indices[start:end]                       # (m,)
            w_nonzero = dataW[start:end].astype(np.float32)     # (m,)
            mask = np.zeros(G, dtype=bool); mask[feat_idx] = True
        else:
            w_row = W[s]
            mask = (w_row != 0)
            feat_idx = np.where(mask)[0]
            w_nonzero = w_row[feat_idx].astype(np.float32)

        # Pool to draw from
        pool = np.where(~mask)[0] if exclude_set else np.arange(G, dtype=int)
        if exclude_set and len(pool) < m:
            pool = np.arange(G, dtype=int)
        n_pool = len(pool)

        # Enumeration vs permutation
        do_enum = False
        if not prefer_permutation:
            try:
                if 0 <= m <= n_pool and m <= 50:
                    n_comb = comb(n_pool, m)
                    if n_comb <= exact_max_combinations:
                        do_enum = True
            except OverflowError:
                pass

        # Streaming moments
        mu = np.zeros(N, np.float64); m2 = np.zeros(N, np.float64); n_seen = 0
        if abs_variant:
            mu_a = np.zeros(N, np.float64); m2_a = np.zeros(N, np.float64); n_seen_a = 0

        if do_enum:
            import itertools
            chunk = max(256, min(4096, perm_batch))
            itc = itertools.combinations(pool, m)
            while True:
                batch = list(itertools.islice(itc, chunk))
                if not batch: break
                idxs = np.asarray(batch, dtype=int)          # (B,m)
                B = idxs.shape[0]
                rows = idxs.ravel()
                cols = np.repeat(np.arange(B, dtype=int), m)
                data = np.tile(w_nonzero, B)
                U = sparse.csr_matrix((data, (rows, cols)), shape=(G, B), dtype=np.float32)
                null = Xc @ U
                mu, m2, n_seen = update_stream(mu, m2, n_seen, null)
                if abs_variant:
                    Ua = sparse.csr_matrix((np.abs(data), (rows, cols)), shape=(G, B), dtype=np.float32)
                    null_a = Xc_abs @ Ua
                    mu_a, m2_a, n_seen_a = update_stream(mu_a, m2_a, n_seen_a, null_a)
            sd = (np.sqrt(m2 / max(n_seen, 1)) + 1e-6)
            if abs_variant:
                sd_a = (np.sqrt(m2_a / max(n_seen_a, 1)) + 1e-6)
        else:
            # Permutation in batches
            nP = int(max(1, n_perm))  # renamed from P
            b = 0
            while b < nP:
                B = min(perm_batch, nP - b)
                idxs = np.stack([rng.choice(pool, size=m, replace=False) for _ in range(B)], axis=0)  # (B,m)
                rows = idxs.ravel()
                cols = np.repeat(np.arange(B, dtype=int), m)
                data = np.tile(w_nonzero, B)
                U = sparse.csr_matrix((data, (rows, cols)), shape=(G, B), dtype=np.float32)
                null = Xc @ U
                mu, m2, n_seen = update_stream(mu, m2, n_seen, null)
                if abs_variant:
                    Ua = sparse.csr_matrix((np.abs(data), (rows, cols)), shape=(G, B), dtype=np.float32)
                    null_a = Xc_abs @ Ua
                    mu_a, m2_a, n_seen_a = update_stream(mu_a, m2_a, n_seen_a, null_a)
                b += B
            sd = (np.sqrt(m2 / max(n_seen, 1)) + 1e-6)
            if abs_variant:
                sd_a = (np.sqrt(m2_a / max(n_seen_a, 1)) + 1e-6)

        # Z + p
        z = (obs[:, s] - mu) / sd
        Z[:, s] = z.astype(np.float32)
        Pmat[:, s] = p_from_Z(z)
        if abs_variant:
            z_a = (obs_abs[:, s] - mu_a) / sd_a
            Z_abs[:, s] = z_a.astype(np.float32)
            P_abs[:, s] = p_from_Z(z_a)

    Z_dir = (np.sign(obs) * (Z_abs if abs_variant else np.abs(Z))).astype(np.float32)

    Z_df    = pd.DataFrame(Z,     index=adata.obs_names, columns=set_names)
    P_df    = pd.DataFrame(Pmat,  index=adata.obs_names, columns=set_names)
    Zdir_df = pd.DataFrame(Z_dir, index=adata.obs_names, columns=set_names)
    if abs_variant:
        Zabs_df = pd.DataFrame(Z_abs, index=adata.obs_names, columns=set_names)
        Pabs_df = pd.DataFrame(P_abs, index=adata.obs_names, columns=set_names)
        return Z_df, P_df, Zabs_df, Pabs_df, Zdir_df
    else:
        return Z_df, P_df, Zdir_df
