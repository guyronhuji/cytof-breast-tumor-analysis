
import numpy as np
import torch
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
    Fast 5th-percentile distance estimate from a random subset.
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
      - k-NN sparse (default): set k (e.g., 32–256). Ignores 'radius' unless provided.
      - Dense chunked: set k=None to compute exact dense smoothing in row chunks.

    Args:
    data: (N,F) float array
    positions: (N,2) float array
    bandwidth: float; if -1, auto-estimate.
    k: k-NN sparse mode if int, dense mode if None.
    radius: optional max distance for k-NN.
    """
    dev = _pick_device(device)
    data = data.astype(float)
    P = torch.as_tensor(positions, dtype=torch.float32, device=dev).contiguous()
    X = torch.as_tensor(data, dtype=torch.float32, device=dev).contiguous()
    N, F = X.shape

    bw = _estimate_bandwidth(P, dev) if bandwidth == -1 else float(bandwidth)
    print(f"bandwidth used: {bw}")

    if k is not None:
        # k-NN sparse mode
        k_eff = min(max(1, int(k)), N)
        out = torch.empty((N, F), dtype=torch.float32, device=dev)
        two_bw2 = 2.0 * (bw ** 2)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            Q = P[start:end]
            D = torch.cdist(Q, P)
            vals, idxs = torch.topk(D, k=k_eff, largest=False, sorted=False)
            
            if radius is not None:
                mask = (vals <= radius)
            else:
                mask = None

            W = torch.exp(- (vals * vals) / two_bw2)
            if mask is not None: W = W * mask
            
            row_sums = W.sum(dim=1, keepdim=True).clamp_min(1e-12)
            Wn = W / row_sums
            
            Xnbr = X.index_select(0, idxs.reshape(-1)).reshape(idxs.shape[0], idxs.shape[1], F)
            smoothed = (Wn.unsqueeze(-1) * Xnbr).sum(dim=1)
            out[start:end] = smoothed
        return out.cpu().numpy()

    else:
        # Dense mode
        out = torch.empty((N, F), dtype=torch.float32, device=dev)
        two_bw2 = 2.0 * (bw ** 2)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            Q = P[start:end]
            D = torch.cdist(Q, P)
            W = torch.exp(- (D * D) / two_bw2)
            row_sums = W.sum(dim=1, keepdim=True).clamp_min(1e-12)
            smoothed = (W @ X) / row_sums
            out[start:end] = smoothed
        return out.cpu().numpy()
