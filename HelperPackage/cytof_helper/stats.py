
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    confusion_matrix,
    d2_log_loss_score,
    cohen_kappa_score
)
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import distance, ConvexHull
from scipy.stats import norm
from scipy import sparse
import umap
import torch
from math import comb
from typing import Optional, Dict, Tuple, List
from collections.abc import Mapping
import tqdm

def evaluate_model_with_cv(
    model,
    X,
    y,
    cv=5,
    n_bootstraps=1000,
    alpha=0.95,
    desired_specificity=0.85,
    plot_roc=True,
    roc_bootstrap_n=1000,
    tit=None,
    col='b',
    fname=None
):
    """
    Evaluates a classifier model using cross-validation, calculates metrics,
    and plots the ROC curve with confidence bands.

    Parameters:
    - model: scikit-learn compatible classifier with predict_proba method.
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: True binary labels (numpy array or pandas Series).
    - cv: Number of cross-validation folds (default=5).
    - n_bootstraps: Number of bootstraps for confidence intervals (default=1000).
    - alpha: Confidence level for confidence intervals (default=0.95).
    - desired_specificity: Desired specificity to find optimal threshold (default=0.85).
    - plot_roc: Boolean flag to plot ROC curve (default=True).
    - roc_bootstrap_n: Number of bootstraps for ROC confidence band (default=1000).

    Returns:
    - metrics_dict: Dictionary containing metrics and their confidence intervals.
    """

    y_true_all = []
    y_scores_all = []

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
        else:
            X_train = X[train_index]
            X_test = X[test_index]

        if isinstance(y, pd.Series):
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
        else:
            y_train = y[train_index]
            y_test = y[test_index]

        model.fit(X_train, y_train)
        y_scores = model.predict_proba(X_test)[:, 1]

        y_true_all.extend(y_test)
        y_scores_all.extend(y_scores)

    y_true_all = np.array(y_true_all)
    y_scores_all = np.array(y_scores_all)

    auc = roc_auc_score(y_true_all, y_scores_all)

    fpr, tpr, thresholds = roc_curve(y_true_all, y_scores_all)
    desired_fpr = 1 - desired_specificity
    idx = np.argmin(np.abs(fpr - desired_fpr))
    optimal_threshold = thresholds[idx]

    y_pred = (y_scores_all >= optimal_threshold).astype(int)

    sensitivity = recall_score(y_true_all, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred).ravel()
    specificity = tn / (tn + fp)
    precision = precision_score(y_true_all, y_pred)
    d2logloss = d2_log_loss_score(y_true_all, y_scores_all)
    kappa = cohen_kappa_score(y_true_all, y_pred)

    metrics_dict = {
        'AUC': auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Optimal Threshold': optimal_threshold,
        'D2 log loss': d2logloss,
        'Kappa': kappa
    }

    def bootstrap_metric(y_true, y_scores, metric_func):
        bootstrapped_scores = []
        rng = np.random.RandomState(seed=42)
        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = metric_func(y_true[indices], y_scores[indices])
            bootstrapped_scores.append(score)
        sorted_scores = np.sort(bootstrapped_scores)
        lower = np.percentile(sorted_scores, ((1 - alpha) / 2) * 100)
        upper = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
        return lower, upper

    def auc_metric(y_true, y_scores):
        return roc_auc_score(y_true, y_scores)

    def sensitivity_metric(y_true, y_scores):
        y_pred = (y_scores >= optimal_threshold).astype(int)
        return recall_score(y_true, y_pred)

    def specificity_metric(y_true, y_scores):
        y_pred = (y_scores >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)

    def precision_metric(y_true, y_scores):
        y_pred = (y_scores >= optimal_threshold).astype(int)
        return precision_score(y_true, y_pred)

    def d2ll_metric(y_true, y_scores):
        return d2_log_loss_score(y_true, y_scores)

    def kappa_metric(y_true, y_scores):
        y_pred = (y_scores >= optimal_threshold).astype(int)
        return cohen_kappa_score(y_true, y_pred)

    auc_lower, auc_upper = bootstrap_metric(y_true_all, y_scores_all, auc_metric)
    sens_lower, sens_upper = bootstrap_metric(y_true_all, y_scores_all, sensitivity_metric)
    spec_lower, spec_upper = bootstrap_metric(y_true_all, y_scores_all, specificity_metric)
    prec_lower, prec_upper = bootstrap_metric(y_true_all, y_scores_all, precision_metric)
    kappa_lower, kappa_upper = bootstrap_metric(y_true_all, y_scores_all, kappa_metric)
    d2_lower, d2_upper = bootstrap_metric(y_true_all, y_scores_all, d2ll_metric)

    metrics_dict['AUC CI'] = (auc_lower, auc_upper)
    metrics_dict['Sensitivity CI'] = (sens_lower, sens_upper)
    metrics_dict['Specificity CI'] = (spec_lower, spec_upper)
    metrics_dict['Precision CI'] = (prec_lower, prec_upper)
    metrics_dict['D2 Log Loss CI'] = (d2_lower, d2_upper)
    metrics_dict['Kappa CI'] = (kappa_lower, kappa_upper)

    if plot_roc:
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        rng = np.random.RandomState(seed=42)

        for i in range(roc_bootstrap_n):
            indices = rng.randint(0, len(y_true_all), len(y_true_all))
            if len(np.unique(y_true_all[indices])) < 2:
                continue
            y_true_boot = y_true_all[indices]
            y_scores_boot = y_scores_all[indices]
            fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_scores_boot)
            auc_boot = roc_auc_score(y_true_boot, y_scores_boot)
            aucs.append(auc_boot)
            tpr_interp = np.interp(mean_fpr, fpr_boot, tpr_boot)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)

        tprs = np.array(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)

        tpr_upper = np.minimum(mean_tpr + (std_tpr * 1.96), 1)
        tpr_lower = mean_tpr - (std_tpr * 1.96)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color=col, label=f'ROC Curve (AUC = {auc:.3f})', lw=2)
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=col, alpha=0.2, label='95% Confidence Band')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        
        plt_tit = 'ROC Curve'
        if tit:
            plt_tit = f'ROC Curve - {tit}'
        plt.title(plt_tit)
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.xlim(0, 1)
        plt.ylim(0, 1.01)
        if fname:
            plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.show()

    return metrics_dict


def test_hetero_mult(DB, UMAPMRK, min_dist=0.001, n_neighbors=60, fname=None, rstate=42):
    """
    Test heterogeneity of multiple lines using UMAP and Convex Hull area.

    Returns list of tuples: (Line, Local Density, Global Dist Quantile, Convex Hull Volume)
    """
    out = []
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='euclidean',
        random_state=rstate,
        verbose=False
    )
    X_2d = fit.fit_transform(DB[UMAPMRK])
    lines = DB.Line.unique()

    for line in lines:
        m = DB['Line'] == line
        
        xmax = X_2d[:, 0].max()
        xmin = X_2d[:, 0].min()
        ymax = X_2d[:, 1].max()
        ymin = X_2d[:, 1].min()
        mx = np.max([xmax, ymax])
        mn = np.min([xmin, ymin])

        b = np.linspace(round(mn, 0) - 1, round(mx, 0) + 1, 50)
        A, _, _ = np.histogram2d(X_2d[m, 0], X_2d[m, 1], bins=b)
        DD = distance.cdist(X_2d[m], X_2d[m]).flatten()
        hull1 = ConvexHull(X_2d[m])

        out.append((line, (A > 0).sum(), np.round(np.quantile(DD, 0.95), 2), hull1.volume))
    return out


def average_knn_distance(x, k: int, batch_size: int = None):
    """
    Compute for each point in x the average distance to its k nearest neighbors.
    
    Args:
      x           : Tensor of shape (N, 2)
      k           : number of nearest neighbors
      batch_size  : if None, does full N×N; otherwise splits into batches of rows
    Returns:
      avg_dists   : Tensor of shape (N,) with the mean distance to the k neighbors
    """
    x = torch.tensor(x, dtype=torch.float32)
    device = torch.device('cuda') if torch.cuda.is_available() else (
             torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
    
    x = x.to(device)

    N = x.size(0)
    out = []

    if batch_size is None:
        D = torch.cdist(x, x)
        vals, _ = torch.topk(D, k=k+1, largest=False)
        avg = vals[:, 1:].mean(dim=1)
        return avg.cpu().numpy()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Xi = x[start:end]
        Di = torch.cdist(Xi, x)
        vals, _ = torch.topk(Di, k=k+1, largest=False)
        avg_i = vals[:, 1:].mean(dim=1)
        out.append(avg_i.cpu())

    out = torch.cat(out, dim=0)
    return out.numpy()

# Residual functions from CyTOFHelper
def residual(params, x, data):
    model = params['a'] * x + params['b']
    return data - model

def residual2(params, x, data):
    model = params['a'] * x + params['b'] + params['c'] * x * x
    return data - model

def two_samp_z(X1, X2):
    """
    Calculate Z-score for difference between two samples.
    """
    from numpy import sqrt, abs, round
    from scipy.stats import norm
    
    mudiff = np.mean(X1) - np.mean(X2)
    sd_pooled = np.sqrt(np.std(X1)**2 + np.std(X2)**2)
    z = mudiff / sd_pooled
    pval = 2 * (1 - norm.cdf(abs(z)))
    return round(z, 3), round(pval, 4)

def statistic(dframe):
    return dframe.iloc[:, 0].mean() - dframe.iloc[:, 1].mean()

def predict_th(model, X, threshold=0.8, default_value=-1):
    """
    Predicts using a classifier (like XGBoost) with a confidence threshold.
    
    Parameters:
        model : trained classifier with predict_proba
        X : array-like, shape (n_samples, n_features)
            Input data.
        threshold : float
            Probability threshold for high confidence.
        default_value : int or float
            Value assigned to low-confidence predictions.
            
    Returns:
        y_pred : array, shape (n_samples,)
            Predictions with default_value where confidence < threshold.
    """
    proba = model.predict_proba(X)
    max_proba = np.max(proba, axis=1)
    preds = np.argmax(proba, axis=1)
    
    # Where confidence is low, set to default
    preds = np.where(max_proba >= threshold, preds, default_value)
    return preds

# ---------------------------------------------------------------------
# Permutation-based Cell Scoring (formerly sipsic_like_scores_v3)
# ---------------------------------------------------------------------

def _to_weight_pairs(obj):
    if obj is None: return []
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
    normalize_set_weights: Optional[str] = None,
    use_sparse_W: bool = False
):
    G = len(var_names)
    vlow = np.char.lower(var_names.astype(str))
    set_names, sizes = [], []

    if use_sparse_W:
        data, indices, indptr = [], [], [0]
    rows_dense = []

    for sname, payload in marker_sets.items():
        pairs = _to_weight_pairs(payload)
        hits_idx, hits_w = [], []
        for g, wt in pairs:
            g_low = str(g).lower()
            hit = (vlow == g_low)
            if tolerate_prefixes:
                hit |= np.char.startswith(vlow, g_low + "(") | np.char.startswith(vlow, g_low + "-")
            idx = np.where(hit)[0]
            if idx.size:
                hits_idx.extend(idx.tolist())
                hits_w.extend([float(wt)] * idx.size)

        if not hits_idx: continue

        if len(hits_idx) != len(set(hits_idx)):
            df = pd.DataFrame({"i": hits_idx, "w": hits_w})
            agg = df.groupby("i", sort=False)["w"].sum()
            hits_idx, hits_w = agg.index.to_numpy(dtype=int), agg.to_numpy(dtype=float)

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

    if not set_names: raise ValueError("No marker sets overlap the features.")

    sizes = np.asarray(sizes, dtype=int)
    if use_sparse_W:
        W = sparse.csr_matrix((np.asarray(data, dtype=np.float32),
                               np.asarray(indices, dtype=np.int32),
                               np.asarray(indptr, dtype=np.int32)),
                               shape=(len(set_names), G), dtype=np.float32)
    else:
        W = np.vstack(rows_dense).astype(np.float32)

    return set_names, W, sizes

def get_layer_matrix(adata, layer: str):
    X = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
    return np.asarray(X, dtype=np.float32)

def perm_cell(
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
    normalize_set_weights: Optional[str] = None,
    use_sparse_W: bool = False,
    prefer_permutation: bool = True,
    perm_batch: int = 1024,
):
    """
    Compute permutation-based scores for marker sets relative to random background.
    Previously known as sipsic_like_scores_v3.
    """
    rng = np.random.default_rng(seed)
    X = get_layer_matrix(adata, layer).astype(np.float32)
    genes = np.asarray(adata.var_names)

    set_names, W, sizes = normalize_weighted_marker_sets(
        genes, marker_sets, tolerate_prefixes=True,
        normalize_set_weights=normalize_set_weights, use_sparse_W=use_sparse_W
    )
    N, G = X.shape
    S = len(set_names)

    Xc = X - X.mean(axis=1, keepdims=True)
    Xc_abs = np.abs(Xc) if abs_variant else None

    if use_sparse_W:
        obs = Xc @ W.T
        obs_abs = (Xc_abs @ (abs(W)).T) if abs_variant else None
    else:
        WT = W.T
        obs = Xc @ WT
        obs_abs = (Xc_abs @ np.abs(WT)) if abs_variant else None

    Z = np.zeros((N, S), np.float32)
    Pmat = np.ones((N, S), np.float64)
    Z_abs, P_abs = (np.zeros((N, S), np.float32), np.ones((N, S), np.float64)) if abs_variant else (None, None)

    def p_from_Z(z): return (2.0 * norm.sf(np.abs(z))) if two_sided else norm.sf(z)

    def update_stream(mu, m2, n_seen, batch):
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
    if progress: from tqdm import tqdm; it = tqdm(it, desc="Scoring weighted sets")

    if use_sparse_W:
        W_csr = W
        indptr, indices, dataW = W_csr.indptr, W_csr.indices, W_csr.data

    for s in it:
        m = int(sizes[s])
        
        if use_sparse_W:
            start, end = indptr[s], indptr[s+1]
            feat_idx = indices[start:end]
            w_nonzero = dataW[start:end].astype(np.float32)
            mask = np.zeros(G, dtype=bool); mask[feat_idx] = True
        else:
            w_row = W[s]
            mask = (w_row != 0)
            feat_idx = np.where(mask)[0]
            w_nonzero = w_row[feat_idx].astype(np.float32)

        pool = np.where(~mask)[0] if exclude_set else np.arange(G, dtype=int)
        if exclude_set and len(pool) < m: pool = np.arange(G, dtype=int)
        n_pool = len(pool)

        do_enum = False
        if not prefer_permutation and 0 <= m <= n_pool and m <= 50:
            try:
                if comb(n_pool, m) <= exact_max_combinations: do_enum = True
            except OverflowError: pass

        mu, m2, n_seen = np.zeros(N), np.zeros(N), 0
        mu_a, m2_a, n_seen_a = (np.zeros(N), np.zeros(N), 0) if abs_variant else (None, None, None)

        if do_enum:
            import itertools
            itc = itertools.combinations(pool, m)
            while True:
                batch = list(itertools.islice(itc, max(256, min(4096, perm_batch))))
                if not batch: break
                B = len(batch)
                idxs = np.asarray(batch, dtype=int)
                rows, cols = idxs.ravel(), np.repeat(np.arange(B), m)
                data = np.tile(w_nonzero, B)
                
                U = sparse.csr_matrix((data, (rows, cols)), shape=(G, B), dtype=np.float32)
                mu, m2, n_seen = update_stream(mu, m2, n_seen, Xc @ U)
                if abs_variant:
                    Ua = sparse.csr_matrix((np.abs(data), (rows, cols)), shape=(G, B), dtype=np.float32)
                    mu_a, m2_a, n_seen_a = update_stream(mu_a, m2_a, n_seen_a, Xc_abs @ Ua)
        else:
            nP = int(max(1, n_perm))
            b = 0
            while b < nP:
                B = min(perm_batch, nP - b)
                idxs = np.stack([rng.choice(pool, size=m, replace=False) for _ in range(B)], axis=0)
                rows, cols = idxs.ravel(), np.repeat(np.arange(B), m)
                data = np.tile(w_nonzero, B)
                
                U = sparse.csr_matrix((data, (rows, cols)), shape=(G, B), dtype=np.float32)
                mu, m2, n_seen = update_stream(mu, m2, n_seen, Xc @ U)
                if abs_variant:
                    Ua = sparse.csr_matrix((np.abs(data), (rows, cols)), shape=(G, B), dtype=np.float32)
                    mu_a, m2_a, n_seen_a = update_stream(mu_a, m2_a, n_seen_a, Xc_abs @ Ua)
                b += B

        sd = (np.sqrt(m2 / max(n_seen, 1)) + 1e-6)
        z = (obs[:, s] - mu) / sd
        Z[:, s] = z.astype(np.float32)
        Pmat[:, s] = p_from_Z(z)
        
        if abs_variant:
            sd_a = (np.sqrt(m2_a / max(n_seen_a, 1)) + 1e-6)
            z_a = (obs_abs[:, s] - mu_a) / sd_a
            Z_abs[:, s] = z_a.astype(np.float32)
            P_abs[:, s] = p_from_Z(z_a)

    Z_dir = (np.sign(obs) * (Z_abs if abs_variant else np.abs(Z))).astype(np.float32)
    Z_df, P_df, Zdir_df = pd.DataFrame(Z, index=adata.obs_names, columns=set_names), \
                          pd.DataFrame(Pmat, index=adata.obs_names, columns=set_names), \
                          pd.DataFrame(Z_dir, index=adata.obs_names, columns=set_names)

    if abs_variant:
        return Z_df, P_df, pd.DataFrame(Z_abs, index=adata.obs_names, columns=set_names), \
               pd.DataFrame(P_abs, index=adata.obs_names, columns=set_names), Zdir_df
    return Z_df, P_df, Zdir_df
