
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple, Dict, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Try importing lmfit, commonly used in the legacy NormMark code
try:
    import lmfit
except ImportError:
    lmfit = None

# ... (Previous imports and data classes remain unchanged) ...

# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass
class CytofTransformResult:
    """Container for CyTOF-transform outputs."""
    corrected: pd.DataFrame          # arcsinh-corrected intensities (same shape as input)
    residuals_z: pd.DataFrame        # z-scored corrected values (for PCA/UMAP)
    tech_factor: pd.Series           # 1D technical factor f (PC1) per cell
    gamma: Dict[str, float]          # per-marker γ (slope vs tech_factor)
    alpha: Dict[str, float]          # per-marker intercepts (optional / for diagnostics)
    pca_model: Optional[PCA] = None  # PCA model used to compute tech_factor (global or last compartment)

@dataclass
class CytofTransformConfig:
    """Configuration for CyTOF-transform."""
    control_markers: Sequence[str]          # histones + DNA used to define technical factor
    markers_to_correct: Sequence[str]       # markers whose dependence on tech factor we remove
    use_compartments: bool = False          # if True, run per compartment
    n_pcs_for_T: int = 1                    # number of PCs; for this 1D CyTOF-transform, keep as 1
    anchor_to_median: bool = True           # anchor at median(f) so median cell is unchanged
    zscore: bool = True                     # compute z-scored residuals
    line_col: str = None

@dataclass
class CompartmentGatingConfig:
    """Configuration for heuristic compartment calling."""
    cd45: Optional[str] = None
    epithelial_markers: Sequence[str] = ()
    stromal_markers: Sequence[str] = ()
    # thresholds can be absolute (arcsinh scale) or None to use quantiles
    cd45_threshold: Optional[float] = None
    epithelial_threshold: Optional[float] = None
    stromal_threshold: Optional[float] = None
    # quantiles used if thresholds are None
    cd45_quantile: float = 0.8
    epithelial_quantile: float = 0.8
    stromal_quantile: float = 0.8

# ---------------------------------------------------------------------
# Core helper: technical factor via PCA on control markers
# ---------------------------------------------------------------------

def _compute_tech_factor_pc1(
    asinh_data: pd.DataFrame,
    control_markers: Sequence[str],
    n_pcs: int = 1,
    label: Optional[str] = None,
) -> Tuple[pd.Series, PCA]:
    """Compute PC1-based technical factor from control markers."""
    missing = [m for m in control_markers if m not in asinh_data.columns]
    if missing:
        raise ValueError(f"Missing control markers in asinh_data{f' ({label})' if label else ''}: {missing}")

    Y = asinh_data[control_markers].copy()
    n_cells = Y.shape[0]

    if n_cells < 2:
        # Graceful fallback or error?
        raise ValueError(f"Not enough cells ({n_cells}) to compute PCA{f' for {label}' if label else ''}.")

    n_pcs_eff = min(n_pcs, Y.shape[1], max(1, n_cells - 1))
    pca = PCA(n_components=n_pcs_eff)
    X_pcs = pca.fit_transform(Y.values)

    # PC1 as technical factor
    pc1 = X_pcs[:, 0]
    tech_factor = pd.Series(pc1, index=asinh_data.index, name="tech1")

    if label is not None:
        print(f"[CyTOF-transform] Computed PC1 technical factor for '{label}' (explained var PC1 = {pca.explained_variance_ratio_[0]:.3f})")

    return tech_factor, pca

# ---------------------------------------------------------------------
# Core helper: per-marker regression and correction
# ---------------------------------------------------------------------

def _regress_and_correct_1d(
    asinh_data: pd.DataFrame,
    tech_factor: pd.Series,
    markers_to_correct: Sequence[str],
    anchor_to_median: bool = True,
    zscore: bool = True,
    line_labels: Optional[pd.Series] = None,
    max_cells_per_line: Optional[int] = None,
) -> CytofTransformResult:
    """
    Regress arcsinh intensities on 1D technical factor, then subtract γ * (f - anchor).
    """
    ddf = asinh_data.copy()
    f_all = tech_factor.loc[ddf.index].values
    
    # Balancing for slope estimation
    if line_labels is not None:
        if isinstance(line_labels, pd.DataFrame):
            line_labels = line_labels.iloc[:, 0]
        if not isinstance(line_labels, pd.Series):
            line_labels = pd.Series(line_labels, index=ddf.index, name="line")
        
        labels = line_labels.astype("category")
        groups = labels.unique()
        target = min((labels == g).sum() for g in groups) if max_cells_per_line is None else max_cells_per_line

        balanced_idx = []
        for g in groups:
            idx_g = np.where(labels == g)[0]
            chosen = np.random.choice(idx_g, size=target, replace=False) if len(idx_g) >= target else idx_g
            balanced_idx.extend(chosen)
        balanced_idx = np.array(balanced_idx)
    else:
        balanced_idx = np.arange(ddf.shape[0])

    f = f_all[balanced_idx]
    f_centered = f - f.mean()
    denom = np.sum(f_centered**2)
    if denom == 0:
        raise ValueError("Technical factor has zero variance in balanced subset.")

    anchor = np.median(f_all) if anchor_to_median else 0.0
    gamma, alpha = {}, {}

    for m in markers_to_correct:
        if m not in ddf.columns:
            raise ValueError(f"Marker '{m}' not found.")
        y_all = ddf[m].values
        y = y_all[balanced_idx]
        y_mean = y.mean()
        
        num = np.sum(f_centered * (y - y_mean))
        gamma_m = num / denom
        gamma[m] = float(gamma_m)
        alpha[m] = float(y_mean - gamma_m * f.mean())
        
        ddf[m] = y_all - gamma_m * (f_all - anchor)

    residuals_z = ddf.copy()
    if zscore:
        for m in markers_to_correct:
            vals = residuals_z[m].values
            sigma = vals.std()
            residuals_z[m] = (vals - vals.mean()) / (sigma if sigma != 0 else 1.0)

    return CytofTransformResult(ddf, residuals_z, tech_factor, gamma, alpha, None)

def compute_marker_tech_correlations(
    data: pd.DataFrame,
    tech_factor: Optional[pd.Series] = None,
    control_markers: Optional[Sequence[str]] = None,
    n_pcs: int = 1,
    tech_name: str = "tech1",
) -> Tuple[pd.Series, pd.Series]:
    """Compute correlation of markers with a technical factor."""
    if tech_factor is not None:
        f = tech_factor.copy()
    else:
        if control_markers is None:
            raise ValueError("Must provide control_markers if tech_factor is None.")
        Y = data[control_markers].copy()
        if Y.shape[0] < 2: raise ValueError("Not enough cells for PCA.")
        pca = PCA(n_components=min(n_pcs, Y.shape[1])).fit(Y)
        f = pd.Series(pca.transform(Y)[:, 0], index=data.index, name=tech_name)
    
    return data.corrwith(f), f

def cytof_transform_global(asinh_data: pd.DataFrame, config: CytofTransformConfig) -> CytofTransformResult:
    """Run CyTOF-transform on ALL cells together."""
    if config.use_compartments:
        raise ValueError("use_compartments=True not supported here. Use cytof_transform_by_compartment.")
    
    tech_factor, pca = _compute_tech_factor_pc1(asinh_data, config.control_markers, config.n_pcs_for_T, "global")
    ll = asinh_data[config.line_col] if config.line_col else None
    result = _regress_and_correct_1d(asinh_data, tech_factor, config.markers_to_correct, 
                                     config.anchor_to_median, config.zscore, ll)
    result.pca_model = pca
    return result

def cytof_transform_by_compartment(asinh_data: pd.DataFrame, compartments: pd.Series, config: CytofTransformConfig) -> CytofTransformResult:
    """Run CyTOF-transform separately within each compartment."""
    if not asinh_data.index.equals(compartments.index):
        raise ValueError("Index mismatch.")
    
    corrected_list, residuals_list, tech_list = [], [], []
    gamma_all, alpha_all = {m: [] for m in config.markers_to_correct}, {m: [] for m in config.markers_to_correct}
    last_pca = None

    for comp in compartments.unique():
        idx = compartments == comp
        sub_data = asinh_data.loc[idx]
        print(f"[CyTOF-transform] Compartment '{comp}': {sub_data.shape[0]} cells")

        tech_factor_c, pca_c = _compute_tech_factor_pc1(sub_data, config.control_markers, config.n_pcs_for_T, str(comp))
        result_c = _regress_and_correct_1d(sub_data, tech_factor_c, config.markers_to_correct, 
                                           config.anchor_to_median, config.zscore, ll)
        
        corrected_list.append(result_c.corrected)
        residuals_list.append(result_c.residuals_z)
        tech_list.append(result_c.tech_factor)
        for m in config.markers_to_correct:
            gamma_all[m].append(result_c.gamma[m])
            alpha_all[m].append(result_c.alpha[m])
        last_pca = pca_c

    corrected_all = pd.concat(corrected_list).loc[asinh_data.index]
    residuals_all = pd.concat(residuals_list).loc[asinh_data.index]
    tech_all = pd.concat(tech_list).loc[asinh_data.index]
    gamma_mean = {m: float(np.mean(vals)) for m, vals in gamma_all.items()}
    alpha_mean = {m: float(np.mean(vals)) for m, vals in alpha_all.items()}

    return CytofTransformResult(corrected_all, residuals_all, tech_all, gamma_mean, alpha_mean, last_pca)

def evaluate_marker_intensity_regime(
    asinh_data: pd.DataFrame, candidate_markers: Sequence[str], med_thresh=0.3, p90_thresh=0.7
) -> pd.DataFrame:
    """Evaluate if markers are in log-like regime."""
    records = []
    for m in candidate_markers:
        if m not in asinh_data: continue
        vals = asinh_data[m].values
        med, p90 = np.median(vals), np.quantile(vals, 0.9)
        too_low = (med < med_thresh) and (p90 < p90_thresh)
        records.append({"marker": m, "median": med, "p90": p90, "too_low": too_low, "use_for_corr": not too_low})
    return pd.DataFrame.from_records(records).set_index("marker")

def infer_compartments(
    asinh_data: pd.DataFrame, cfg: CompartmentGatingConfig, 
    label_immune="immune", label_epithelial="epithelial", label_stromal="stromal", label_other="other"
) -> pd.Series:
    """Infer broad compartments using heuristic gating."""
    n_cells = asinh_data.shape[0]
    comp = np.full(n_cells, label_other, dtype=object)
    
    def get_score(markers):
        valid = [m for m in markers if m in asinh_data.columns]
        return asinh_data[valid].mean(axis=1) if valid else None

    cd45 = asinh_data[cfg.cd45] if cfg.cd45 and cfg.cd45 in asinh_data else None
    thr_cd45 = cfg.cd45_threshold if cfg.cd45_threshold else (cd45.quantile(cfg.cd45_quantile) if cd45 is not None else None)

    epi = get_score(cfg.epithelial_markers)
    thr_epi = cfg.epithelial_threshold if cfg.epithelial_threshold else (epi.quantile(cfg.epithelial_quantile) if epi is not None else None)

    stroma = get_score(cfg.stromal_markers)
    thr_stroma = cfg.stromal_threshold if cfg.stromal_threshold else (stroma.quantile(cfg.stromal_quantile) if stroma is not None else None)

    mask_epi = (epi >= thr_epi) if epi is not None and thr_epi is not None else np.zeros(n_cells, bool)
    mask_stroma = (stroma >= thr_stroma) if stroma is not None and thr_stroma is not None else np.zeros(n_cells, bool)
    mask_cd45 = (cd45 >= thr_cd45) if cd45 is not None and thr_cd45 is not None else np.zeros(n_cells, bool)

    comp[mask_epi] = label_epithelial
    comp[mask_stroma & ~mask_epi] = label_stromal
    comp[mask_cd45 & ~mask_epi & ~mask_stroma] = label_immune
    
    return pd.Series(comp, index=asinh_data.index)

def cluster_compartments_by_markers(
    asinh_data: pd.DataFrame, clustering_markers: Sequence[str], 
    n_clusters=6, n_pcs=10, random_state=0
) -> Tuple[pd.Series, PCA, KMeans]:
    """Cluster cells into unsupervised groups."""
    Y = asinh_data[[m for m in clustering_markers if m in asinh_data]].copy()
    Y = (Y - Y.mean()) / Y.std().replace(0, 1)
    
    pca = PCA(n_components=min(n_pcs, Y.shape[1]))
    X_pcs = pca.fit_transform(Y)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = pd.Series(kmeans.fit_predict(X_pcs), index=asinh_data.index, name="cluster")
    
    print(f"Clustering: {n_clusters} clusters using {len(clustering_markers)} markers")
    return labels, pca, kmeans

def map_clusters_to_compartments(cluster_labels: pd.Series, mapping: Dict[int, str], default_label="other") -> pd.Series:
    """Map integer clusters to labeled compartments."""
    return cluster_labels.map(mapping).fillna(default_label)

def plot_tech_factor_qc(asinh_data, control_markers, tech_factor=None, n_pcs=2, tech_name="tech1", figsize=(14, 4)):
    """QC plot for technical factor."""
    ctrl = list(control_markers)
    Y = asinh_data[ctrl].values
    
    if tech_factor is None:
        pca = PCA(n_components=min(n_pcs, Y.shape[1])).fit(Y)
        tech_factor = pd.Series(pca.transform(Y)[:, 0], index=asinh_data.index, name=tech_name)
    else:
        pca = PCA(n_components=min(n_pcs, Y.shape[1])).fit(Y)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].hist(tech_factor.values, bins=50, color='b')
    axes[0].set_title(f"{tech_name} distribution")
    
    pd.Series(pca.components_[0], index=ctrl).sort_values(ascending=False).plot(kind="bar", ax=axes[1], color='b')
    axes[1].set_title("PC1 loadings")
    
    axes[2].bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, color='blue')
    axes[2].set_title("PCA explained variance")
    plt.tight_layout()
    plt.show()

def plot_marker_correlations_qc(asinh_pre, asinh_post, tech_factor, markers_to_highlight=None, top_n=25):
    """Plot correlations pre/post normalization."""
    num_cols = asinh_pre.select_dtypes(include=[np.number]).columns
    c_pre = asinh_pre[num_cols].corrwith(tech_factor)
    c_post = asinh_post[num_cols].corrwith(tech_factor)
    
    df = pd.DataFrame({"marker": c_pre.index, "corr_pre": c_pre.values, "corr_post": c_post.reindex(c_pre.index).values})
    
    focus = markers_to_highlight if markers_to_highlight else df.assign(a=df.corr_pre.abs()).sort_values("a", ascending=False).head(top_n).marker.tolist()
    long_df = df[df.marker.isin(focus)].melt(id_vars="marker", value_vars=["corr_pre", "corr_post"])
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    sns.barplot(data=long_df, x="marker", y="value", hue="variable", ax=ax[0])
    ax[0].tick_params(axis='x', rotation=45)
    
    ax[1].scatter(c_pre, c_post, s=10, alpha=0.6)
    lim = max(c_pre.abs().max(), c_post.abs().max()) * 1.05
    ax[1].plot([-lim, lim], [-lim, lim], "k--")
    plt.tight_layout()
    plt.show()

def plot_gamma_qc(gamma, marker_groups=None, figsize=(12, 4)):
    """Barplot of gamma values."""
    s = pd.Series(gamma).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    s.plot(kind="bar", ax=ax)
    ax.tick_params(axis='x', rotation=90)
    plt.title("Gamma values")
    plt.show()

# ---------------------------------------------------------------------------
# Technical factor estimation (per compartment) and regression
# ---------------------------------------------------------------------------

def compute_technical_factors_single_compartment(
    asinh_data: pd.DataFrame,
    control_markers: Sequence[str],
    n_factors: int = 2,
    compartment_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[PCA]]:
    """
    Compute K (= n_factors) technical factors from control markers in a SINGLE compartment.
    """
    missing = [m for m in control_markers if m not in asinh_data.columns]
    if missing:
        raise ValueError(f"Missing control markers in asinh_data: {missing}")

    Yc = asinh_data[control_markers].copy()
    n_cells = Yc.shape[0]

    if n_cells == 0:
        raise ValueError(f"Compartment '{compartment_name}' has zero cells.")

    if n_cells <= n_factors:
        print(f"[WARN] Compartment '{compartment_name}' too small. Returning zero factors.")
        tech = np.zeros((n_cells, n_factors), dtype=float)
        cols = [f"tech{k+1}" for k in range(n_factors)]
        return pd.DataFrame(tech, index=Yc.index, columns=cols), None

    pca = PCA(n_components=n_factors)
    U = pca.fit_transform(Yc.values)
    cols = [f"tech{k+1}" for k in range(n_factors)]
    tech_factors = pd.DataFrame(U, index=Yc.index, columns=cols)

    if compartment_name:
        print(f"Computed {n_factors} factors for '{compartment_name}'.")

    return tech_factors, pca

def regress_out_technical_single_compartment(
    asinh_data: pd.DataFrame,
    tech_factors: pd.DataFrame,
    markers_to_correct: Sequence[str],
) -> pd.DataFrame:
    """
    Regress out multiple technical factors from selected markers in a SINGLE compartment.
    """
    ddf = asinh_data.copy()
    tech_factors = tech_factors.loc[ddf.index]
    U = tech_factors.values
    n_cells, K = U.shape

    if K == 0: return ddf

    U_centered = U - U.mean(axis=0, keepdims=True)
    XtX = U_centered.T @ U_centered
    if np.linalg.matrix_rank(XtX) < K:
        XtX += 1e-8 * np.eye(K)
    XtX_inv = np.linalg.inv(XtX)
    U_median = np.median(U, axis=0, keepdims=True)

    for m in markers_to_correct:
        if m not in ddf.columns: continue
        y = ddf[m].values
        beta = XtX_inv @ (U_centered.T @ (y - y.mean()))
        tech_contrib = (U - U_median) @ beta
        ddf[m] = y - tech_contrib

    return ddf

def normalize_multi_factor_by_compartment(
    asinh_data: pd.DataFrame,
    compartments: pd.Series,
    control_markers: Sequence[str],
    markers_to_correct: Sequence[str],
    n_factors: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-factor, compartment-aware normalization of CyTOF arcsinh data.
    """
    if not asinh_data.index.equals(compartments.index):
        raise ValueError("Index mismatch.")

    corrected_list, tech_list = [], []

    for comp in compartments.unique():
        idx = compartments == comp
        sub_data = asinh_data.loc[idx]
        print(f"\n=== Normalizing compartment '{comp}' ({sub_data.shape[0]} cells) ===")

        tech_factors, _ = compute_technical_factors_single_compartment(
            sub_data, control_markers, n_factors, str(comp)
        )
        tech_list.append(tech_factors)

        corrected_sub = regress_out_technical_single_compartment(
            sub_data, tech_factors, markers_to_correct
        )
        corrected_list.append(corrected_sub)

    corrected_all = pd.concat(corrected_list).loc[asinh_data.index]
    tech_factors_all = pd.concat(tech_list).loc[asinh_data.index]

    return corrected_all, tech_factors_all


# ---------------------------------------------------------------------------
# Optimization-based normalization (NormMark) - using lmfit if available
# ---------------------------------------------------------------------------

def normalize_markers_optimization(data: pd.DataFrame, 
                                   norm_columns: List[str] = ['H3.3', 'H4', 'H3'], 
                                   norm_markers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize data using a weighted combination of normalization columns.
    Optimization via lmfit to find weights that minimize internal variance.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data to normalize.
    norm_columns : list of str
        List of column names to use for normalization (1-4 columns).
    norm_markers : list of str, optional
        List of markers to normalize. If None, uses all columns except norm_columns.
    
    Returns:
    --------
    pd.DataFrame
        Normalized data.
    """
    if lmfit is None:
        raise ImportError("lmfit is required for normalize_markers_optimization. Please install it.")
    
    from lmfit import Parameters, minimize

    if len(norm_columns) < 1:
        raise ValueError("norm_columns must contain at least 1 column")
    
    if norm_markers is None:
        norm_markers = [col for col in data.columns if col not in norm_columns]
    
    ddf = data.copy()
    ddf2 = data.copy()
    
    # Check bounds
    missing_cols = [c for c in norm_columns if c not in ddf.columns]
    if missing_cols:
        raise ValueError(f"Normalization columns not found in data: {missing_cols}")

    # Case 1: Single column - no optimization needed
    if len(norm_columns) == 1:
        Q = ddf[norm_markers].mean()
        M = (ddf / Q)[norm_columns[0]]
        ddf[norm_markers] = ddf[norm_markers].divide(M, axis=0).copy()
        ddf2[norm_markers] = ddf[norm_markers]
        print(f"Single column normalization using {norm_columns[0]}")
        return ddf2
    
    # Case 2+: Normalized weighted combination
    Q = ddf[norm_markers].mean()
    M_list = [(ddf / Q)[col] for col in norm_columns]
    
    n_params = len(norm_columns) - 1
    
    def objective(p, x, data, Q, M_list, cols_list):
        # Get parameter values
        param_values = [p[f'p{i}'].value for i in range(n_params)]
        # Last weight is 1 - sum of others
        last_weight = 1 - sum(param_values)
        
        # Ensure weights sum to 1 and are non-negative
        if last_weight < 0:
            return 1e10  # Penalty for invalid weights
        
        # Build weighted combination
        M_combined = param_values[0] * M_list[0]
        for i in range(1, n_params):
            M_combined += param_values[i] * M_list[i]
        M_combined += last_weight * M_list[-1]
        
        # Divide and compute sum of squared standard deviations
        d = x.divide(M_combined, axis=0)
        return sum(d.std()[col]**2 for col in cols_list)
    
    # Create parameters
    params = Parameters()
    param_names = [f'p{i}' for i in range(n_params)]
    
    initial_value = 1.0 / len(norm_columns)
    for i, pname in enumerate(param_names):
        # Constraints based on original code logic
        if i == 0:
            max_val = 1.0 if len(norm_columns) == 2 else 0.9
            params.add(pname, value=initial_value, min=0.0, max=max_val)
        else:
            params.add(pname, value=initial_value, min=0, max=0.9)

    # Minimize
    # Minimize
    # Wrapping objective to pass fixed args easily
    def objective_wrapper(p):
        # We want to minimize the variance of the NORMALIZATION columns (norm_columns).
        # We pass ddf (which contains them) as 'x' and 'norm_columns' as 'cols_list'.
        # Note: The original code passed ddf[norm_markers], but that requires norm_markers 
        # to contain norm_columns. To be safe/correct, we pass ddf (or a subset containing them).
        return objective(p, ddf, None, Q, M_list, norm_columns)

    out = minimize(objective_wrapper, params, method='cg')
    
    # Extract optimized parameters
    param_values = [out.params[pname].value for pname in param_names]
    last_weight = 1 - sum(param_values)
    
    # Build final weighted combination
    M = param_values[0] * M_list[0]
    for i in range(1, n_params):
        M += param_values[i] * M_list[i]
    M += last_weight * M_list[-1]
    
    # Apply normalization
    ddf[norm_markers] = ddf[norm_markers].divide(M, axis=0).copy()
    ddf2[norm_markers] = ddf[norm_markers]
    
    print(f"Normalization using columns: {norm_columns}")
    print(f"Optimized weights: {param_values + [last_weight]}")
    
    return ddf2
