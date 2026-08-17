"""
cytof_transform.py

CyTOF-transform: sctransform-like normalization for CyTOF data.

Core ideas:
- Use core histones + DNA as control markers to estimate a per-cell
  technical factor T (via PC1 on arcsinh-transformed data).
- Assume intracellular markers follow a multiplicative model:
      X_{i,m} ≈ Biology_{i,m} * T_i^{γ_m} * noise
  which in arcsinh/log space becomes additive:
      y_{i,m} ≈ α_m + γ_m * f_i + BiologyResidual_{i,m}
  where f_i is a 1D technical factor (PC1).
- Fit γ_m by linear regression of y_m on f.
- Correct by subtracting γ_m * (f - median(f)), i.e. regress out
  the technical factor but anchor at a median “reference” cell.
- Optionally z-score the corrected data for downstream PCA/UMAP.

Supports:
- Global transform (all cells together)
- Compartment-aware transform (per lineage / compartment)

Author: ChatGPT (joint design with user)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap  # pip install umap-learn

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
    """
    Configuration for CyTOF-transform.
    """
    control_markers: Sequence[str]          # histones + DNA used to define technical factor
    markers_to_correct: Sequence[str]       # markers whose dependence on tech factor we remove
    use_compartments: bool = False          # if True, run per compartment
    n_pcs_for_T: int = 1                    # number of PCs; for this 1D CyTOF-transform, keep as 1
    anchor_to_median: bool = True           # anchor at median(f) so median cell is unchanged
    zscore: bool = True                     # compute z-scored residuals
    line_col: str = None


# ---------------------------------------------------------------------
# Core helper: technical factor via PCA on control markers
# ---------------------------------------------------------------------

def _compute_tech_factor_pc1(
    asinh_data: pd.DataFrame,
    control_markers: Sequence[str],
    n_pcs: int = 1,
    label: Optional[str] = None,
) -> Tuple[pd.Series, PCA]:
    """
    Compute PC1-based technical factor from control markers (histones + DNA).

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed.
    control_markers : list of str
        Names of control markers in asinh_data (core histones + DNA).
    n_pcs : int, default 1
        Number of PCs to compute; only PC1 is used as technical factor here.
    label : str, optional
        For logging, e.g. compartment name.

    Returns
    -------
    tech_factor : Series
        Length = n_cells, index = asinh_data.index, values = PC1 scores.
    pca : PCA
        Fitted PCA object from sklearn.decomposition.
    """
    missing = [m for m in control_markers if m not in asinh_data.columns]
    if missing:
        raise ValueError(f"Missing control markers in asinh_data{f' ({label})' if label else ''}: {missing}")

    Y = asinh_data[control_markers].copy()
    n_cells = Y.shape[0]

    if n_cells < 2:
        raise ValueError(f"Not enough cells ({n_cells}) to compute PCA{f' for {label}' if label else ''}.")

    n_pcs_eff = min(n_pcs, Y.shape[1], max(1, n_cells - 1))
    pca = PCA(n_components=n_pcs_eff)
    X_pcs = pca.fit_transform(Y.values)  # shape (n_cells, n_pcs_eff)

    # PC1 as technical factor
    pc1 = X_pcs[:, 0]
    tech_factor = pd.Series(pc1, index=asinh_data.index, name="tech1")

    if label is not None:
        print(
            f"[CyTOF-transform] Computed PC1 technical factor for '{label}' "
            f"(explained var PC1 = {pca.explained_variance_ratio_[0]:.3f})"
        )

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
    For each marker, regress arcsinh intensities on a 1D technical factor (PC1),
    then subtract γ * (f - anchor) to obtain corrected values. Optionally z-score.

    If line_labels is provided, gamma is estimated using a balanced subset
    with (approximately) the same number of cells per line, and then applied
    to all cells.
    """

    ddf = asinh_data.copy()
    f_all = tech_factor.loc[ddf.index].values  # full vector for correction
    
    # ---- Step 1: Build balanced subset for slope estimation ----
    if line_labels is not None:
        # Coerce to a 1D Series aligned with asinh_data.index
        if isinstance(line_labels, pd.DataFrame):
            if line_labels.shape[1] != 1:
                raise ValueError(
                    "line_labels DataFrame must have exactly one column, "
                    f"got {line_labels.shape[1]}."
                )
            line_labels = line_labels.iloc[:, 0]

        if not isinstance(line_labels, pd.Series):
            # Assume array-like; wrap as Series with same index as asinh_data
            line_labels = pd.Series(line_labels, index=ddf.index, name="line")

        if not line_labels.index.equals(ddf.index):
            raise ValueError("line_labels index must match asinh_data.index")

        labels = line_labels.astype("category")
        groups = labels.unique()

        # Determine target size per line
        if max_cells_per_line is None:
            target = min((labels == g).sum() for g in groups)
        else:
            target = max_cells_per_line

        balanced_idx = []
        for g in groups:
            idx_g = np.where(labels == g)[0]
            if len(idx_g) >= target:
                chosen = np.random.choice(idx_g, size=target, replace=False)
            else:
                chosen = idx_g  # if smaller than target, take all
            balanced_idx.extend(chosen)

        balanced_idx = np.array(balanced_idx)
    else:
        # no balancing – use all cells
        balanced_idx = np.arange(ddf.shape[0])

    # Subset for fitting
    f = f_all[balanced_idx]
    f_centered = f - f.mean()
    denom = np.sum(f_centered**2)
    if denom == 0:
        raise ValueError("Technical factor has zero variance in balanced subset.")

    # ---- Anchoring ----
    if anchor_to_median:
        anchor = np.median(f_all)  # median over full dataset
    else:
        anchor = 0.0

    gamma: Dict[str, float] = {}
    alpha: Dict[str, float] = {}

    # ---- Step 2: Estimate gamma using only balanced subset ----
    for m in markers_to_correct:
        if m not in ddf.columns:
            raise ValueError(f"Marker '{m}' not found in asinh_data columns.")

        y_all = ddf[m].values
        y = y_all[balanced_idx]

        y_mean = y.mean()
        y_centered = y - y_mean

        num = np.sum(f_centered * y_centered)
        gamma_m = num / denom
        gamma[m] = float(gamma_m)

        # intercept from balanced set (mainly for diagnostics)
        alpha_m = y_mean - gamma_m * f.mean()
        alpha[m] = float(alpha_m)

        # ---- Step 3: Apply correction to ALL cells ----
        ddf[m] = y_all - gamma_m * (f_all - anchor)

    # ---- Step 4: z-score if requested ----
    if zscore:
        residuals_z = ddf.copy()
        for m in markers_to_correct:
            vals = residuals_z[m].values
            mu = vals.mean()
            sigma = vals.std()
            if sigma == 0:
                sigma = 1.0
            residuals_z[m] = (vals - mu) / sigma
    else:
        residuals_z = ddf.copy()

    return CytofTransformResult(
        corrected=ddf,
        residuals_z=residuals_z,
        tech_factor=tech_factor,
        gamma=gamma,
        alpha=alpha,
        pca_model=None,
    )




import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Optional, Sequence, Tuple


def compute_marker_tech_correlations(
    data: pd.DataFrame,
    tech_factor: Optional[pd.Series] = None,
    control_markers: Optional[Sequence[str]] = None,
    n_pcs: int = 1,
    tech_name: str = "tech1",
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute correlation of every marker (column) in `data` with a technical factor.

    - If `tech_factor` is provided: use it directly.
    - If `tech_factor` is None: compute it as PC1 of `control_markers` using PCA.

    Parameters
    ----------
    data : DataFrame
        Cells × markers (e.g. arcsinh-transformed CyTOF data).
    tech_factor : Series, optional
        Length = n_cells, index matching `data.index`. If given, used as the
        technical factor. If None, PC1 is computed from `control_markers`.
    control_markers : list of str, optional
        Core histones / DNA used to compute PC1 if `tech_factor` is None.
        Must be provided if `tech_factor` is None.
    n_pcs : int, default 1
        Number of PCs to compute if `tech_factor` is None (only PC1 is used).
    tech_name : str, default "tech1"
        Name to give to the computed technical factor Series.

    Returns
    -------
    corr : Series
        Index = marker names (columns of `data`), values = Pearson correlation
        with the technical factor.
    tech_factor : Series
        The technical factor used (either the one you gave, or the computed PC1).
    """
    # -----------------------------
    # 1. Get / compute technical factor
    # -----------------------------
    if tech_factor is not None:
        # Sanity check index alignment
        if not tech_factor.index.equals(data.index):
            raise ValueError("Index of tech_factor must match data.index.")
        f = tech_factor.copy()
    else:
        if control_markers is None:
            raise ValueError(
                "If tech_factor is None, you must provide control_markers "
                "to compute PC1 as the technical factor."
            )
        missing = [m for m in control_markers if m not in data.columns]
        if missing:
            raise ValueError(f"control_markers not found in data: {missing}")

        Y = data[control_markers].copy()
        n_cells = Y.shape[0]
        if n_cells < 2:
            raise ValueError("Not enough cells to compute PCA (need at least 2).")

        n_pcs_eff = min(n_pcs, Y.shape[1], max(1, n_cells - 1))
        pca = PCA(n_components=n_pcs_eff)
        X_pcs = pca.fit_transform(Y.values)  # (cells × n_pcs_eff)

        # PC1 as technical factor
        pc1 = X_pcs[:, 0]
        f = pd.Series(pc1, index=data.index, name=tech_name)
        print(
            f"[compute_marker_tech_correlations] "
            f"Computed {tech_name} from PC1 of control markers. "
            f"Explained var (PC1) = {pca.explained_variance_ratio_[0]:.3f}"
        )

    # -----------------------------
    # 2. Compute correlations
    # -----------------------------
    # Use pandas corrwith, which centers internally
    corr = data.corrwith(f)

    return corr, f


    
# ---------------------------------------------------------------------
# Public API: global CyTOF-transform
# ---------------------------------------------------------------------

def cytof_transform_global(
    asinh_data: pd.DataFrame,
    config: CytofTransformConfig,
) -> CytofTransformResult:
    """
    Run CyTOF-transform on ALL cells together (no compartments).

    Steps:
    1. Compute PC1 of control markers → technical factor f.
    2. For each marker in markers_to_correct, regress y_m on f.
    3. Subtract γ_m * (f - median(f)) to get corrected y_m^corr.
    4. Optionally z-score corrected markers.

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed.
    config : CytofTransformConfig
        Configuration specifying control markers, markers_to_correct, etc.

    Returns
    -------
    CytofTransformResult
    """
    if config.use_compartments:
        raise ValueError(
            "config.use_compartments=True is not supported in cytof_transform_global. "
            "Use cytof_transform_by_compartment instead."
        )

    # 1) PC1 technical factor
    tech_factor, pca = _compute_tech_factor_pc1(
        asinh_data=asinh_data,
        control_markers=config.control_markers,
        n_pcs=config.n_pcs_for_T,
        label="global",
    )
    # 2) Regression & correction
    if config.line_col is not None:
        ll=asinh_data[config.line_col]
    else:
        ll=None 
    result = _regress_and_correct_1d(
        asinh_data=asinh_data,
        tech_factor=tech_factor,
        markers_to_correct=config.markers_to_correct,
        anchor_to_median=config.anchor_to_median,
        zscore=config.zscore,
        line_labels=ll
    )

    result.pca_model = pca
    return result


# ---------------------------------------------------------------------
# Public API: compartment-aware CyTOF-transform
# ---------------------------------------------------------------------

def cytof_transform_by_compartment(
    asinh_data: pd.DataFrame,
    compartments: pd.Series,
    config: CytofTransformConfig,
) -> CytofTransformResult:
    """
    Run CyTOF-transform separately within each compartment (e.g. immune / tumor / stroma),
    then recombine results.

    For each compartment:
      1. Compute PC1 from control markers → f^(c)
      2. Regress markers_to_correct on f^(c)
      3. Subtract γ_m^(c) * (f^(c) - median(f^(c))) within that compartment

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed.
    compartments : Series
        Length = n_cells, index matching asinh_data.index, with compartment labels.
    config : CytofTransformConfig
        Configuration specifying control markers, markers_to_correct, etc.
        config.use_compartments is ignored; this function is explicitly compartment-aware.

    Returns
    -------
    CytofTransformResult
        corrected: full data, corrected per compartment
        residuals_z: z-scored corrected values
        tech_factor: concatenated technical factor (PC1) across all compartments
                     (per-cell; same index as asinh_data)
        gamma/alpha: dicts with **global** γ/α per marker (simple average across compartments)
                     (for compartment-specific γ, you may adapt this function)
    """
    if not asinh_data.index.equals(compartments.index):
        raise ValueError("Index of asinh_data and compartments must match.")

    corrected_list = []
    residuals_list = []
    tech_list = []
    gamma_all: Dict[str, List[float]] = {m: [] for m in config.markers_to_correct}
    alpha_all: Dict[str, List[float]] = {m: [] for m in config.markers_to_correct}
    last_pca = None

    for comp in compartments.unique():
        idx = compartments == comp
        sub_data = asinh_data.loc[idx]

        print(
            f"[CyTOF-transform] Compartment '{comp}': {sub_data.shape[0]} cells "
            f"({sub_data.shape[1]} markers)"
        )

        # 1) PC1 technical factor for this compartment
        tech_factor_c, pca_c = _compute_tech_factor_pc1(
            asinh_data=sub_data,
            control_markers=config.control_markers,
            n_pcs=config.n_pcs_for_T,
            label=str(comp),
        )

        # 2) Regression & correction in this compartment
        result_c = _regress_and_correct_1d(
            asinh_data=sub_data,
            tech_factor=tech_factor_c,
            markers_to_correct=config.markers_to_correct,
            anchor_to_median=config.anchor_to_median,
            zscore=config.zscore,
        )

        corrected_list.append(result_c.corrected)
        residuals_list.append(result_c.residuals_z)
        tech_list.append(result_c.tech_factor)

        for m in config.markers_to_correct:
            gamma_all[m].append(result_c.gamma[m])
            alpha_all[m].append(result_c.alpha[m])

        last_pca = pca_c  # keep last for reference

    # Concatenate in original order
    corrected_all = pd.concat(corrected_list, axis=0).loc[asinh_data.index]
    residuals_all = pd.concat(residuals_list, axis=0).loc[asinh_data.index]
    tech_all = pd.concat(tech_list, axis=0).loc[asinh_data.index]

    # Average γ/α across compartments (for summary)
    gamma_mean = {m: float(np.mean(vals)) for m, vals in gamma_all.items()}
    alpha_mean = {m: float(np.mean(vals)) for m, vals in alpha_all.items()}

    return CytofTransformResult(
        corrected=corrected_all,
        residuals_z=residuals_all,
        tech_factor=tech_all,
        gamma=gamma_mean,
        alpha=alpha_mean,
        pca_model=last_pca,
    )

def evaluate_marker_intensity_regime(
    asinh_data: pd.DataFrame,
    candidate_markers: Sequence[str],
    med_thresh: float = 0.3,
    p90_thresh: float = 0.7,
) -> pd.DataFrame:
    """
    Evaluate which markers live mostly in the 'log-like' region of the arcsinh transform
    and which live mostly in the near-zero / linear region.

    Heuristic:
      - For each marker, compute median and 90th percentile in arcsinh space.
      - If BOTH median < med_thresh AND p90 < p90_thresh, we consider the marker
        'too low' to confidently assume arcsinh ≈ log, and we flag it as NOT
        recommended for permeability-based correction.

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed.
    candidate_markers : list of str
        Markers you are considering for correction (e.g., NormMRK).
    med_thresh : float, default 0.3
        Threshold on the median arcsinh value.
    p90_thresh : float, default 0.7
        Threshold on the 90th percentile arcsinh value.

    Returns
    -------
    df : DataFrame
        Index: marker names
        Columns:
            'median'         : median(asinh)
            'p90'            : 90th percentile(asinh)
            'too_low'        : True if marker is mostly in linear regime
            'use_for_corr'   : True if NOT too_low
    """
    records = []
    missing = [m for m in candidate_markers if m not in asinh_data.columns]
    if missing:
        raise ValueError(f"Markers not found in asinh_data: {missing}")

    for m in candidate_markers:
        vals = asinh_data[m].values
        med = np.median(vals)
        p90 = np.quantile(vals, 0.9)
        too_low = (med < med_thresh) and (p90 < p90_thresh)
        records.append(
            {
                "marker": m,
                "median": med,
                "p90": p90,
                "too_low": too_low,
                "use_for_corr": not too_low,
            }
        )

    df = pd.DataFrame.from_records(records).set_index("marker")
    return df


def plot_tech_factor_qc(
    asinh_data: pd.DataFrame,
    control_markers,
    tech_factor: pd.Series | None = None,
    n_pcs: int = 2,
    tech_name: str = "tech1",
    figsize=(14, 4),
):
    """
    QC for the technical factor (PC1).
    
    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed.
    control_markers : list-like
        Markers used to define the technical factor (e.g. core histones).
    tech_factor : Series, optional
        If provided, used as the technical factor. If None, PC1 is computed.
    n_pcs : int
        Number of PCs to compute if tech_factor is None (only PC1 is plotted).
    tech_name : str
        Name of the technical factor (used for labels).
    figsize : tuple
        Figure size.
    """
    #sns.set(style="whitegrid")
    ctrl = list(control_markers)

    missing = [m for m in ctrl if m not in asinh_data.columns]
    if missing:
        raise ValueError(f"Missing control markers in asinh_data: {missing}")

    if tech_factor is None:
        Y = asinh_data[ctrl].values
        n_cells = Y.shape[0]
        n_pcs_eff = min(n_pcs, len(ctrl), max(1, n_cells - 1))
        pca = PCA(n_components=n_pcs_eff)
        X_pcs = pca.fit_transform(Y)
        tech_factor = pd.Series(X_pcs[:, 0], index=asinh_data.index, name=tech_name)
        explained = pca.explained_variance_ratio_
        loadings = pd.Series(pca.components_[0], index=ctrl)
    else:
        # If you provide tech_factor, we still compute PCA to get loadings for QC.
        Y = asinh_data[ctrl].values
        n_cells = Y.shape[0]
        n_pcs_eff = min(n_pcs, len(ctrl), max(1, n_cells - 1))
        pca = PCA(n_components=n_pcs_eff)
        pca.fit(Y)
        explained = pca.explained_variance_ratio_
        loadings = pd.Series(pca.components_[0], index=ctrl)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1) Histogram of tech_factor
    ax = axes[0]
    ax.hist(tech_factor.values, bins=50, alpha=1,color='b')
    ax.set_xlabel(tech_name)
    ax.set_ylabel("Cell count")
    ax.set_title(f"{tech_name} distribution")

    # 2) Loadings barplot
    ax = axes[1]
    loadings.sort_values(ascending=False).plot(kind="bar", ax=ax,color='b')
    ax.set_ylabel("PC1 loading")
    ax.set_title("PC1 loadings for control markers")

    # 3) Explained variance
    ax = axes[2]
    x = np.arange(1, len(explained) + 1)
    ax.bar(x, explained,color='blue')
    ax.set_xticks(x)
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained variance (control markers)")

    plt.tight_layout()
    plt.show()

    return tech_factor, loadings, explained

def plot_marker_correlations_qc(
    asinh_pre: pd.DataFrame,
    asinh_post: pd.DataFrame,
    tech_factor: pd.Series,
    markers_to_highlight=None,
    top_n: int = 25,
    figsize=(20, 5),
):
    """
    Plot correlations of all markers with the technical factor, pre vs post.

    Parameters
    ----------
    asinh_pre : DataFrame
        Cells × markers, BEFORE normalization.
    asinh_post : DataFrame
        Cells × markers, AFTER normalization.
    tech_factor : Series
        Technical factor (e.g. PC1). Index must match asinh_pre/asinh_post.
    markers_to_highlight : list-like, optional
        Markers to focus the barplot on. If None, top_n by |corr_pre| are used.
    top_n : int
        If markers_to_highlight is None, number of markers with largest
        |corr_pre| to show.
    figsize : tuple
        Figure size.
    """
    assert asinh_pre.index.equals(asinh_post.index)
    assert asinh_pre.index.equals(tech_factor.index)
    numerical_columns_list = asinh_pre.select_dtypes(include=[np.number]).columns.tolist()

    # Correlations pre & post
    corr_pre = asinh_pre[numerical_columns_list].corrwith(tech_factor)
    corr_post = asinh_post[numerical_columns_list].corrwith(tech_factor)

    corr_df = pd.DataFrame({
        "marker": corr_pre.index,
        "corr_pre": corr_pre.values,
        "corr_post": corr_post.reindex(corr_pre.index).values,
    })

    if markers_to_highlight is not None:
        markers = [m for m in markers_to_highlight if m in corr_df["marker"].values]
    else:
        # Take the top_n markers by absolute pre-correction correlation
        markers = (
            corr_df
            .assign(abs_pre=lambda d: d["corr_pre"].abs())
            .sort_values("abs_pre", ascending=False)
            .head(top_n)["marker"]
            .tolist()
        )

    long_df = (
        corr_df
        .loc[corr_df["marker"].isin(markers)]
        .melt(id_vars="marker", value_vars=["corr_pre", "corr_post"],
              var_name="state", value_name="corr")
    )
    state_map = {"corr_pre": "Before normalization", "corr_post": "After normalization"}
    long_df["state"] = long_df["state"].map(state_map)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1) Focused barplot
    ax = axes[0]
    sns.barplot(
        data=long_df,
        x="marker",
        y="corr",
        hue="state",
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Pearson corr(marker, tech)")
    ax.set_xlabel("")
    ax.set_title("Correlations with technical factor\n(selected markers)")
    ax.legend(frameon=False)

    # 2) All markers scatter
    ax = axes[1]
    ax.scatter(corr_pre, corr_post, s=10, alpha=0.6)
    lim = max(corr_pre.abs().max(), corr_post.abs().max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("corr pre")
    ax.set_ylabel("corr post")
    ax.set_title("All markers: correlation pre vs post")

    plt.tight_layout()
    plt.show()

    return corr_df

def plot_gamma_qc(
    gamma: dict[str, float],
    marker_groups: dict[str, list[str]] | None = None,
    figsize=(12, 4),
):
    """
    Barplot of gamma values for all markers.

    Parameters
    ----------
    gamma : dict
        Mapping marker -> gamma value.
    marker_groups : dict, optional
        Mapping group_name -> list of markers. Used to color bars by group.
        If None, all markers shown in a single color.
    figsize : tuple
        Figure size.
    """
    gamma_series = pd.Series(gamma).sort_values(ascending=False)

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    if marker_groups is None:
        gamma_series.plot(kind="bar", ax=ax)
        ax.set_ylabel("gamma (slope vs tech factor)")
        ax.set_xlabel("marker")
        ax.set_title("Marker-specific gamma values")
        ax.tick_params(axis="x", rotation=90)
    else:
        df = gamma_series.rename("gamma").reset_index().rename(columns={"index": "marker"})
        # Assign groups
        group_labels = []
        for m in df["marker"]:
            assigned = None
            for gname, mlist in marker_groups.items():
                if m in mlist:
                    assigned = gname
                    break
            group_labels.append(assigned if assigned is not None else "other")
        df["group"] = group_labels
        sns.set_palette("bright")
        sns.barplot(
            data=df,
            x="marker",
            y="gamma",
            hue="group",
            dodge=False,
            ax=ax,
        )
        ax.set_ylabel("gamma (slope vs tech factor)")
        ax.set_xlabel("marker")
        ax.set_title("Marker-specific gamma values by group")
        ax.tick_params(axis="x", rotation=90)
        ax.legend(frameon=False, title="Group")

    plt.tight_layout()
    plt.show()

def plot_umap_qc(
    asinh_pre: pd.DataFrame,
    asinh_post: pd.DataFrame,
    tech_factor: pd.Series,
    umap_markers=None,
    bio_marker: str | None = None,
    control_histones=None,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    random_state: int = 0,
    figsize=(14, 10),
):
    """
    UMAP QC: compare pre vs post normalization in a shared embedding.

    Parameters
    ----------
    asinh_pre : DataFrame
        Cells × markers, BEFORE normalization.
    asinh_post : DataFrame
        Cells × markers, AFTER normalization.
    tech_factor : Series
        Technical factor (e.g. PC1).
    umap_markers : list-like, optional
        Markers to use as UMAP features. If None, all columns of asinh_post.
    bio_marker : str, optional
        A biological marker to color UMAPs with (pre vs post).
    control_histones : list-like, optional
        Markers whose mean will be plotted as 'total histone intensity'.
    n_neighbors, min_dist, random_state : UMAP parameters.
    figsize : tuple
        Figure size.
    """
    assert asinh_pre.index.equals(asinh_post.index)
    assert asinh_pre.index.equals(tech_factor.index)

    if umap_markers is None:
        umap_markers = asinh_post.columns.tolist()

    X = asinh_post[umap_markers].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state, verbose=True
    )
    umap_coords = reducer.fit_transform(X_std)

    # Build figure
    sns.set(style="white")
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: UMAP colored by tech factor
    ax = axes[0, 0]
    sc = ax.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=tech_factor.values,
        s=3,
        cmap="viridis",
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP colored by technical factor")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("tech factor")

    # Panel 2: UMAP colored by total histones (pre-normalization)
    if control_histones is not None:
        total_hist_pre = asinh_pre[control_histones].mean(axis=1)
        ax = axes[0, 1]
        sc2 = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=total_hist_pre.values,
            s=3,
            cmap="magma",
        )
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title("Total core-histone intensity (pre-normalization)")
        cbar2 = plt.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04)
        cbar2.set_label("mean(core histones)")
    else:
        axes[0, 1].axis("off")

    # Panel 3 & 4: UMAP colored by a biological marker (pre vs post)
    if bio_marker is not None and bio_marker in asinh_pre.columns:
        bio_pre = asinh_pre[bio_marker].values
        bio_post = asinh_post[bio_marker].values

        ax = axes[1, 0]
        sc3 = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=bio_pre,
            s=3,
            cmap="plasma",
        )
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"{bio_marker} before normalization")
        cbar3 = plt.colorbar(sc3, ax=ax, fraction=0.046, pad=0.04)
        cbar3.set_label(f"{bio_marker} (asinh)")

        ax = axes[1, 1]
        sc4 = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=bio_post,
            s=3,
            cmap="plasma",
        )
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"{bio_marker} after normalization")
        cbar4 = plt.colorbar(sc4, ax=ax, fraction=0.046, pad=0.04)
        cbar4.set_label(f"{bio_marker} (asinh)")
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()
    return umap_coords