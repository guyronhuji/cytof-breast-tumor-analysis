"""
core.py — CyTOF-transform implementation.

CyTOF-transform: sctransform-like normalization for CyTOF data.

Core ideas
----------
- Use core histones + DNA as control markers to estimate a per-cell
  technical factor T (via PC1 on arcsinh-transformed data).
- Assume intracellular markers follow a multiplicative model::

      X_{i,m} ≈ Biology_{i,m} * T_i^{γ_m} * noise

  which in arcsinh/log space becomes additive::

      y_{i,m} ≈ α_m + γ_m * f_i + BiologyResidual_{i,m}

  where f_i is a 1D technical factor (PC1).
- Fit γ_m by linear regression of y_m on f.
- Correct by subtracting γ_m * (f - median(f)), i.e. regress out
  the technical factor but anchor at a median "reference" cell.
- Optionally z-score the corrected data for downstream PCA/UMAP.

Normalization families (see :func:`cytof_normalize`)
----------------------------------------------------
- ``"regress"`` — sctransform-inspired PC1 regression (Method A, default).
- ``"divide"``  — legacy std-minimizing permeabilization division (Method B1),
  for backwards compatibility / reproducing older results. Division is
  multiplicative and belongs on RAW pre-arcsinh counts; see
  ``CytofTransformConfig.arcsinh_cofactor``.

Supports
--------
- Global transform (all cells together)
- Compartment-aware transform (per lineage / compartment)
- Balanced-line sampling to prevent large batches from biasing γ estimates
- γ pooling / shrinkage (``gamma_mode``) and cross-group stability shrinkage
- Protected biological covariates (``protect_covariates``) that are adjusted
  for but retained rather than regressed out
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass
class CytofTransformResult:
    """Container for CyTOF-transform outputs."""
    corrected: pd.DataFrame          # arcsinh-corrected intensities (same shape as input)
    residuals_z: pd.DataFrame        # z-scored corrected values (for PCA/UMAP)
    tech_factor: pd.Series           # 1D technical factor f (PC1) per cell
    gamma: Dict[str, float]          # per-marker γ actually applied (after pooling/shrinkage)
    alpha: Dict[str, float]          # per-marker intercepts (optional / for diagnostics)
    pca_model: Optional[PCA] = None  # PCA model used to compute tech_factor (global or last compartment)
    gamma_shrink: Optional[pd.DataFrame] = None  # per-marker diagnostics: gamma_ols, se, B, gamma_used
                                                 # .attrs holds gamma_bar, tau2, gamma_mode


@dataclass
class CytofTransformConfig:
    """
    Configuration for CyTOF-transform.
    """
    control_markers: Sequence[str]          # histones + DNA used to define technical factor
    markers_to_correct: Sequence[str]       # markers whose dependence on tech factor we remove
    # --- which normalization family (see cytof_normalize) ---
    # "regress" (default) = sctransform-inspired PC1 regression (Method A).
    # "divide"            = legacy std-minimizing permeabilization DIVISION (Method B1),
    #                       for backwards compatibility / reproducing older results.
    method: str = "regress"
    # For method="divide" ONLY: division is a multiplicative correction and belongs on
    # RAW (linear, pre-arcsinh) counts. If arcsinh_cofactor is set, cytof_normalize
    # treats the input as RAW, divides by the permeabilization factor, THEN applies
    # arcsinh(x / cofactor) so the output is on the arcsinh scale (correct pipeline:
    # raw -> divide -> arcsinh). If None, the divide is applied to the input as-is and
    # returned on that scale (exact legacy reproduction; you arcsinh downstream).
    arcsinh_cofactor: Optional[float] = None
    use_compartments: bool = False          # if True, run per compartment
    n_pcs_for_T: int = 1                    # number of PCs; for this 1D CyTOF-transform, keep as 1
    anchor_to_median: bool = True           # anchor at median(f) so median cell is unchanged
    zscore: bool = True                     # compute z-scored residuals
    line_col: str = None
    # --- how the per-marker slope gamma is estimated ---
    gamma_mode: str = "per_marker"          # "per_marker" | "single" | "shrink"
    shrink_target: str = "control"          # target for "single"/"shrink": "control" | "global"
    # --- biological covariates to adjust for but KEEP (not remove) ---
    # e.g. ["IdU", "CyclinB1", "pRb", "DNA"]. Only those present in the data are
    # used, so a panel missing some of them still works. None = plain 1D regression.
    protect_covariates: Optional[Sequence[str]] = None
    # --- cross-group stability shrinkage (gamma_mode="shrink_stability") ---
    # Column in the data holding group labels (compartments or samples) used to
    # measure how much each marker's slope varies across biology. Large-n-robust.
    stability_group_col: Optional[str] = None
    min_group_cells: int = 50               # groups smaller than this are ignored


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
    gamma_mode: str = "per_marker",
    shrink_target: str = "control",
    control_markers: Optional[Sequence[str]] = None,
    protect_covariates: Optional[Sequence[str]] = None,
    stability_groups: Optional[pd.Series] = None,
    min_group_cells: int = 50,
) -> CytofTransformResult:
    """
    For each marker, regress arcsinh intensities on a 1D technical factor (PC1),
    then subtract γ * (f - anchor) to obtain corrected values. Optionally z-score.

    If line_labels is provided, gamma is estimated using a balanced subset
    with (approximately) the same number of cells per line, and then applied
    to all cells.

    gamma_mode controls how the per-marker slope γ_m is turned into the slope
    that is actually applied:
      "per_marker" : independent OLS slope per marker (original behaviour).
      "single"     : one shared slope γ̄ applied to every marker (pure uniform
                     multiplicative-permeabilization model).
      "shrink"     : empirical-Bayes partial pooling toward γ̄:
                         γ_used_m = γ̄ + B_m * (γ_ols_m - γ̄),
                         B_m = τ² / (τ² + SE_m²),
                     where SE_m is the standard error of the marker's slope and
                     τ² is the (method-of-moments) between-marker variance of the
                     true slopes. Markers with sharp, distinctive slopes keep
                     their own γ (B_m→1); noisy/indistinct ones are pooled to γ̄
                     (B_m→0). NOTE: with very many cells SE_m→0 and B_m→1, so this
                     mode effectively turns off at large n — use "shrink_stability"
                     there.
      "shrink_stability" : same shrinkage formula, but the trust denominator is the
                     *cross-group instability* of the slope, s_m², instead of SE_m².
                     γ_m is re-estimated within each group of `stability_groups`
                     (e.g. compartments or samples); s_m² = between-group variance
                     of those slopes minus the mean within-group sampling variance
                     (floored at 0). B_m = τ² / (τ² + s_m²). A marker whose slope is
                     the same in every group is a stable technical relationship and
                     keeps its γ; a marker whose slope swings across groups (biology
                     riding on f) is pooled to γ̄. Unlike SE, s_m² does NOT vanish
                     with n, so this is the large-n-robust shrinkage. Requires
                     `stability_groups`.

    shrink_target selects γ̄ (used by "single" and "shrink"):
      "control" : γ̄ = mean OLS slope over control_markers (calibrate the shared
                  technical slope on the core histones we trust as ~pure technical).
      "global"  : γ̄ = inverse-variance-weighted mean slope over markers_to_correct.

    Note: absolute slopes are directly comparable here because the correction lives
    in arcsinh/log space, where a permeabilization gain is the same additive shift
    on every marker; γ_m ≈ (antibody sensitivity) × (a common 1/scale factor).

    protect_covariates:
        Optional biological covariates (e.g. cell-cycle markers like IdU, CyclinB1,
        pRb, DNA). Each marker is fit on [f, covariates...] jointly, but ONLY the
        f term is subtracted — the covariates are kept. This removes the technical
        factor *conditional on* the covariates, so γ is de-biased against biology
        that correlates with f (proliferation / DNA content), and that biology is
        preserved in the output. Only covariates actually present as columns are
        used, so a panel missing some of them still runs (the rest are skipped).
        A marker that is itself a covariate is fit on f + the OTHER covariates.
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

    gamma: Dict[str, float] = {}      # slope actually applied per marker
    alpha: Dict[str, float] = {}
    gamma_ols: Dict[str, float] = {}  # f-slope from the (covariate-adjusted) fit
    se: Dict[str, float] = {}         # standard error of that slope

    markers = list(markers_to_correct)
    n_fit = len(balanced_idx)

    # ---- Protected covariates: included in the fit but NOT removed ----
    # Only those actually present in the data are used, so a panel missing some
    # cell-cycle markers still runs. f is always design column 0.
    protect = [c for c in (protect_covariates or []) if c in ddf.columns]
    if protect_covariates:
        missing_cov = [c for c in protect_covariates if c not in ddf.columns]
        print(f"[CyTOF-transform] Protected covariates (kept, not removed): {protect}"
              + (f"  | skipped (not in panel): {missing_cov}" if missing_cov else ""))

    for m in markers:
        if m not in ddf.columns:
            raise ValueError(f"Marker '{m}' not found in asinh_data columns.")

    # Pre-extract covariate matrix once (full data); slices are taken per index set.
    C_all = ddf[protect].values.astype(float) if protect else None

    def _slopes_on(idx):
        """Fit every marker's f-slope (covariate-adjusted) on rows `idx`, centering
        f, covariates and y WITHIN idx. Returns (gamma dict, se dict). A degenerate
        group (no technical variation) yields gamma 0 and se inf for every marker."""
        floc = f_all[idx]
        fc = floc - floc.mean()
        if np.sum(fc ** 2) == 0:
            return ({m: 0.0 for m in markers}, {m: np.inf for m in markers})
        nloc = len(idx)
        if protect:
            Cloc = C_all[idx] - C_all[idx].mean(axis=0, keepdims=True)

        def design(exclude=None):
            cols = [fc.reshape(-1, 1)]
            if protect:
                for j, c in enumerate(protect):
                    if c == exclude:
                        continue
                    cols.append(Cloc[:, j:j + 1])
            return np.hstack(cols)

        X_common = design(None)
        g_out, se_out = {}, {}
        for m in markers:
            y = ddf[m].values[idx]
            yc = y - y.mean()
            X = design(exclude=m) if (protect and m in protect) else X_common
            XtX = X.T @ X
            if np.linalg.matrix_rank(XtX) < X.shape[1]:
                XtX = XtX + 1e-8 * np.eye(X.shape[1])
            XtX_inv = np.linalg.inv(XtX)
            beta = XtX_inv @ (X.T @ yc)
            resid = yc - X @ beta
            dof = max(1, nloc - X.shape[1] - 1)
            sigma2 = float(np.sum(resid ** 2) / dof)
            g_out[m] = float(beta[0])
            se_out[m] = float(np.sqrt(max(sigma2 * XtX_inv[0, 0], 0.0)))
        return g_out, se_out

    # ---- Pass 1: per-marker f-slope on the (line-balanced) fit subset ----
    gamma_ols, se = _slopes_on(balanced_idx)
    fbar = f.mean()
    for m in markers:
        alpha[m] = float(ddf[m].values[balanced_idx].mean() - gamma_ols[m] * fbar)

    # ---- Shared target slope γ̄ ----
    g_arr = np.array([gamma_ols[m] for m in markers], dtype=float)
    se_arr = np.array([se[m] for m in markers], dtype=float)

    def _inv_var_mean(vals, ses):
        w = 1.0 / np.maximum(ses ** 2, 1e-12)
        return float(np.sum(w * vals) / np.sum(w))

    ctrl = [m for m in (control_markers or []) if m in gamma_ols]
    if shrink_target == "control" and ctrl:
        gamma_bar = float(np.mean([gamma_ols[m] for m in ctrl]))
    else:
        gamma_bar = _inv_var_mean(g_arr, se_arr)

    # ---- Between-marker variance τ² (method of moments), floored at 0 ----
    if len(markers) > 1:
        tau2 = float(np.var(g_arr, ddof=1) - np.mean(se_arr ** 2))
    else:
        tau2 = 0.0
    tau2 = max(tau2, 0.0)

    # ---- Cross-group instability s_m² (only for gamma_mode="shrink_stability") ----
    # Re-estimate each marker's slope within each group, then s_m² = between-group
    # variance of those slopes minus mean within-group sampling variance (>=0).
    s2_inst: Dict[str, float] = {m: 0.0 for m in markers}
    n_groups_used = 0
    if gamma_mode == "shrink_stability":
        if stability_groups is None:
            raise ValueError(
                "gamma_mode='shrink_stability' requires stability_groups "
                "(a per-cell Series of compartment/sample labels)."
            )
        gvals = pd.Series(stability_groups).loc[ddf.index].values
        per_group_g = {m: [] for m in markers}
        per_group_se2 = {m: [] for m in markers}
        for gv in pd.unique(gvals):
            gidx = np.where(gvals == gv)[0]
            if len(gidx) < min_group_cells:
                continue
            gg, gse = _slopes_on(gidx)
            for m in markers:
                if np.isfinite(gg[m]) and np.isfinite(gse[m]):
                    per_group_g[m].append(gg[m])
                    per_group_se2[m].append(gse[m] ** 2)
        counts = [len(per_group_g[m]) for m in markers]
        n_groups_used = max(counts) if counts else 0
        if n_groups_used < 2:
            print("[CyTOF-transform] WARNING: <2 usable stability groups "
                  f"(min_group_cells={min_group_cells}); shrink_stability falls back "
                  "to no shrinkage (s_m²=0, B_m=1).")
        for m in markers:
            arr = np.array(per_group_g[m], dtype=float)
            if len(arr) >= 2:
                between = float(np.var(arr, ddof=1))
                within = float(np.mean(per_group_se2[m]))
                s2_inst[m] = max(between - within, 0.0)
            else:
                s2_inst[m] = 0.0

    # ---- Turn OLS slopes into applied slopes according to gamma_mode ----
    B: Dict[str, float] = {}
    for m in markers:
        if gamma_mode == "per_marker":
            gamma[m] = gamma_ols[m]
            B[m] = 1.0
        elif gamma_mode == "single":
            gamma[m] = gamma_bar
            B[m] = 0.0
        elif gamma_mode == "shrink":
            denom_b = tau2 + se[m] ** 2
            b = (tau2 / denom_b) if denom_b > 0 else 0.0
            gamma[m] = gamma_bar + b * (gamma_ols[m] - gamma_bar)
            B[m] = float(b)
        elif gamma_mode == "shrink_stability":
            denom_b = tau2 + s2_inst[m]
            b = (tau2 / denom_b) if denom_b > 0 else 1.0
            gamma[m] = gamma_bar + b * (gamma_ols[m] - gamma_bar)
            B[m] = float(b)
        else:
            raise ValueError(
                f"Unknown gamma_mode '{gamma_mode}'. Use 'per_marker', 'single', "
                "'shrink', or 'shrink_stability'."
            )

    # ---- Apply correction to ALL cells (separate pass; reads original ddf) ----
    for m in markers:
        y_all = ddf[m].values
        ddf[m] = y_all - gamma[m] * (f_all - anchor)

    # ---- Per-marker diagnostics ----
    gamma_shrink = pd.DataFrame(
        {
            "gamma_ols": pd.Series(gamma_ols),
            "se": pd.Series(se),
            "s2_instability": pd.Series(s2_inst),
            "B": pd.Series(B),
            "gamma_used": pd.Series(gamma),
        }
    ).loc[markers]
    gamma_shrink.attrs["gamma_bar"] = gamma_bar
    gamma_shrink.attrs["tau2"] = tau2
    gamma_shrink.attrs["gamma_mode"] = gamma_mode
    gamma_shrink.attrs["shrink_target"] = shrink_target
    gamma_shrink.attrs["protect_covariates"] = list(protect)
    gamma_shrink.attrs["n_stability_groups"] = n_groups_used

    # ---- z-score if requested ----
    if zscore:
        residuals_z = ddf.copy()
        for m in markers:
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
        gamma_shrink=gamma_shrink,
    )


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
    sg = asinh_data[config.stability_group_col] if config.stability_group_col else None
    result = _regress_and_correct_1d(
        asinh_data=asinh_data,
        tech_factor=tech_factor,
        markers_to_correct=config.markers_to_correct,
        anchor_to_median=config.anchor_to_median,
        zscore=config.zscore,
        line_labels=ll,
        gamma_mode=config.gamma_mode,
        shrink_target=config.shrink_target,
        control_markers=config.control_markers,
        protect_covariates=config.protect_covariates,
        stability_groups=sg,
        min_group_cells=config.min_group_cells,
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
            gamma_mode=config.gamma_mode,
            shrink_target=config.shrink_target,
            control_markers=config.control_markers,
            protect_covariates=config.protect_covariates,
            stability_groups=(sub_data[config.stability_group_col]
                              if config.stability_group_col else None),
            min_group_cells=config.min_group_cells,
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


# ---------------------------------------------------------------------
# Legacy Method B1: std-minimizing permeabilization factor, DIVISION
# ---------------------------------------------------------------------

def _std_min_divide_weights(asinh_data, control_markers):
    """
    Convex weights over control markers that minimize the summed variance of the
    weighted-and-divided controls, then the per-cell factor M (weighted mean of
    each control normalized by its own mean). Native re-implementation of the
    std-minimizing permeabilization factor (scipy; no lmfit dependency).
    Returns (M: np.ndarray, weights: dict).
    """
    from scipy.optimize import minimize as _sp_minimize

    ctrl = list(control_markers)
    Q = asinh_data[ctrl].mean()
    norm_hist = asinh_data[ctrl].divide(Q, axis=1).values  # cells × k, each ~1
    k = norm_hist.shape[1]
    if k == 1:
        w = np.array([1.0])
    else:
        def obj(u):
            u = u - u.max()
            w = np.exp(u); w = w / w.sum()          # softmax -> non-neg, sums to 1
            M = norm_hist @ w
            d = norm_hist / M[:, None]              # divide controls by M
            return float(np.sum(d.std(axis=0) ** 2))  # sum of squared stds
        res = _sp_minimize(obj, np.zeros(k), method="Nelder-Mead")
        u = res.x - res.x.max()
        w = np.exp(u); w = w / w.sum()
    M = norm_hist @ w
    return M, dict(zip(ctrl, w))


def _cytof_divide_b1(asinh_data, config):
    """
    Method B1 (backwards compatibility): compute the std-minimizing permeabilization
    factor M from control markers and DIVIDE markers_to_correct by it. Reproduces the
    legacy CyTOFHelper.normalize_data when that module is importable; otherwise falls
    back to a native implementation. Returns a CytofTransformResult.

    SCALE: division is multiplicative and belongs on RAW (linear, pre-arcsinh) counts.
      - If config.arcsinh_cofactor is set: input is treated as RAW; markers are divided
        by M and the result is arcsinh(x / cofactor) — the correct raw -> divide ->
        arcsinh pipeline, output on the arcsinh scale.
      - If None: the divide is applied to the input as-is (exact legacy reproduction).
        Do NOT pass arcsinh data here without a cofactor: dividing already-arcsinh
        values is a misuse that distorts distributions.
    """
    ddf = asinh_data.copy()
    ctrl = list(config.control_markers)
    markers = list(config.markers_to_correct)

    M, weights = _std_min_divide_weights(ddf, ctrl)  # per-cell factor + weights (on input scale)
    try:
        from CyTOFHelper import normalize_data  # exact legacy behavior if available
        corrected = normalize_data(ddf, norm_columns=ctrl, norm_markers=markers)
    except Exception as e:
        print(f"[CyTOF-normalize] legacy CyTOFHelper.normalize_data unavailable "
              f"({type(e).__name__}); using native std-min divide.")
        corrected = ddf.copy()
        corrected[markers] = ddf[markers].divide(M, axis=0)

    # Correct pipeline: if a cofactor is given, the input was RAW -> arcsinh after divide.
    if config.arcsinh_cofactor is not None:
        num_cols = corrected.select_dtypes(include=[np.number]).columns
        corrected[num_cols] = np.arcsinh(corrected[num_cols] / config.arcsinh_cofactor)

    M_series = pd.Series(M, index=ddf.index, name="permeabilization_factor")
    scale = f"raw->divide->arcsinh(/{config.arcsinh_cofactor})" if config.arcsinh_cofactor else "divide on input scale (legacy)"
    print(f"[CyTOF-normalize] method='divide' (B1); {scale}. weights={weights}")

    residuals_z = corrected.copy()
    if config.zscore:
        for m in markers:
            v = residuals_z[m].values
            s = v.std() or 1.0
            residuals_z[m] = (v - v.mean()) / s

    res = CytofTransformResult(
        corrected=corrected,
        residuals_z=residuals_z,
        tech_factor=M_series,
        gamma={},
        alpha={},
        pca_model=None,
        gamma_shrink=None,
    )
    return res


# ---------------------------------------------------------------------
# Unified entry point: pick the normalization family via config.method
# ---------------------------------------------------------------------

def cytof_normalize(asinh_data, config, compartments=None):
    """
    Single entry point for all normalization families. Dispatches on config.method:

      "regress" (default) : sctransform-inspired PC1 regression (Method A). If
                            `compartments` is given, runs cytof_transform_by_compartment;
                            otherwise cytof_transform_global. Honors gamma_mode,
                            protect_covariates, shrink/stability options, etc.
      "divide"            : legacy std-minimizing permeabilization DIVISION (Method B1),
                            for backwards compatibility. `compartments` is ignored.
                            SCALE: division belongs on RAW (pre-arcsinh) counts. Pass RAW
                            data and set config.arcsinh_cofactor to run the correct
                            raw -> divide -> arcsinh pipeline. (With no cofactor it
                            divides the input as-is; don't hand it arcsinh data.)

    Note: "regress" expects arcsinh-transformed input; "divide" expects RAW input
    (with arcsinh_cofactor). Returns a CytofTransformResult in every case.
    """
    method = getattr(config, "method", "regress")
    if method == "regress":
        if compartments is not None:
            return cytof_transform_by_compartment(asinh_data, compartments, config)
        return cytof_transform_global(asinh_data, config)
    elif method == "divide":
        return _cytof_divide_b1(asinh_data, config)
    else:
        raise ValueError(f"Unknown config.method '{method}'. Use 'regress' or 'divide'.")


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
    if not _UMAP_AVAILABLE:
        raise ImportError(
            "umap-learn is required for plot_umap_qc. "
            "Install it with:  pip install umap-learn"
        )

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