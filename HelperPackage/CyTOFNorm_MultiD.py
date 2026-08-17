from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
hKWD={'element':'step','fill':False,'stat':'density'}
import random
import matplotlib.patches as mpatches

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass







# ---------------------------------------------------------------------------
# Data classes (optional, for configuration)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Compartment inference
# ---------------------------------------------------------------------------

def infer_compartments(
    asinh_data: pd.DataFrame,
    cfg: CompartmentGatingConfig,
    label_immune: str = "immune",
    label_epithelial: str = "epithelial",
    label_stromal: str = "stromal",
    label_other: str = "other",
) -> pd.Series:
    """
    Infer broad compartments (immune / epithelial / stromal / other) from a few markers.

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed intensities.
    cfg : CompartmentGatingConfig
        Gating configuration specifying key markers and thresholds or quantiles.
        Typical usage:
            cfg = CompartmentGatingConfig(
                cd45="CD45",
                epithelial_markers=["EpCAM", "Pan-KRT"],
                stromal_markers=["PDPN", "CD90"],
                # use quantile-based thresholds by leaving *_threshold=None
            )

    Returns
    -------
    compartments : Series
        Length = n_cells, index = asinh_data.index, values in
        {label_immune, label_epithelial, label_stromal, label_other}.
    """
    n_cells = asinh_data.shape[0]
    compartments = pd.Series(np.full(n_cells, label_other, dtype=object),
                             index=asinh_data.index)

    # helper to compute composite scores for marker sets
    def _composite_score(markers: Sequence[str]) -> Optional[pd.Series]:
        markers = [m for m in markers if m in asinh_data.columns]
        if not markers:
            return None
        return asinh_data[markers].mean(axis=1)

    # CD45 (immune)
    if cfg.cd45 is not None and cfg.cd45 in asinh_data.columns:
        cd45_vals = asinh_data[cfg.cd45]
        if cfg.cd45_threshold is None:
            thr_cd45 = cd45_vals.quantile(cfg.cd45_quantile)
        else:
            thr_cd45 = cfg.cd45_threshold
    else:
        cd45_vals = None
        thr_cd45 = None

    # epithelial composite
    epi_score = _composite_score(cfg.epithelial_markers)
    if epi_score is not None:
        if cfg.epithelial_threshold is None:
            thr_epi = epi_score.quantile(cfg.epithelial_quantile)
        else:
            thr_epi = cfg.epithelial_threshold
    else:
        thr_epi = None

    # stromal composite
    stromal_score = _composite_score(cfg.stromal_markers)
    if stromal_score is not None:
        if cfg.stromal_threshold is None:
            thr_stroma = stromal_score.quantile(cfg.stromal_quantile)
        else:
            thr_stroma = cfg.stromal_threshold
    else:
        thr_stroma = None

    # start with all "other"
    comp = np.full(n_cells, label_other, dtype=object)

    # epithelial: high epithelial composite, low/moderate CD45
    if epi_score is not None and thr_epi is not None:
        mask_epithigh = epi_score >= thr_epi
    else:
        mask_epithigh = np.zeros(n_cells, dtype=bool)

    # stromal: high stromal composite, low/moderate CD45
    if stromal_score is not None and thr_stroma is not None:
        mask_stromahigh = stromal_score >= thr_stroma
    else:
        mask_stromahigh = np.zeros(n_cells, dtype=bool)

    # immune: high CD45, not strongly epithelial or stromal
    if cd45_vals is not None and thr_cd45 is not None:
        mask_cd45high = cd45_vals >= thr_cd45
    else:
        mask_cd45high = np.zeros(n_cells, dtype=bool)

    # Assign labels; order matters slightly
    # 1. epithelial wins if clearly high epithelial signal
    comp[mask_epithigh] = label_epithelial
    # 2. stromal wins if high stromal, not already epithelial
    comp[mask_stromahigh & ~mask_epithigh] = label_stromal
    # 3. immune assigned where high CD45 and not epithelial or stromal
    comp[mask_cd45high & ~mask_epithigh & ~mask_stromahigh] = label_immune

    compartments[:] = comp
    return compartments


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
    Compute K (= n_factors) technical factors from control markers in a SINGLE compartment,
    using PCA on arcsinh-transformed data.

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed intensities, restricted to ONE compartment.
    control_markers : list of str
        Marker names (columns in asinh_data) used as technical controls,
        e.g. core histones + DNA.
    n_factors : int, default 2
        Number of technical factors (PCs) to compute.
    compartment_name : str, optional
        Used only for logging.

    Returns
    -------
    tech_factors : DataFrame
        Cells × n_factors matrix of technical factors (PC scores).
        Columns named ["tech1", "tech2", ...].
        Index matches asinh_data.index.
    pca_model : PCA or None
        Fitted PCA object (sklearn). None if degenerate (too few cells).
    """
    # sanity check
    missing = [m for m in control_markers if m not in asinh_data.columns]
    if missing:
        raise ValueError(
            f"Missing control markers in asinh_data for compartment "
            f"{compartment_name}: {missing}"
        )

    Yc = asinh_data[control_markers].copy()
    n_cells = Yc.shape[0]

    if n_cells == 0:
        raise ValueError(f"Compartment '{compartment_name}' has zero cells.")

    if n_cells <= n_factors:
        print(
            f"[WARN] Compartment '{compartment_name}' has only {n_cells} cells, "
            f"<= n_factors={n_factors}. Returning zero technical factors."
        )
        tech = np.zeros((n_cells, n_factors), dtype=float)
        cols = [f"tech{k+1}" for k in range(n_factors)]
        tech_factors = pd.DataFrame(tech, index=Yc.index, columns=cols)
        return tech_factors, None

    pca = PCA(n_components=n_factors)
    U = pca.fit_transform(Yc.values)  # shape (n_cells, n_factors)

    cols = [f"tech{k+1}" for k in range(n_factors)]
    tech_factors = pd.DataFrame(U, index=Yc.index, columns=cols)

    if compartment_name is not None:
        print(
            f"Computed {n_factors} technical factors for compartment '{compartment_name}' "
            f"using control markers: {list(control_markers)}"
        )

    return tech_factors, pca


def regress_out_technical_single_compartment(
    asinh_data: pd.DataFrame,
    tech_factors: pd.DataFrame,
    markers_to_correct: Sequence[str],
) -> pd.DataFrame:
    """
    Regress out multiple technical factors from selected markers in a SINGLE compartment.

    Model (per marker j):
        Y_j = alpha_j + sum_k beta_{jk} * U_k + epsilon
    Correction:
        Y_j_corr = Y_j - (U - median(U)) @ beta_j

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed, restricted to one compartment.
    tech_factors : DataFrame
        Cells × K technical factors, with same index as asinh_data.
    markers_to_correct : list of str
        Marker names whose dependence on technical factors we remove.

    Returns
    -------
    corrected : DataFrame
        Same shape as asinh_data, with only markers_to_correct modified.
    """
    ddf = asinh_data.copy()

    # align
    tech_factors = tech_factors.loc[ddf.index]
    U = tech_factors.values  # shape (n_cells, K)
    n_cells, K = U.shape

    if K == 0:
        # nothing to regress out
        return ddf

    # center U per factor
    U_mean = U.mean(axis=0, keepdims=True)
    U_centered = U - U_mean

    # precompute (U^T U)^{-1} once per compartment
    XtX = U_centered.T @ U_centered  # K × K
    if np.linalg.matrix_rank(XtX) < K:
        XtX = XtX + 1e-8 * np.eye(K)
    XtX_inv = np.linalg.inv(XtX)

    # medians (for anchoring correction)
    U_median = np.median(U, axis=0, keepdims=True)  # 1 × K

    for m in markers_to_correct:
        if m not in ddf.columns:
            raise ValueError(f"Marker '{m}' not found in asinh_data columns.")

        y = ddf[m].values  # length n_cells
        y_centered = y - y.mean()

        # beta_j = (U^T U)^(-1) U^T y_centered    → K-vector
        XtY = U_centered.T @ y_centered
        beta = XtX_inv @ XtY  # shape (K,)

        # technical contribution to subtract: (U - median(U)) @ beta
        tech_contrib = (U - U_median) @ beta  # N-vector

        ddf[m] = y - tech_contrib

    return ddf


# ---------------------------------------------------------------------------
# High-level multi-factor, compartment-aware normalization
# ---------------------------------------------------------------------------

def normalize_multi_factor_by_compartment(
    asinh_data: pd.DataFrame,
    compartments: pd.Series,
    control_markers: Sequence[str],
    markers_to_correct: Sequence[str],
    n_factors: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-factor, compartment-aware normalization of CyTOF arcsinh data.

    For each compartment:
      1. Compute K technical factors via PCA on control markers.
      2. Regress out these factors from markers_to_correct.

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed intensities.
    compartments : Series
        Length = n_cells, index matching asinh_data.index.
        Values: compartment labels (e.g. "immune", "tumor", "stromal").
    control_markers : list of str
        Markers used as technical controls (core histones, DNA, viability, ...).
    markers_to_correct : list of str
        Markers whose dependence on technical factors will be removed
        (typically intracellular / nuclear markers; can include control markers).
    n_factors : int, default 2
        Number of technical factors per compartment.

    Returns
    -------
    corrected_all : DataFrame
        Same shape as asinh_data, with selected markers corrected.
    tech_factors_all : DataFrame
        Cells × n_factors technical factors, stacked across compartments.
        Columns named ["tech1", ..., "techK"].
    """
    if not asinh_data.index.equals(compartments.index):
        raise ValueError("Index of asinh_data and compartments must match.")

    corrected_list = []
    tech_list = []

    for comp in compartments.unique():
        idx = compartments == comp
        sub_data = asinh_data.loc[idx]

        print(f"\n=== Normalizing compartment '{comp}' "
              f"({sub_data.shape[0]} cells) ===")

        # 1) compute technical factors for this compartment
        tech_factors, pca_model = compute_technical_factors_single_compartment(
            asinh_data=sub_data,
            control_markers=control_markers,
            n_factors=n_factors,
            compartment_name=str(comp),
        )

        tech_list.append(tech_factors)

        # 2) regress out technical factors from markers_to_correct
        corrected_sub = regress_out_technical_single_compartment(
            asinh_data=sub_data,
            tech_factors=tech_factors,
            markers_to_correct=markers_to_correct,
        )

        corrected_list.append(corrected_sub)

    # concatenate all compartments back together in original order
    corrected_all = pd.concat(corrected_list, axis=0)
    corrected_all = corrected_all.loc[asinh_data.index]  # ensure same order

    tech_factors_all = pd.concat(tech_list, axis=0)
    tech_factors_all = tech_factors_all.loc[asinh_data.index]

    print("\nMulti-factor normalization complete.")
    print(f"  control_markers        : {list(control_markers)}")
    print(f"  markers_to_correct     : {list(markers_to_correct)}")
    print(f"  n_factors per compartment: {n_factors}")

    return corrected_all, tech_factors_all



from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def cluster_compartments_by_markers(
    asinh_data: pd.DataFrame,
    clustering_markers: Sequence[str],
    n_clusters: int = 6,
    n_pcs: int = 10,
    random_state: int = 0,
) -> Tuple[pd.Series, PCA, KMeans]:
    """
    Cluster cells into unsupervised groups using a small set of key markers.

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed intensities.
    clustering_markers : list of str
        Marker names to use for clustering (e.g. CD45, EpCAM, Pan-KRT, PDPN, CD90, ...).
        These should be the markers you'd normally use to gate major compartments.
    n_clusters : int, default 6
        Number of clusters for k-means.
    n_pcs : int, default 10
        Number of principal components to use for clustering.
    random_state : int, default 0
        Random seed for k-means reproducibility.

    Returns
    -------
    cluster_labels : Series
        Length = n_cells, index = asinh_data.index, values = integer cluster IDs.
    pca_model : PCA
        Fitted PCA model (from sklearn.decomposition).
    kmeans_model : KMeans
        Fitted KMeans model (from sklearn.cluster).
    """
    # sanity check
    missing = [m for m in clustering_markers if m not in asinh_data.columns]
    if missing:
        raise ValueError(f"Missing clustering markers in asinh_data: {missing}")

    Y = asinh_data[clustering_markers].copy()

    # standardize per marker (zero mean, unit variance) for clustering
    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0).replace(0, 1.0)  # avoid division by zero
    Y_stdzd = (Y - Y_mean) / Y_std

    # PCA to reduce dimensionality
    max_pcs = min(n_pcs, Y_stdzd.shape[1], max(1, Y_stdzd.shape[0] - 1))
    pca = PCA(n_components=max_pcs)
    X_pcs = pca.fit_transform(Y_stdzd.values)  # cells × PCs

    # k-means clustering on PCs
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    cluster_ids = kmeans.fit_predict(X_pcs)  # length n_cells

    cluster_labels = pd.Series(
        cluster_ids,
        index=asinh_data.index,
        name="cluster",
    )

    print(
        f"Clustering-based compartments: {n_clusters} clusters "
        f"using markers {list(clustering_markers)}"
    )

    return cluster_labels, pca, kmeans


def map_clusters_to_compartments(
    cluster_labels: pd.Series,
    mapping: Dict[int, str],
    default_label: str = "other",
) -> pd.Series:
    """
    Map integer cluster IDs to high-level compartment labels.

    Parameters
    ----------
    cluster_labels : Series
        Length = n_cells, index = cell IDs, values = integer cluster IDs.
    mapping : dict
        Dictionary mapping cluster_id -> compartment label, e.g.:
            {0: "immune", 1: "epithelial", 2: "stromal", 3: "immune"}
    default_label : str, default "other"
        Label to assign to any cluster IDs not present in `mapping`.

    Returns
    -------
    compartments : Series
        Length = n_cells, index = cluster_labels.index, values = compartment labels.
    """
    compartments = cluster_labels.map(mapping).fillna(default_label)
    compartments.name = "compartment"
    return compartments




