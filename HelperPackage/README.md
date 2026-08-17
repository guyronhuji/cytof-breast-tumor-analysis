
# CyTOF Helper Package

A comprehensive Python package for CyTOF (Cytometry by Time-of-Flight) data analysis. This package provides a suite of tools for normalization, statistical analysis, visualization, interactive exploration, and data smoothing.

## Table of Contents

- [Installation](#installation)
- [API Reference](docs/API.md)
- [Modules Overview](#modules-overview)
    - [1. Normalization (`cytof_helper.normalization`)](#1-normalization-cytof_helpernormalization)
    - [2. Plotting (`cytof_helper.plotting`)](#2-plotting-cytof_helperplotting)
    - [3. Statistics (`cytof_helper.stats`)](#3-statistics-cytof_helperstats)
    - [4. Interactive Tools (`cytof_helper.interactive`)](#4-interactive-tools-cytof_helperinteractive)
    - [5. Smoothing (`cytof_helper.smoothing`)](#5-smoothing-cytof_helpersmoothing)
    - [6. Utilities (`cytof_helper.utils`)](#6-utilities-cytof_helperutils)
- [Requirements](#requirements)

---

## Installation

You can install the package directly from the source directory. It is recommended to install it in "editable" mode (`-e`) so that changes to the code are immediately reflected.

```bash
cd /path/to/HelperPackage
pip install -e .
```

---

## Modules Overview

### 1. Normalization (`cytof_helper.normalization`)

This module handles the correction of technical effects in CyTOF data, particularly using control markers (e.g., Histones, DNA, Iridium) to normalize signal intensities.

#### Key Functions

*   **`cytof_transform_by_compartment(asinh_data, compartments, config)`**
    *   **Description**: Performs normalization by dealing with each biological compartment (e.g., Immune, Epithelial) separately.
    
*   **`infer_compartments(asinh_data, config)`**
    *   **Description**: Heuristically assigns cells to broad compartments (Immune, Epithelial, Stromal) based on marker expression levels.

*   **`normalize_markers_optimization(data, norm_columns=['H3.3', 'H4', 'H3'], norm_markers=None)`**
    *   **Description**: Optimization-based normalization (NormMark). Optimizes weights for `norm_columns` to minimize their variance across the dataset, then applies this correction to `norm_markers`.
    *   **Arguments**:
        *   `norm_columns`: List of 1-4 markers to use as controls.
        *   `norm_markers`: List of markers to correct (defaults to all others).

#### Usage Example

```python
from cytof_helper.normalization import (
    cytof_transform_by_compartment, 
    infer_compartments, 
    CytofTransformConfig, 
    CompartmentGatingConfig,
    normalize_markers_optimization
)

# 1. Compartment-based normalization
result = cytof_transform_by_compartment(adata_df, compartments, norm_cfg)

# 2. Optimization-based normalization (NormMark)
corrected_df = normalize_markers_optimization(
    df, 
    norm_columns=['H3', 'H4', 'DNA'], 
    norm_markers=['pStat3', 'Ki67']
)
```

---

### 2. Plotting (`cytof_helper.plotting`)

A collection of visualization tools specifically tailored for CyTOF/single-cell data comparisons.

#### Key Functions

*   **`mean_dist(data1, data2, markers, ...)`** / **`mean_dist_resamp(...)`**
    *   **Description**: Plots the mean difference in marker intensity between two datasets, with optional bootstrapping.
    
*   **`delta_corr(data1, data2, markers)`**
    *   **Description**: Heatmap visualization of the *difference* in correlation matrices between two conditions.

*   **`wfall(shap_values)`**
    *   **Description**: SHAP waterfall plot.

*   **`draw_umap(data, ...)`** / **`umap_plot(...)`**
    *   **Description**: UMAP generation and overlay plotting.

---

### 3. Statistics (`cytof_helper.stats`)

Statistical tests, model evaluation, and permutation-based scoring.

#### Key Functions

*   **`perm_cell(adata, marker_sets, ...)`**
    *   **Description**: (Formerly `sipsic_like_scores_v3`) Computes permutation-based scores for defined marker sets (pathways/signatures) against a random background.
    *   **Arguments**:
        *   `adata`: AnnData object or similar matrix wrapper.
        *   `marker_sets`: Dictionary defining up/down markers for each signature.
        *   `n_perm`: Number of permutations.

*   **`evaluate_model_with_cv(model, X, y, ...)`**
    *   **Description**: Cross-validation of classifiers with ROC plotting.

*   **`predict_th(model, X, threshold=0.8)`**
    *   **Description**: Threshold-based prediction wrapper.

*   **`test_hetero_mult(...)`**
    *   **Description**: Heterogeneity testing using UMAP/Convex Hulls.

#### Usage Example

```python
from cytof_helper.stats import perm_cell, evaluate_model_with_cv

# Permutation-based scoring
scores, pvals, _ = perm_cell(
    adata, 
    marker_sets={'Cycle': ['Ki67', 'CyclinB1'], 'Apoptosis': ['cCaspase3']},
    n_perm=1000
)

# Model evaluation
metrics = evaluate_model_with_cv(model, X, y)
```

---

### 4. Interactive Tools (`cytof_helper.interactive`)

Widget-based tools for Jupyter notebooks.

#### Key Functions

*   **`InteractiveClusterLabeler`**
*   **`create_cutoff_interface(df)`**
*   **`ManualSelection(df)`**

---

### 5. Smoothing (`cytof_helper.smoothing`)

Algorithms for spatial or graph-based smoothing.

#### Key Functions

*   **`gaussian_smooth_all_torch(data, positions, bandwidth, ...)`**
    *   **Description**: Fast Gaussian smoothing using PyTorch (MPS/CUDA supported).

---

### 6. Utilities (`cytof_helper.utils`)

*   **`get_markers(marker_list, base_dir)`**: Marker categorization helper.

---

## Requirements

- Python >= 3.8
- `numpy`, `pandas`, `scipy`
- `matplotlib`, `seaborn`
- `scikit-learn`, `umap-learn`, `xgboost`
- `torch`
- `plotly`, `ipywidgets`
- `lmfit`
