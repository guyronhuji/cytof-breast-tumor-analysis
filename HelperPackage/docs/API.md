
# CyTOF Helper API Reference

## Table of Contents

- [cytof_helper.normalization](#cytof_helpernormalization)
- [cytof_helper.stats](#cytof_helperstats)
- [cytof_helper.plotting](#cytof_helperplotting)
- [cytof_helper.interactive](#cytof_helperinteractive)
- [cytof_helper.smoothing](#cytof_helpersmoothing)
- [cytof_helper.utils](#cytof_helperutils)

---

## cytof_helper.normalization

Tools for correcting technical variation in CyTOF data.

### `cytof_transform_by_compartment`
```python
def cytof_transform_by_compartment(
    asinh_data: pd.DataFrame, 
    compartments: pd.Series, 
    config: CytofTransformConfig
) -> CytofTransformResult
```
Performs multi-factor regression-based normalization separately within each biological compartment.
- **asinh_data**: Input dataframe (arcsinh-transformed).
- **compartments**: Series mapping each cell to a compartment (e.g., 'Internal', 'External').
- **config**: `CytofTransformConfig` object defining controls and markers to correct.
- **Returns**: `CytofTransformResult` containing corrected data, residuals, technical factors, and regression parameters.

### `cytof_transform_global`
```python
def cytof_transform_global(
    asinh_data: pd.DataFrame, 
    config: CytofTransformConfig
) -> CytofTransformResult
```
Runs the CyTOF-transform technical correction on the entire dataset (global mode), ignoring compartments.

### `normalize_multi_factor_by_compartment`
```python
def normalize_multi_factor_by_compartment(
    asinh_data: pd.DataFrame,
    compartments: pd.Series,
    control_markers: Sequence[str],
    markers_to_correct: Sequence[str],
    n_factors: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]
```
Corrects data by regressing out `n_factors` (principal components) of the control markers within each compartment.
- **Returns**: Tuple of (corrected_data, tech_factors_all).

### `normalize_markers_optimization` (NormMark)
```python
def normalize_markers_optimization(
    data: pd.DataFrame, 
    norm_columns: List[str] = ['H3.3', 'H4', 'H3'], 
    norm_markers: Optional[List[str]] = None
) -> pd.DataFrame
```
Normalization using optimization. Finds the weighted combination of `norm_columns` that minimizes the sum of squared standard deviations of the corrected data.
- **norm_columns**: List of control columns (1-4 cols).
- **norm_markers**: Columns to apply the correction to.

### `infer_compartments`
```python
def infer_compartments(
    asinh_data: pd.DataFrame, 
    cfg: CompartmentGatingConfig, 
    label_immune="immune", 
    label_epithelial="epithelial", 
    label_stromal="stromal", 
    label_other="other"
) -> pd.Series
```
Heuristic gating to assign cells to compartments based on marker expression.

---

## cytof_helper.stats

Statistical tools and metrics.

### `perm_cell`
```python
def perm_cell(
    adata,
    marker_sets: Dict[str, object],
    layer: str = "scaled",
    n_perm: int = 2000,
    seed: int = 0,
    two_sided: bool = False,
    abs_variant: bool = True,
    ...
)
```
(Formerly `sipsic_like_scores_v3`) Computes permutation-based enrichment scores for defined marker sets.
- **adata**: AnnData-like object or matrix.
- **marker_sets**: Dictionary of signatures (e.g., `{'Sig1': {'up': ['A'], 'down': ['B']}}`).
- **n_perm**: Number of permutations for background estimation.
- **Returns**: Tuple of DataFrames (Z-scores, P-values, Z-direction).

### `evaluate_model_with_cv`
```python
def evaluate_model_with_cv(
    model, X, y, cv=5, n_bootstraps=1000, alpha=0.95, 
    desired_specificity=0.85, plot_roc=True
) -> dict
```
Evaluates a classifier (e.g., XGBoost, LogReg) using Stratified K-Fold CV.
- **Returns**: Dictionary of metrics (AUC, Sensitivity, Specificity, Precision, etc.) with confidence intervals. Plots ROC curve if `plot_roc=True`.

### `predict_th`
```python
def predict_th(
    model, 
    X, 
    threshold=0.8, 
    default_value=-1
) -> np.ndarray
```
Predicts class labels but assigns `default_value` if the maximum class probability is below `threshold`.

### `test_hetero_mult`
```python
def test_hetero_mult(
    DB, 
    UMAPMRK, 
    min_dist=0.001, 
    n_neighbors=60
) -> list
```
Tests heterogeneity of cell lines/groups in UMAP space using Convex Hull volume and local density.

---

## cytof_helper.plotting

Visualization functions.

### `mean_dist` / `med_dist`
```python
def mean_dist(data1, data2, markers, title='', clr=['darkgreen','purple'])
def med_dist(data1, data2, markers, title='', clr=['darkgreen','purple'])
```
Bar plots showing the mean (or median) difference in marker expression between two populations (`data2 - data1`).

### `mean_dist_resamp`
```python
def mean_dist_resamp(
    data1, data2, markers, title='', 
    clr=['darkgreen','purple'], nsamp=10, f=0.5
)
```
Same as `mean_dist` but performs resampling (`nsamp` times with fraction `f`) to generate error bars.

### `delta_corr`
```python
def delta_corr(data1, data2, markers, title_sup='')
```
Plots a heatmap of the difference matrix: `Correlation(data2) - Correlation(data1)`.

### `plot_histograms_multi_df`
```python
def plot_histograms_multi_df(
    dataframes, columns, df_names=None, ncols=3, 
    figsize=(20, 20), colors=None, hist_kwargs=None
)
```
Plots histograms for multiple dataframes, with one panel per column/marker.
- **dataframes**: List of DataFrames.
- **columns**: List of columns to plot.
- **colors**: Optional list of colors (uses `distinctipy` if available).

### `wfall`
```python
def wfall(shap_values, max_display=10, show=True)
```
Generates a SHAP waterfall plot for a single observation.

### `draw_umap` / `umap_plot`
Wrappers for UMAP calculation and plotting.

---

## cytof_helper.interactive

Interactive Jupyter widgets.

### `InteractiveClusterLabeler`
Class for interactively labeling UMAP clusters.
- **`show()`**: Displays the widget.
- **`get_labels()`**: Returns the current labels as a Series.

### `create_cutoff_interface`
```python
def create_cutoff_interface(df, s=0.1)
```
Creates a UI with histograms and scatter plots for manually setting expression cutoffs.

### `ManualSelection`
```python
def ManualSelection(df, id_column="region_id", ...)
```
Lasso tool for manual point selection on a scatter plot.

---

## cytof_helper.smoothing

Smoothing algorithms.

### `gaussian_smooth_all_torch`
```python
def gaussian_smooth_all_torch(
    data: np.ndarray, 
    positions: np.ndarray, 
    bandwidth: float, 
    device: str = None
) -> np.ndarray
```
Performs Gaussian smoothing of marker expression data using spatial/UMAP coordinates. Supports `mps`, `cuda`, and `cpu`.

---

## cytof_helper.utils

### `get_markers`
```python
def get_markers(marker_list: List[str], marker_file: Optional[str] = None, base_dir: str = "/Users/ronguy/") -> Tuple
```
Categorizes a list of markers into (All, Epigenetic, Normalization, Cell Identity, Cell Cycle) using a reference file.
- **marker_file**: Optional path to the marker Excel file. If provided, overrides default location.


### `gate_cells`
```python
def gate_cells(
    data: pd.DataFrame, 
    gate_columns: List[str] = ['H3.3', 'H4', 'H3'], 
    threshold: float = 5.0,
    remove_outliers: bool = False,
    verbose: bool = True
) -> pd.DataFrame
```
Filters cells where control markers (`gate_columns`) are above `threshold`. Optionally removes outliers.
