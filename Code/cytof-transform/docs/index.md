# cytof-transform

**CyTOF-transform** removes per-cell technical variation from mass-cytometry data while preserving biological signal — analogous to `sctransform` for single-cell RNA-seq.

## The problem

In CyTOF, each cell's marker intensities are affected by a cell-specific *technical factor*: cell size, permeabilisation efficiency, and other non-biological sources of variation. This inflates variance and biases downstream clustering and dimensionality reduction.

## The model

CyTOF-transform assumes a multiplicative model:

```
X_{i,m} ≈ Biology_{i,m} × T_i^{γ_m} × noise
```

In arcsinh / log space this becomes additive:

```
y_{i,m} ≈ α_m + γ_m × f_i + biology_{i,m}
```

- `f_i` — 1-D technical factor estimated as PC1 of core-histone and DNA control markers
- `γ_m` — marker-specific sensitivity to the technical factor, estimated by OLS regression

Correction subtracts `γ_m × (f_i − median(f))` from each cell, anchoring at the *median* cell so absolute intensities remain interpretable.

## Installation

```bash
pip install git+https://github.com/guyronhuji/cytof-transform.git
```

With UMAP support:

```bash
pip install "git+https://github.com/guyronhuji/cytof-transform.git#egg=cytof-transform[umap]"
```

## Quick start

```python
from cytof_transform import CytofTransformConfig, cytof_transform_global

config = CytofTransformConfig(
    control_markers=["H3", "H4", "H2A", "H2B", "DNA1", "DNA2"],
    markers_to_correct=["CD3", "CD4", "CD8a", "CD20", "CD45", "Ki67"],
    anchor_to_median=True,
    zscore=True,
)

result = cytof_transform_global(asinh_data, config)

# result.corrected    — corrected arcsinh values
# result.residuals_z  — z-scored (use for PCA / UMAP)
# result.tech_factor  — per-cell technical factor
# result.gamma        — per-marker γ slopes
```

See the [API Reference](api/index.md) for full documentation.
