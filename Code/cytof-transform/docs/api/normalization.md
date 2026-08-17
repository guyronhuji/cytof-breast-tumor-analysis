# Normalization

## cytof_normalize

Unified entry point. Dispatches on `CytofTransformConfig.method`:

- `"regress"` (default) — sctransform-inspired PC1 regression (Method A).
- `"divide"` — legacy std-minimizing permeabilization division (Method B1),
  kept for backwards compatibility and reproducing older results.

!!! warning "Input scale differs by method"
    `"regress"` expects **arcsinh-transformed** input. `"divide"` expects **RAW**
    (linear, pre-arcsinh) counts, since division is a multiplicative correction.
    Set `arcsinh_cofactor` so the pipeline runs raw &rarr; divide &rarr; arcsinh and
    returns data on the arcsinh scale.

::: cytof_transform.core.cytof_normalize

---

## cytof_transform_global

::: cytof_transform.core.cytof_transform_global

---

## cytof_transform_by_compartment

::: cytof_transform.core.cytof_transform_by_compartment
