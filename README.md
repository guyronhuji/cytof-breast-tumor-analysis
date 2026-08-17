# CyTOF analysis code

Jupyter notebooks and supporting packages for the single-cell epigenetic CyTOF
analysis in *Epigenetic Heterogeneity in Breast Tumors with Implications for
Disease Diagnosis and Monitoring* (Furth, Riemenschneider, Arieli et al.).

All notebooks are stored with outputs stripped. No measurement data, figures, or
generated files are included — see **Data** below for what must be supplied.

## Notebooks

| Path | Contents |
|---|---|
| `BRCA_SHAP/` | Tumour-cohort analysis: joint UMAP, cross-patient clustering, XGBoost classification and SHAP feature attribution, PermCell signature scoring, Vendi diversity scores. Also PermCell benchmarks on the public Levine13/Levine32/Samusik datasets. |
| `CyTOF_BreastCancer/Kaplan/` | Per-tumour preprocessing, UMAP + DBSCAN clustering of the primary tumour cohort; cell-cycle / pRb / Ki-67 analysis. |
| `CyTOF_BreastCancer/MCF7/`, `OrenMCF7/` | MCF7 parental cultures and single-cell-derived clones. |
| `CyTOF_BreastCancer/PDX/`, `PDXTam/` | Luminal PDX models (PDX-1, PDX-2) and the independently grown PDX-1 tumours. |
| `CyTOF_BreastCancer/VFACS/` | Cytokeratin-5-based in-silico gating ("virtual FACS") of tumours 5, 8 and 15. |
| `CyTOF_Christina/` | Breast cancer cell lines (MCF7, MDA-MB-468, HCC70, HCC1937, SUM149): core-histone normalization comparisons, single-γ normalization, joint embeddings, archetype and PermCell analysis. `scripts/` holds the sweep drivers these notebooks call. |
| `HelperPackage/` | Refactored end-to-end per-sample pipeline (load → arcsinh → core-histone normalization → UMAP → DBSCAN), plus the helper library itself (see below). |
| `Corr_EPINUC_CyTOF/` | Correlation of CyTOF measurements with EPINUC plasma measurements. |
| `Code/cytof-transform/examples/` | Quickstart for the ECDF Q–Q batch-correction transform. |

## Bundled packages

These are vendored here so the repository is self-contained. Upstream copies
live at:

| Package | Path here | Upstream |
|---|---|---|
| `cytof_transform` | `Code/cytof-transform/` | https://github.com/guyronhuji/cytof-transform |
| `cytofstandard` | `Code/CyTOFSTD/` | https://github.com/guyronhuji/CyTOFSTD |
| `CyEmbed` | `Experiments/CyEmbed/` | https://github.com/guyronhuji/CyEmbed |
| `CyTOFHelper`, `cytof_helper`, `PermCell_Smooth` | `HelperPackage/` (and `Code/CyTOFHelper.py`) | — |

Only source, docs and tests were copied; build artefacts, rendered
documentation sites, caches, git history, outputs and example datasets were
left out. To use git submodules instead of vendored copies, delete the three
directories above and run e.g.
`git submodule add https://github.com/guyronhuji/cytof-transform Code/cytof-transform`.

**Two non-code files are included because the library requires them:**
`HelperPackage/Markers_Names.xlsx` and `HelperPackage/Mapping.xlsx` are
marker-name lookup tables read by `CyTOFHelper.py` — reference metadata, not
measurements.

## Paths that must be edited before running

The notebooks resolve their imports by injecting absolute paths into
`sys.path`. Every one of these points at the original working tree and needs
repointing:

| Variable | Current value | Notebooks |
|---|---|---|
| `CT_PATH` | `.../Work/CyTOF/Code/cytof-transform` | 11 |
| `CE_PATH` | `.../Work/CyTOF/Experiments/CyEmbed` | 5 |
| `HELPER` | `.../Work/CyTOF/HelperPackage` | 2 |
| `SCRIPTS` | `.../Work/CyTOF/CyTOF_Christina/scripts` | 2 |

Note that `CT_PATH` is overloaded: in the `*-NormalizationComparison`,
`*-SingleGamma` and `VendiScore_*` notebooks it instead names a
`cytofstandard` **project directory** (`Work/CyTOF/Projects/{…}_NormCompare`,
`{…}_SingleGamma`, `{BRCA,PDX,CellLines}_VendiScore`). Those are run outputs and
are deliberately not included.

Third-party requirements: numpy, pandas, scipy, scikit-learn, xgboost, shap,
umap-learn, scanpy, anndata, matplotlib, seaborn, pyarrow, torch (for CyEmbed).

## Data

Raw CyTOF measurements are deposited at Zenodo, DOI `10.5281/zenodo.15336152`.

The notebooks currently read intermediate, batch-corrected and labelled files
rather than the raw deposition, at these absolute paths:

```
~/Dropbox/CyTOF_Breast/for_guy/normalized_not_scaled_{N}.parquet   batch-corrected cohort
~/Dropbox/CyTOF_Breast/data/guy/with_labels/cytoff/                cohort with subpopulation labels
~/Dropbox/CyTOF_Breast/Kaplan_1st/                                 primary tumour cohort
~/Dropbox/CyTOF_Breast/PDX/202403_lum_PDX/, PDX/20240425/          PDX models
~/Dropbox/CyTOF_Breast/CyTOF_CR7/CyTOF7_scMCF7/csv_scale_value/    single-cell MCF7 clones
~/Dropbox/CyTOF_Breast/data_revision_CR7_Nature/CyTOF1_BCcell-lines/csv_files/
~/Dropbox/CyTOF_Breast/202509_CyTOF-UCell/PermCell_on_existing_UMAPs/
~/Dropbox/CyTOF_Breast/BCK-virtual_FACS/data/
~/Dropbox/Breast-Cytof-EPINUC/Figure_source/Cytof_raw/
~/Dropbox/Work/CyTOF/Projects/{BRCA,PDX,CellLines}_VendiScore/, *_NormCompare/, *_SingleGamma/
```

These need repointing at a released data location before the notebooks are
reproducible by a third party.

## Methods implemented here

- **Batch correction** — quantile mapping between anchor samples via empirical
  cumulative distribution functions, with linear extrapolation outside the
  anchor range. (`cytof_transform`)
- **Normalization** — intracellular signals divided by a core-histone term
  `1 + αH3.3 + βH4 + γH3`, with coefficients chosen to minimise the summed
  variance of the core histones. (`cytof_transform`, `HelperPackage/CyTOFNorm_MultiD.py`)
- **Dimensionality reduction and clustering** — UMAP followed by DBSCAN;
  clusters merged by expression similarity into Cycling, luminal and
  basal-like subpopulations.
- **PermCell** — permutation-calibrated marker-program scoring against a
  size-matched competitive null drawn from the same cell's remaining markers;
  returns per-cell Z-scores. (`cytofstandard/run.py`,
  `HelperPackage/PermCell_Smooth.py`, `CyTOF_Christina/scripts/permcell_*.py`)
- **Classification** — XGBoost (learning rate 0.1, max depth 10, 1000
  estimators), with SHAP for feature attribution; also used to assign cells
  left unclustered by DBSCAN.
- **Vendi score** — epigenetic diversity per subpopulation, rarefied to equal
  class sizes.
