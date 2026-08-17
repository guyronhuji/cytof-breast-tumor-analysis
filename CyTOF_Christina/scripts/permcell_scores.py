"""Compute RAW (unsmoothed) PermCell signature Z-scores for each cell line.

Deliberately calls `sipsic_like_scores_v3` directly rather than going through
`Run.run_permcell`:

  * run_permcell ALWAYS runs the Gaussian smoothing step first and requires a 2-D embedding.
    We only want the raw scores, so smoothing is wasted work.
  * More importantly, smoothing over a UMAP built from the same markers makes the scores
    spatially autocorrelated, while PermCell's null permutes markers rather than positions.
    Skipping smoothing removes that circularity entirely.

Input layer: `norm_divide` (the normalised layer used throughout this project), z-scored with a
scaler fitted ONCE across all five lines (balanced, 5,000 cells/line). The z-scoring is required,
not cosmetic: `sipsic_like_scores_v3` row-centres each cell across markers, which is only
meaningful when markers share a scale.

Outputs, per line, under outputs/permcell_scores/:
    {LINE}_Z.csv       raw signature Z-scores        (n_cells x n_signatures)
    {LINE}_P.csv       associated p-values
    {LINE}_Zdir.csv    sign(obs) * Z_abs
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HELPER = "/Users/ronguy/Dropbox/Work/CyTOF/HelperPackage"
CT_PATH = "/Users/ronguy/Dropbox/Work/CyTOF/Code/cytof-transform"
CE_PATH = "/Users/ronguy/Dropbox/Work/CyTOF/Experiments/CyEmbed"
HERE = str(Path(__file__).resolve().parent)
for p in (HELPER, CT_PATH, CE_PATH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import PermCell_Smooth as PCS  # noqa: E402
from permcell_signatures import SIGNATURES, validate  # noqa: E402
from cytofstandard import Project  # noqa: E402
from CyEmbed.data import extract_matrix, fit_scaler, preprocess_array  # noqa: E402

BASE = Path("/Users/ronguy/Dropbox/Work/CyTOF/CyTOF_Christina")
OUT = BASE / "outputs/permcell_scores"
OUT.mkdir(parents=True, exist_ok=True)

LINES = ["MDAMB468", "HCC70", "SUM149", "HCC1937", "MCF7"]
LINE_DISP = {"MDAMB468": "MDA-MB-468", "HCC70": "HCC70", "SUM149": "SUM149",
             "HCC1937": "HCC1937", "MCF7": "MCF7"}

PERM_PARAMS = dict(
    n_perm=2000, seed=0, exclude_set=True, two_sided=True, abs_variant=True,
    progress=False, normalize_set_weights="l2", use_sparse_W=False,
    prefer_permutation=True,   # keep True: the enumeration branch has an unimported `comb`
    perm_batch=512,
)


def load_lines():
    adatas = {}
    for line in LINES:
        a = Project.load(f"/Users/ronguy/Dropbox/Work/CyTOF/Projects/{line}_NormCompare") \
                   .get_run(line).read_adata()
        bio = [m for m in a.var_names if m not in ["H3", "H3.3", "H4"]]
        sub = ad.AnnData(X=a[:, bio].layers["norm_divide"].copy(),
                         obs=a.obs[["cell_uuid", "line_id"]].copy(),
                         var=a[:, bio].var.copy())
        sub.obs["cell_line"] = LINE_DISP[line]
        adatas[line] = sub
    combined = ad.concat([adatas[l] for l in LINES], join="outer")
    bundle = extract_matrix(adata=combined, layer=None, sample_col="cell_line")
    scaler, _ = fit_scaler(bundle.X, mode="zscore",
                           sample_ids=bundle.sample_ids, balanced_max_per_sample=5000)
    return adatas, scaler, list(bundle.marker_names)


def main():
    adatas, scaler, markers = load_lines()
    print(f"[data] shared scaler over 5 lines; {len(markers)} markers", flush=True)
    ok, problems = validate(markers, verbose=False)
    if not ok:
        raise SystemExit(f"signature validation failed: {problems}")
    print(f"[sig] {len(SIGNATURES)} signatures validated against the panel", flush=True)

    for line in LINES:
        sub = adatas[line]
        b = extract_matrix(adata=sub, layer=None, sample_col="cell_line")
        X = preprocess_array(b.X, scaler).astype(np.float32)
        tmp = ad.AnnData(X=X, obs=sub.obs.copy(), var=sub.var.copy())

        Z, P, Zabs, Pabs, Zdir = PCS.sipsic_like_scores_v3(
            adata=tmp, marker_sets=SIGNATURES, layer="X", **PERM_PARAMS
        )
        for name, df in [("Z", Z), ("P", P), ("Zdir", Zdir)]:
            df.to_csv(OUT / f"{line}_{name}.csv", index=False)
        print(f"[{line}] {X.shape[0]:,} cells -> Z {Z.shape}  saved", flush=True)

    print("[permcell] ALL LINES DONE", flush=True)


if __name__ == "__main__":
    main()
