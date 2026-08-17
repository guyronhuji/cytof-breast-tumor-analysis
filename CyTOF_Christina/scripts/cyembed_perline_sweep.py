"""Per-cell-line CyEmbed sweeps for cross-line archetype matching.

Each cell line is fitted INDEPENDENTLY (no sample offset -- there is only one sample per run,
so an offset term would be unidentifiable).

Scaling choice, which matters for everything downstream:
    The z-score scaler is fitted ONCE on all five lines (balanced, 5,000 cells per line) --
    the same scaler the joint analysis uses -- and then applied to each line's cells. Fitting a
    separate scaler per line would express every line's archetypes in its own private units and
    make cross-line profile comparison meaningless. With a shared scaler, an archetype vector from
    MCF7 and one from HCC70 live in the same 29-dimensional space and can be compared directly.

Usage:
    python cyembed_perline_sweep.py --line MCF7
    python cyembed_perline_sweep.py --line all
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np

CT_PATH = "/Users/ronguy/Dropbox/Work/CyTOF/Code/cytof-transform"
CE_PATH = "/Users/ronguy/Dropbox/Work/CyTOF/Experiments/CyEmbed"
for p in (CT_PATH, CE_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytofstandard import Project  # noqa: E402
from CyEmbed.data import (  # noqa: E402
    extract_matrix,
    fit_scaler,
    preprocess_array,
    split_train_val_indices,
)
from CyEmbed.train import build_sweep_configs, run_sweep  # noqa: E402

BASE = Path("/Users/ronguy/Dropbox/Work/CyTOF/CyTOF_Christina")
PERLINE_DIR = BASE / "outputs/cyembed_perline_sweep"

LINES = ["MDAMB468", "HCC70", "SUM149", "HCC1937", "MCF7"]
LINE_DISP = {
    "MDAMB468": "MDA-MB-468", "HCC70": "HCC70", "SUM149": "SUM149",
    "HCC1937": "HCC1937", "MCF7": "MCF7",
}
SPLIT_SEED = 42
K_RANGE = list(range(2, 13))
SEEDS = [42, 1, 2]

# Same architecture and optimisation settings as the joint analysis, minus the offset machinery.
BASE_CONFIG = dict(
    model_type="deterministic", decoder_type="factorized",
    d=16, hidden_dims=[64, 32], tau=1.0,
    epochs=1500, early_stopping=True, patience=20,
    min_delta=0.0, restore_best_weights=True,
    lr=1e-3, batch_size=2048, weight_decay=1e-5, dropout=0.0,
    logit_normalizer="entmax", entmax_alpha=1.5, grad_clip_norm=5.0,
    separation_mode="cosine_sq", balance_mode="l2_uniform",
    lambda_entropy=1e-3, lambda_sep=1e-3, lambda_balance=0.05,
    recon_loss_type="mse", device="cpu", deterministic=True,
    seed=SPLIT_SEED,
)


def build_all_lines():
    """Load all five lines and fit the SHARED scaler (identical to the joint analysis)."""
    adatas = {}
    for line in LINES:
        proj = Project.load(f"/Users/ronguy/Dropbox/Work/CyTOF/Projects/{line}_NormCompare")
        adata = proj.get_run(line).read_adata()
        bio = [m for m in adata.var_names if m not in ["H3", "H3.3", "H4"]]
        sub = ad.AnnData(
            X=adata[:, bio].layers["norm_divide"].copy(),
            obs=adata.obs[["cell_uuid", "line_id"]].copy(),
            var=adata[:, bio].var.copy(),
        )
        sub.obs["cell_line"] = LINE_DISP[line]
        adatas[line] = sub

    combined = ad.concat([adatas[l] for l in LINES], join="outer")
    bundle_all = extract_matrix(adata=combined, layer=None, sample_col="cell_line")
    scaler, _ = fit_scaler(
        bundle_all.X, mode="zscore",
        sample_ids=bundle_all.sample_ids, balanced_max_per_sample=5000,
    )
    return adatas, scaler, list(bundle_all.marker_names)


def sweep_line(line, adatas, scaler):
    sub = adatas[line]
    bundle = extract_matrix(adata=sub, layer=None, sample_col="cell_line")
    x = preprocess_array(bundle.X, scaler)          # shared scaler, not a per-line one
    train_idx, val_idx = split_train_val_indices(
        n_cells=len(x), val_fraction=0.2, seed=SPLIT_SEED, stratify_labels=None
    )
    out_dir = PERLINE_DIR / line

    # Clear empty dirs left by aborted runs; they silently make run_sweep skip a grid cell.
    if out_dir.exists():
        for d in out_dir.glob("run_*"):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    configs = build_sweep_configs({"K": K_RANGE, "seed": SEEDS})
    print(f"[{line}] {x.shape[0]:,} cells | {len(configs)} configs "
          f"(K={min(K_RANGE)}..{max(K_RANGE)} x seeds {SEEDS})", flush=True)

    run_sweep(
        x=x,
        marker_names=list(bundle.marker_names),
        cell_ids=list(bundle.cell_ids),
        output_root=out_dir,
        base_config=BASE_CONFIG,
        sweep_configs=configs,
        train_idx=train_idx,
        val_idx=val_idx,
        sample_ids=None,                 # single line -> no offset, nothing to stratify
        scaler_state=scaler.to_dict(),
    )
    print(f"[{line}] DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True, help="one of MDAMB468/HCC70/SUM149/HCC1937/MCF7, or 'all'")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="restrict to these seeds (lets one line be split across processes)")
    args = ap.parse_args()
    global SEEDS
    if args.seeds:
        SEEDS = list(args.seeds)

    adatas, scaler, markers = build_all_lines()
    print(f"[data] shared scaler fitted on all 5 lines; {len(markers)} markers", flush=True)

    targets = LINES if args.line == "all" else [args.line]
    for line in targets:
        sweep_line(line, adatas, scaler)
    print("[sweep] ALL REQUESTED LINES DONE", flush=True)


if __name__ == "__main__":
    main()
