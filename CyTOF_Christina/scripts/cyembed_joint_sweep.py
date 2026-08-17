"""Extended CyEmbed sweep for the joint cell-line archetype analysis.

Mirrors the data preparation in CellLines_CyEmbed_JointAnalysis.ipynb exactly so that
run fingerprints match and completed runs are reused rather than retrained.

Usage:
    python cyembed_joint_sweep.py --stage grid  --seeds 42
    python cyembed_joint_sweep.py --stage grid  --seeds 1
    python cyembed_joint_sweep.py --stage arma  --seeds 42
    python cyembed_joint_sweep.py --stage sparsity --k 12 --seeds 42
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
SWEEP_DIR = BASE / "outputs/cyembed_joint_sweep"

LINES = ["MDAMB468", "HCC70", "SUM149", "HCC1937", "MCF7"]
LINE_DISP = {
    "MDAMB468": "MDA-MB-468",
    "HCC70": "HCC70",
    "SUM149": "SUM149",
    "HCC1937": "HCC1937",
    "MCF7": "MCF7",
}
SPLIT_SEED = 42

# Reproduces the fingerprints of the existing 22 runs exactly (verified against the
# on-disk run ids), so previously completed configurations are skipped by run_sweep.
BASE_CONFIG = dict(
    model_type="deterministic",
    decoder_type="factorized",
    d=16,
    hidden_dims=[64, 32],
    tau=1.0,
    epochs=1500,
    early_stopping=True,
    patience=20,
    min_delta=0.0,
    restore_best_weights=True,
    lr=1e-3,
    batch_size=2048,
    weight_decay=1e-5,
    dropout=0.0,
    logit_normalizer="entmax",
    entmax_alpha=1.5,
    grad_clip_norm=5.0,
    separation_mode="cosine_sq",
    balance_mode="l2_uniform",
    lambda_entropy=1e-3,
    lambda_sep=1e-3,
    lambda_balance=0.05,
    recon_loss_type="mse",
    device="cpu",
    deterministic=True,
    seed=SPLIT_SEED,
    n_samples=len(LINES),
)


def build_joint_data():
    """Identical to notebook sections 1-2."""
    adatas = []
    for line in LINES:
        proj = Project.load(f"/Users/ronguy/Dropbox/Work/CyTOF/Projects/{line}_NormCompare")
        adata = proj.get_run(line).read_adata()
        bio_markers = [m for m in adata.var_names if m not in ["H3", "H3.3", "H4"]]
        ad_sub = ad.AnnData(
            X=adata[:, bio_markers].layers["norm_divide"].copy(),
            obs=adata.obs[["cell_uuid", "line_id"]].copy(),
            var=adata[:, bio_markers].var.copy(),
        )
        ad_sub.obs["cell_line"] = LINE_DISP[line]
        adatas.append(ad_sub)

    combined = ad.concat(adatas, join="outer")
    bundle = extract_matrix(adata=combined, layer=None, sample_col="cell_line")
    scaler, _ = fit_scaler(
        bundle.X, mode="zscore", sample_ids=bundle.sample_ids, balanced_max_per_sample=5000
    )
    x_scaled = preprocess_array(bundle.X, scaler)
    train_idx, val_idx = split_train_val_indices(
        n_cells=len(x_scaled),
        val_fraction=0.2,
        seed=SPLIT_SEED,
        stratify_labels=bundle.sample_ids,
    )
    return bundle, x_scaled, train_idx, val_idx, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["grid", "arma", "sparsity"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--k", type=int, nargs="+", default=None)
    args = ap.parse_args()

    bundle, x_scaled, train_idx, val_idx, scaler = build_joint_data()
    print(f"[data] X={x_scaled.shape} train={len(train_idx):,} val={len(val_idx):,}", flush=True)

    if args.stage == "grid":
        # Arm B across the full K range, one grid per seed.
        grid = {
            "K": args.k or list(range(4, 15)),
            "use_sample_offset": [True],
            "seed": args.seeds,
        }
    elif args.stage == "arma":
        # Arm A extension so the arm comparison spans the same K range.
        grid = {
            "K": args.k or list(range(4, 15)),
            "use_sample_offset": [False],
            "seed": args.seeds,
        }
    else:
        # Sparsity variants at the selected K to address near-uniform archetype weights.
        grid = {
            "K": args.k,
            "use_sample_offset": [True],
            "entmax_alpha": [1.5, 2.0],
            "lambda_entropy": [1e-3, 1e-2],
            "seed": args.seeds,
        }

    # An empty run directory left by an aborted run makes run_sweep skip that configuration
    # ("found matching run directory but missing/invalid summary"), silently dropping a grid cell.
    # Safe under concurrency: a worker that is mid-run recreates its own directory when it saves.
    stale = [d for d in SWEEP_DIR.glob("run_*") if d.is_dir() and not any(d.iterdir())]
    for d in stale:
        d.rmdir()
    if stale:
        print(f"[sweep] cleared {len(stale)} empty run dir(s): "
              f"{', '.join(d.name[-10:] for d in stale)}", flush=True)

    configs = build_sweep_configs(grid)
    print(f"[sweep] stage={args.stage} {len(configs)} configs", flush=True)

    run_sweep(
        x=x_scaled,
        marker_names=list(bundle.marker_names),
        cell_ids=list(bundle.cell_ids),
        output_root=SWEEP_DIR,
        base_config=BASE_CONFIG,
        sweep_configs=configs,
        train_idx=train_idx,
        val_idx=val_idx,
        sample_ids=bundle.sample_ids,
        scaler_state=scaler.to_dict() if hasattr(scaler, "to_dict") else None,
    )
    print(f"[sweep] stage={args.stage} seeds={args.seeds} DONE", flush=True)


if __name__ == "__main__":
    main()
