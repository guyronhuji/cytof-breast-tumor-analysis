"""Per-line CyEmbed sweeps in PermCell SIGNATURE space (not marker space).

Each line is fitted independently on its raw (unsmoothed) PermCell Z-score matrix
(n_cells x 15 signatures) produced by scripts/permcell_scores.py.

Design mirrors scripts/cyembed_perline_sweep.py so the two are directly comparable:
  * no sample offset -- one line per run, so a per-sample intercept is unidentifiable
  * a SHARED scaler fitted across all five lines' Z matrices, so signature axes are on the same
    scale in every line and archetype profiles can be compared across lines afterwards
  * same architecture, same K grid, same seeds

Usage:
    python cyembed_signature_sweep.py --line all
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

CE_PATH = "/Users/ronguy/Dropbox/Work/CyTOF/Experiments/CyEmbed"
if CE_PATH not in sys.path:
    sys.path.insert(0, CE_PATH)

from CyEmbed.data import (  # noqa: E402
    extract_matrix,
    fit_scaler,
    preprocess_array,
    split_train_val_indices,
)
from CyEmbed.train import build_sweep_configs, run_sweep  # noqa: E402

BASE = Path("/Users/ronguy/Dropbox/Work/CyTOF/CyTOF_Christina")
SCORES = BASE / "outputs/permcell_scores"
SWEEP = BASE / "outputs/cyembed_signature_sweep"

LINES = ["MDAMB468", "HCC70", "SUM149", "HCC1937", "MCF7"]
LINE_DISP = {"MDAMB468": "MDA-MB-468", "HCC70": "HCC70", "SUM149": "SUM149",
             "HCC1937": "HCC1937", "MCF7": "MCF7"}
SPLIT_SEED = 42
K_RANGE = list(range(2, 13))
SEEDS = [42, 1, 2]

BASE_CONFIG = dict(
    model_type="deterministic", decoder_type="factorized",
    d=16, hidden_dims=[64, 32], tau=1.0,
    epochs=1500, early_stopping=True, patience=20,
    min_delta=0.0, restore_best_weights=True,
    lr=1e-3, batch_size=2048, weight_decay=1e-5, dropout=0.0,
    logit_normalizer="entmax", entmax_alpha=1.5, grad_clip_norm=5.0,
    separation_mode="cosine_sq", balance_mode="l2_uniform",
    lambda_entropy=1e-3, lambda_sep=1e-3, lambda_balance=0.05,
    recon_loss_type="mse", device="cpu", deterministic=True, seed=SPLIT_SEED,
)


def load_signature_matrices():
    """Load per-line Z matrices and fit one shared scaler across all of them."""
    zs = {line: pd.read_csv(SCORES / f"{line}_Z.csv") for line in LINES}
    sig_names = list(zs[LINES[0]].columns)
    for line, df in zs.items():
        if list(df.columns) != sig_names:
            raise ValueError(f"signature order mismatch in {line}")

    stacked = np.vstack([zs[l].to_numpy(np.float32) for l in LINES])
    sample_ids = np.concatenate([
        np.repeat(LINE_DISP[l], len(zs[l])) for l in LINES
    ])
    combined = ad.AnnData(
        X=stacked,
        obs=pd.DataFrame({"cell_line": sample_ids},
                         index=[str(i) for i in range(len(stacked))]),
        var=pd.DataFrame(index=sig_names),
    )
    bundle_all = extract_matrix(adata=combined, layer=None, sample_col="cell_line")
    scaler, _ = fit_scaler(bundle_all.X, mode="zscore",
                           sample_ids=bundle_all.sample_ids, balanced_max_per_sample=5000)
    return zs, sig_names, scaler


def sweep_line(line, zs, sig_names, scaler):
    df = zs[line]
    adata = ad.AnnData(
        X=df.to_numpy(np.float32),
        obs=pd.DataFrame({"cell_line": LINE_DISP[line]},
                         index=[f"{line}_{i}" for i in range(len(df))]),
        var=pd.DataFrame(index=sig_names),
    )
    bundle = extract_matrix(adata=adata, layer=None, sample_col="cell_line")
    x = preprocess_array(bundle.X, scaler)
    train_idx, val_idx = split_train_val_indices(
        n_cells=len(x), val_fraction=0.2, seed=SPLIT_SEED, stratify_labels=None
    )
    out_dir = SWEEP / line
    if out_dir.exists():
        for d in out_dir.glob("run_*"):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    configs = build_sweep_configs({"K": K_RANGE, "seed": SEEDS})
    print(f"[{line}] {x.shape[0]:,} cells x {x.shape[1]} signatures | {len(configs)} configs",
          flush=True)
    run_sweep(
        x=x, marker_names=list(bundle.marker_names), cell_ids=list(bundle.cell_ids),
        output_root=out_dir, base_config=BASE_CONFIG, sweep_configs=configs,
        train_idx=train_idx, val_idx=val_idx, sample_ids=None,
        scaler_state=scaler.to_dict(),
    )
    print(f"[{line}] DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="restrict to these seeds (lets one line be split across processes)")
    args = ap.parse_args()
    global SEEDS
    if args.seeds:
        SEEDS = list(args.seeds)
    zs, sig_names, scaler = load_signature_matrices()
    print(f"[data] {len(sig_names)} signatures, shared scaler across 5 lines", flush=True)
    targets = LINES if args.line == "all" else [args.line]
    for line in targets:
        sweep_line(line, zs, sig_names, scaler)
    print("[sweep] ALL REQUESTED LINES DONE", flush=True)


if __name__ == "__main__":
    main()
