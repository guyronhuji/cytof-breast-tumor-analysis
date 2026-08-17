"""JOINT CyEmbed sweep over PermCell signature scores (all five lines pooled).

Counterpart to three earlier analyses:
  * joint marker space          -> outputs/cyembed_joint_sweep        (29 markers, offset arms)
  * per-line signature space    -> outputs/cyembed_signature_sweep    (15 signatures, per line)
  * THIS: joint signature space -> outputs/cyembed_joint_signature_sweep

Two properties make this design clean:

1. PermCell Z is a PER-CELL statistic. `sipsic_like_scores_v3` row-centres each cell and builds its
   null by permuting markers within that cell, so a cell's score does not depend on which other
   cells are in the matrix. Verified empirically: joint scoring vs concatenating the per-line score
   files agree to max |diff| = 1.2e-06. The saved per-line Z files are therefore reused directly.

2. With all five lines pooled, `use_sample_offset` is identifiable again (5 samples, not 1), so both
   arms can be run:
       Arm A  use_sample_offset=False
       Arm B  use_sample_offset=True   -- per-line intercepts in SIGNATURE space

Usage:
    python cyembed_joint_signature_sweep.py --arm both --seeds 42 --k 4 5 6 7 8 9
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
SWEEP = BASE / "outputs/cyembed_joint_signature_sweep"

LINES = ["MDAMB468", "HCC70", "SUM149", "HCC1937", "MCF7"]
LINE_DISP = {"MDAMB468": "MDA-MB-468", "HCC70": "HCC70", "SUM149": "SUM149",
             "HCC1937": "HCC1937", "MCF7": "MCF7"}
SPLIT_SEED = 42
K_RANGE = list(range(4, 15))
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
    recon_loss_type="mse", device="cpu", deterministic=True,
    seed=SPLIT_SEED, n_samples=len(LINES),
)


def build_joint():
    """Pool the per-line PermCell Z matrices into one joint dataset."""
    zs = {l: pd.read_csv(SCORES / f"{l}_Z.csv") for l in LINES}
    sig_names = list(zs[LINES[0]].columns)
    for l, df in zs.items():
        if list(df.columns) != sig_names:
            raise ValueError(f"signature order mismatch in {l}")

    X = np.vstack([zs[l].to_numpy(np.float32) for l in LINES])
    lines = np.concatenate([np.repeat(LINE_DISP[l], len(zs[l])) for l in LINES])
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_line": lines}, index=[f"c{i}" for i in range(len(X))]),
        var=pd.DataFrame(index=sig_names),
    )
    bundle = extract_matrix(adata=adata, layer=None, sample_col="cell_line")
    scaler, _ = fit_scaler(bundle.X, mode="zscore",
                           sample_ids=bundle.sample_ids, balanced_max_per_sample=5000)
    x = preprocess_array(bundle.X, scaler)
    train_idx, val_idx = split_train_val_indices(
        n_cells=len(x), val_fraction=0.2, seed=SPLIT_SEED,
        stratify_labels=bundle.sample_ids,
    )
    return bundle, x, train_idx, val_idx, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="both", choices=["A", "B", "both"],
                    help="A = no offset, B = use_sample_offset=True")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--k", type=int, nargs="+", default=None)
    args = ap.parse_args()

    seeds = args.seeds or SEEDS
    ks = args.k or K_RANGE
    arms = {"A": [False], "B": [True], "both": [False, True]}[args.arm]

    bundle, x, train_idx, val_idx, scaler = build_joint()
    print(f"[data] joint {x.shape[0]:,} cells x {x.shape[1]} signatures | "
          f"train {len(train_idx):,} / val {len(val_idx):,}", flush=True)

    SWEEP.mkdir(parents=True, exist_ok=True)
    stale = [d for d in SWEEP.glob("run_*") if d.is_dir() and not any(d.iterdir())]
    for d in stale:
        d.rmdir()
    if stale:
        print(f"[sweep] cleared {len(stale)} empty run dir(s)", flush=True)

    configs = build_sweep_configs({"K": ks, "use_sample_offset": arms, "seed": seeds})
    print(f"[sweep] arms={arms} seeds={seeds} K={ks} -> {len(configs)} configs", flush=True)

    run_sweep(
        x=x, marker_names=list(bundle.marker_names), cell_ids=list(bundle.cell_ids),
        output_root=SWEEP, base_config=BASE_CONFIG, sweep_configs=configs,
        train_idx=train_idx, val_idx=val_idx, sample_ids=bundle.sample_ids,
        scaler_state=scaler.to_dict(),
    )
    print("[sweep] DONE", flush=True)


if __name__ == "__main__":
    main()
