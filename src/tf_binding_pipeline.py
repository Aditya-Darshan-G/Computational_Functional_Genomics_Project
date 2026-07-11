#!/usr/bin/env python3
"""
tf_binding_pipeline.py

TF binding prediction pipeline using Gradient Boosting
with configurable neighboring bin ATAC context features.

Use --neighbors to control how many bins either side to include:
    --neighbors 0  :  ATAC_binary only (baseline)
    --neighbors 1  :  adds atac_prev1, atac_next1, atac_window_sum (3-bin window)
    --neighbors 2  :  adds atac_prev1/2, atac_next1/2, atac_window_sum (5-bin window)

For CTCF and REST, FIMO features are always included regardless of --neighbors.

Usage (run one TF at a time to save memory):
    # ±1 neighbor
    python tf_binding_pipeline.py --neighbors 1 \
        --data-dir data \
        --train-chromosomes 1 2 4 5 6 7 8 9 11 12 13 14 15 16 18 19 20 21 22 \
        --predict-chromosomes 3 10 17 \
        --ctcf-fimo CTCF_fimo.tsv \
        --rest-fimo REST_fimo.tsv \
        --output-dir gb_predictions_n1 \
        --n-jobs 8 \
        --tf CTCF
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def bu_to_binary(val):
    return 1 if str(val).strip().upper() == "B" else 0


def safe_neglog10(val):
    try:
        v = float(val)
        if v <= 0 or math.isnan(v):
            return 0.0
        return -math.log10(v)
    except Exception:
        return 0.0


def resolve_njobs(n):
    cores = os.cpu_count() or 1
    if n == -1:
        return cores
    return max(1, min(n, cores))


def normalise_chr(series):
    s = series.astype(str).str.strip()
    mask = ~s.str.startswith("chr")
    s[mask] = "chr" + s[mask]
    return s


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_bins(paths, tf_name=None):
    """
    Load bin TSV files. Returns a DataFrame sorted by (chr, start)
    so that neighbor shifts are always positionally correct.
    TF_binary label is added if tf_name is given.
    """
    frames = [pd.read_csv(p, sep="\t") for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined["chr"] = normalise_chr(combined["chr"])
    combined["ATAC_binary"] = combined["ATAC"].apply(bu_to_binary)

    # Sort ONCE here so all downstream operations are on a consistent order
    combined = combined.sort_values(["chr", "start"]).reset_index(drop=True)

    if tf_name is not None:
        if tf_name not in combined.columns:
            raise ValueError(f"Column '{tf_name}' not found in TSV.")
        combined["TF_binary"] = combined[tf_name].apply(bu_to_binary)

    return combined


def load_fimo(fimo_path):
    if fimo_path is None:
        return None

    df = pd.read_csv(fimo_path, sep="\t", comment="#")
    df.columns = [c.strip() for c in df.columns]

    required = {"sequence_name", "start", "score", "p-value", "q-value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FIMO file missing columns: {missing}")

    df["chr"] = normalise_chr(df["sequence_name"])
    df["pos"] = df["start"].astype(int) - 1  # 0-based

    df = (
        df.groupby(["chr", "pos"], as_index=False)
        .agg(
            fimo_score=("score", "max"),
            fimo_pval=("p-value", "min"),
            fimo_qval=("q-value", "min"),
        )
    )
    df["fimo_neglog10_pval"] = df["fimo_pval"].apply(safe_neglog10)
    df["fimo_neglog10_qval"] = df["fimo_qval"].apply(safe_neglog10)
    df["fimo_hit"] = 1.0
    df = df.drop(columns=["fimo_pval", "fimo_qval"])
    return df


# ──────────────────────────────────────────────
# NEIGHBOR ATAC FEATURES
# ──────────────────────────────────────────────

def add_neighbor_atac(df, n_neighbors):
    """
    Add neighbor ATAC features using fast vectorised pandas shifts.

    n_neighbors controls how many bins either side to include:
        0 → no neighbor columns added
        1 → atac_prev1, atac_next1, atac_window_sum  (3-bin window)
        2 → atac_prev1/2, atac_next1/2, atac_window_sum  (5-bin window)

    KEY REQUIREMENT: df must already be sorted by (chr, start) before calling.
    groupby + shift ensures neighbors never cross chromosome boundaries.
    """
    if n_neighbors == 0:
        return df

    grp = df.groupby("chr", sort=False)["ATAC_binary"]
    window = df["ATAC_binary"].astype(float)

    for k in range(1, n_neighbors + 1):
        df[f"atac_prev{k}"] = grp.shift(k).fillna(0).astype(float)
        df[f"atac_next{k}"] = grp.shift(-k).fillna(0).astype(float)
        window = window + df[f"atac_prev{k}"] + df[f"atac_next{k}"]

    df["atac_window_sum"] = window
    return df


# ──────────────────────────────────────────────
# FEATURE BUILDING
# ──────────────────────────────────────────────

def build_features(bins_df, fimo_df, n_neighbors):
    """
    Build feature DataFrame from a pre-sorted bins_df.
    Row order is never changed — bins_df must be sorted by (chr, start).
    """
    result = bins_df[["chr", "start", "end", "ATAC_binary"]].copy()
    result["ATAC"] = bins_df["ATAC"].values

    # ── Neighbor ATAC first (before any merge that could reorder rows) ──
    result = add_neighbor_atac(result, n_neighbors)

    # ── FIMO features (left-merge preserves row order) ──
    if fimo_df is not None:
        fimo_work = fimo_df.copy()
        fimo_work["start"] = (fimo_work["pos"] // 200) * 200

        fimo_agg = (
            fimo_work.groupby(["chr", "start"], as_index=False)
            .agg(
                fimo_hit=("fimo_hit", "max"),
                fimo_score=("fimo_score", "max"),
                fimo_neglog10_pval=("fimo_neglog10_pval", "max"),
                fimo_neglog10_qval=("fimo_neglog10_qval", "max"),
            )
        )
        result = result.merge(fimo_agg, on=["chr", "start"], how="left")
        fimo_cols = ["fimo_hit", "fimo_score", "fimo_neglog10_pval", "fimo_neglog10_qval"]
        result[fimo_cols] = result[fimo_cols].fillna(0.0)
    else:
        result["fimo_hit"]           = 0.0
        result["fimo_score"]         = 0.0
        result["fimo_neglog10_pval"] = 0.0
        result["fimo_neglog10_qval"] = 0.0

    result = result.reset_index(drop=True)
    return result


def get_feature_cols(has_fimo, n_neighbors):
    """Return the feature column list for the given config."""
    cols = ["ATAC_binary"]
    for k in range(1, n_neighbors + 1):
        cols += [f"atac_prev{k}", f"atac_next{k}"]
    if n_neighbors > 0:
        cols.append("atac_window_sum")
    if has_fimo:
        cols += ["fimo_hit", "fimo_score", "fimo_neglog10_pval", "fimo_neglog10_qval"]
    return cols


# ──────────────────────────────────────────────
# TRAINING AND PREDICTION
# ──────────────────────────────────────────────

def train_model(feature_df, labels, tf_name, feature_cols, n_jobs):

    y = labels.values
    X = feature_df[feature_cols].values

    assert len(X) == len(y), \
        f"BUG: feature rows ({len(X)}) != label rows ({len(y)})"

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"  [{tf_name}] Samples: {len(y)} | Positives: {n_pos} | Negatives: {n_neg}")

    # GradientBoostingClassifier has no class_weight — compute sample weights
    weight_pos = (n_pos + n_neg) / (2 * n_pos) if n_pos > 0 else 1.0
    weight_neg = (n_pos + n_neg) / (2 * n_neg) if n_neg > 0 else 1.0
    sample_weights = np.where(y == 1, weight_pos, weight_neg)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.5,
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weights)

    train_probs = model.predict_proba(X)[:, 1]
    if len(np.unique(y)) > 1:
        auroc = roc_auc_score(y, train_probs)
        auprc = average_precision_score(y, train_probs)
        print(f"  [{tf_name}] Train AUROC: {auroc:.3f} | Train AUPRC: {auprc:.3f}")

    importances = dict(zip(feature_cols, model.feature_importances_))
    print(f"  [{tf_name}] Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {feat:<30} {imp:.4f}")

    return model, feature_cols


def predict_and_save(model, feature_cols, predict_feature_df, tf_name, output_dir):
    X_pred = predict_feature_df[feature_cols].values
    scores = model.predict_proba(X_pred)[:, 1]

    out_df = predict_feature_df[["chr", "start", "end", "ATAC"]].copy()
    out_df[tf_name] = scores

    out_path = Path(output_dir) / f"{tf_name}_predictions.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"  [{tf_name}] Predictions saved to: {out_path}")
    return out_df


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="GB TF binding pipeline with neighbor ATAC context features."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-chromosomes", nargs="+", type=int, required=True)
    parser.add_argument("--predict-chromosomes", nargs="+", type=int, required=True)
    parser.add_argument("--ctcf-fimo", default=None)
    parser.add_argument("--rest-fimo", default=None)
    parser.add_argument(
        "--neighbors", type=int, default=2, choices=[0, 1, 2],
        help="Number of neighboring bins each side (0=none, 1=±1, 2=±2). Default: 2"
    )
    parser.add_argument("--output-dir", default="gb_predictions_neighbors")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--tf", default=None, choices=["CTCF", "REST", "EP300"],
        help="Run only one TF. Omit to run all three."
    )
    return parser.parse_args()


def resolve_paths(data_dir, chromosomes, template):
    paths = []
    for c in chromosomes:
        p = Path(data_dir) / template.format(chrom=c)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        paths.append(str(p))
    return paths


def main():
    args = parse_args()
    n_jobs = resolve_njobs(args.n_jobs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using {n_jobs} CPU core(s).")
    print("Note: GradientBoosting training is single-threaded by design.\n")

    train_paths = resolve_paths(
        args.data_dir, args.train_chromosomes, "chr{chrom}_200bp_bins.tsv"
    )
    predict_paths = resolve_paths(
        args.data_dir, args.predict_chromosomes, "chr{chrom}_200bp_bins_unknown.tsv"
    )

    print(f"Training chromosomes  : {len(train_paths)}")
    print(f"Prediction chromosomes: {len(predict_paths)}")

    print("\nLoading FIMO files...")
    ctcf_fimo  = load_fimo(args.ctcf_fimo)
    rest_fimo  = load_fimo(args.rest_fimo)
    ep300_fimo = None

    if ctcf_fimo is not None:
        print(f"  CTCF FIMO hits: {len(ctcf_fimo)}")
    if rest_fimo is not None:
        print(f"  REST FIMO hits: {len(rest_fimo)}")
    print("  EP300: no FIMO, using ATAC + neighbors only")

    print("\nLoading prediction bins...")
    predict_bins_df = load_bins(predict_paths, tf_name=None)
    print(f"  Prediction bins: {len(predict_bins_df)}")

    all_tfs = [
        ("CTCF",  ctcf_fimo),
        ("REST",  rest_fimo),
        ("EP300", ep300_fimo),
    ]
    tfs = [(t, f) for t, f in all_tfs if args.tf is None or t == args.tf]

    n_neighbors = args.neighbors
    print(f"Neighbor window: ±{n_neighbors} bin(s)\n")

    all_predictions = None

    for tf_name, fimo_df in tfs:
        print(f"\n{'='*50}")
        print(f"Processing TF: {tf_name}")
        print(f"{'='*50}")

        print("  Loading training bins...")
        train_bins_df = load_bins(train_paths, tf_name=tf_name)
        print(f"  Training bins: {len(train_bins_df)}")

        print("  Building features (including neighbor ATAC)...")
        train_features_df = build_features(train_bins_df, fimo_df, n_neighbors)
        labels = train_bins_df["TF_binary"].reset_index(drop=True)

        predict_features_df = build_features(predict_bins_df, fimo_df, n_neighbors)

        feature_cols = get_feature_cols(fimo_df is not None, n_neighbors)
        print(f"  Features used: {feature_cols}")

        print("  Training model...")
        model, feature_cols = train_model(
            train_features_df, labels, tf_name, feature_cols, n_jobs
        )

        print("  Predicting...")
        pred_df = predict_and_save(
            model, feature_cols, predict_features_df, tf_name, output_dir
        )

        if all_predictions is None:
            all_predictions = pred_df
        else:
            all_predictions[tf_name] = pred_df[tf_name].values

    if all_predictions is not None:
        combined_path = output_dir / "all_tf_predictions.tsv"
        all_predictions.to_csv(combined_path, sep="\t", index=False)
        print(f"\nCombined predictions saved to: {combined_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
