"""Reproducible nested scaffold benchmark for three heavy-tail hypotheses."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from si_tail import (DESC, check_response, extract_chemberta_embeddings,
                     fit_outer_fold, scaffold_folds, score_predictions, smiles_features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--target-column", default="SI VACV")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--cache", default="work/chemberta_cache")
    parser.add_argument("--revision", help="Prefer the exact HF commit recorded by an earlier run")
    parser.add_argument("--skip-chemberta", action="store_true", help="Explicit classical-only ablation")
    parser.add_argument("--descriptor-transform", choices=["raw", "signedlog"], default="raw",
                        help="Stateless feature transform; never transforms target or clips predictions")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--low-threshold", type=float, default=100.0)
    parser.add_argument("--tail-threshold", type=float, default=10000.0)
    args = parser.parse_args()
    if args.outer_splits < 2 or args.inner_splits < 2 or args.batch_size < 1 or args.threads < 1:
        parser.error("Splits must be >=2; batch size and threads must be >=1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    # Strict allowlist. No test, concentration, missingness or assay columns read.
    data = pd.read_csv(args.train, usecols=[args.smiles_column, args.target_column])
    y = check_response(pd.to_numeric(data[args.target_column], errors="raise").to_numpy())
    features, keys, groups = smiles_features(data[args.smiles_column].tolist(), args.descriptor_transform)
    folds = scaffold_folds(groups, args.outer_splits, args.seed)
    protocol = {"args": vars(args), "train_sha256": hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
                "rows": len(y), "scaffold_groups": len(set(groups)),
                "canonical_structures": len(set(keys)), "columns_loaded": list(data.columns),
                "selection_metric": "unweighted pooled raw-scale inner scaffold OOF MSE",
                "evaluation_metric": "unweighted pooled raw-scale outer scaffold OOF MSE",
                "outer_folds": len(folds), "predictions_clipped": False,
                "external_training_rows": 0, "test_data_read": False,
                "feature_policy": "stateless RDKit and frozen ChemBERTa from SMILES only; all fitted transforms inside training folds",
                "group_policy": "historical generic BM; merge any groups sharing canonical tautomer identity; all acyclic in one group",
                "selection_caution": "All hypotheses prespecified; ranking outer scores is exploratory, not an independent estimate of the winner",
                "python": platform.python_version(), "descriptor_names": [name for name, _ in DESC]}
    try:
        protocol["base_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        protocol["base_commit"] = None
    protocol["versions"] = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "rdkit", "lightgbm", "torch", "transformers", "huggingface-hub"]:
        try:
            protocol["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    if not args.skip_chemberta:
        embeddings, embedding_metadata = extract_chemberta_embeddings(
            keys, cache_dir=args.cache, revision=args.revision, batch_size=args.batch_size,
            device=args.device, threads=args.threads)
        features["embeddings"] = embeddings
        features["combined"] = np.column_stack([features["descriptors"], embeddings])
        protocol["chemberta"] = embedding_metadata
        (out / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False))

    split_table = pd.DataFrame({"row_index": np.arange(len(y)), "source_row": np.arange(len(y)) + 2,
                                "canonical_smiles": keys, "scaffold_group": groups, "fold": -1})
    predictions, fold_scores, diagnostics, inner_audit = {}, [], [], []
    with threadpool_limits(limits=args.threads):
        for fold, (train, valid) in enumerate(folds):
            if not set(np.array(keys)[train]).isdisjoint(np.array(keys)[valid]):
                raise AssertionError("Canonical molecule crosses outer scaffold fold")
            split_table.loc[valid, "fold"] = fold
            prediction, selected, inner = fit_outer_fold(
                {key: value[train] for key, value in features.items()}, y[train], groups[train],
                {key: value[valid] for key, value in features.items()},
                inner_splits=args.inner_splits, seed=args.seed + fold + 1)
            for name, p in prediction.items():
                predictions.setdefault(name, np.full(len(y), np.nan))[valid] = p
                fold_scores.append({"model": name, "fold": fold, "train_n": len(train), "valid_n": len(valid),
                                    "train_max": float(y[train].max()),
                                    **score_predictions(y[valid], p, low_threshold=args.low_threshold,
                                                        tail_threshold=args.tail_threshold)})
            diagnostics.append({"fold": fold, "models": selected})
            inner_audit.extend({"outer_fold": fold, "row_index": int(i), "inner_fold": int(inner[j]),
                                "scaffold_group": groups[i]} for j, i in enumerate(train))
            split_table.to_csv(out / "validation_splits.csv", index=False)
            pd.DataFrame(fold_scores).to_csv(out / "fold_metrics.csv", index=False)
            (out / "training_selections.json").write_text(json.dumps(diagnostics, indent=2))
            print(f"Finished outer scaffold fold {fold + 1}/{len(folds)}: train={len(train)}, valid={len(valid)}", flush=True)
    rows = [{"model": name, **score_predictions(y, p, low_threshold=args.low_threshold,
                                                 tail_threshold=args.tail_threshold)}
            for name, p in predictions.items()]
    table = pd.DataFrame(rows).sort_values("MSE")
    reference = next(row["MSE"] for row in rows if row["model"] == "reference_krr_raw")
    table["MSE_change_vs_krr_pct"] = 100 * (table.MSE / reference - 1)
    table.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame({"row_index": np.arange(len(y)), "source_row": np.arange(len(y)) + 2,
                  "observed": y, "fold": split_table.fold, **predictions}).to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(inner_audit).to_csv(out / "inner_splits.csv", index=False)
    # Threshold fixed before training: strictly above 10,000 is distinct from codes.
    extreme = np.flatnonzero(y > args.tail_threshold)
    extreme_rows = []
    for index in extreme:
        fold = int(split_table.loc[index, "fold"])
        train = folds[fold][0]
        for name, p in predictions.items():
            gap = float(y[index] - p[index])
            extreme_rows.append({"model": name, "row_index": int(index), "source_row": int(index + 2),
                                 "fold": fold, "observed": float(y[index]), "prediction": float(p[index]),
                                 "gap": gap, "abs_gap": abs(gap), "squared_error": gap ** 2,
                                 "train_max": float(y[train].max()),
                                 "fraction_of_total_SSE": gap ** 2 / max(float(np.sum((p - y) ** 2)), 1e-300)})
    pd.DataFrame(extreme_rows, columns=["model", "row_index", "source_row", "fold", "observed", "prediction",
                                      "gap", "abs_gap", "squared_error", "train_max", "fraction_of_total_SSE"]
                 ).to_csv(out / "extreme_predictions.csv", index=False)
    protocol["completed"] = True
    protocol["outer_scaffold_overlap"] = 0
    protocol["outer_canonical_structure_overlap"] = 0
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    print(table[["model", "MSE", "low_MSE", "tail_MSE", "MSE_change_vs_krr_pct"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
