"""SMILES-only features and nested scaffold models for an unclipped SI benchmark.

No function that fits a model accepts validation responses. All learned feature
transforms, mixture coefficients and hyperparameters are fitted on training data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from train_external_v3 import canonical
from train_model import DESC, featurize, standardize, tanimoto

CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MTR"


def check_response(y):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("SI must be a finite, nonnegative 1D response")
    return y


def finite_prediction(prediction):
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite prediction; no clipping is applied")
    return prediction


def inverse_log(prediction):
    # Overflow is a visible failure, never a reason to cap a large prediction.
    with np.errstate(over="raise", invalid="raise"):
        return finite_prediction(np.expm1(prediction))


def scaffold_folds(groups, n_splits=3, seed=2718):
    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("At least two training scaffold groups are required")
    folds = list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                 .split(np.zeros(len(groups)), groups=groups))
    for train, valid in folds:
        if not set(groups[train]).isdisjoint(groups[valid]):
            raise AssertionError("Scaffold overlap")
    return folds


class TanimotoRegressor(RegressorMixin, BaseEstimator):
    """KRR taking fingerprints, so all kernel slicing stays inside the model."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        self.X_fit_ = np.asarray(X, dtype=float).copy()
        self.model_ = KernelRidge(alpha=self.alpha, kernel="precomputed").fit(
            tanimoto(self.X_fit_, self.X_fit_), y, sample_weight=sample_weight)
        self.n_features_in_ = self.X_fit_.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return finite_prediction(self.model_.predict(tanimoto(np.asarray(X), self.X_fit_)))


class RawLogEnsemble(RegressorMixin, BaseEstimator):
    """Convex raw/log1p mixture with training-only scaffold OOF calibration.

    w_raw + w_log = 1, both nonnegative. The optimizer minimizes UNWEIGHTED
    raw-scale MSE. Training OOF score is diagnostic, not an outer CV estimate.
    Pipelines passed as base estimators refit imputation/scaling in every split.
    """

    def __init__(self, raw_estimator, log_estimator=None, inner_splits=3, seed=2718):
        self.raw_estimator = raw_estimator
        self.log_estimator = log_estimator
        self.inner_splits = inner_splits
        self.seed = seed

    def fit(self, X, y, groups):
        X, y, groups = np.asarray(X), check_response(y), np.asarray(groups)
        if not (len(X) == len(y) == len(groups)):
            raise ValueError("X, y and groups must have equal length")
        log_est = self.raw_estimator if self.log_estimator is None else self.log_estimator
        raw_oof, log_oof = np.empty(len(y)), np.empty(len(y))
        self.inner_fold_ = np.full(len(y), -1, dtype=int)
        for fold, (train, valid) in enumerate(scaffold_folds(groups, self.inner_splits, self.seed)):
            raw = clone(self.raw_estimator).fit(X[train], y[train])
            log = clone(log_est).fit(X[train], np.log1p(y[train]))
            raw_oof[valid] = finite_prediction(raw.predict(X[valid]))
            log_oof[valid] = inverse_log(log.predict(X[valid]))
            self.inner_fold_[valid] = fold
        delta = raw_oof - log_oof
        scale = max(float(np.mean(y ** 2)), 1.0)

        def objective(weight):
            error = log_oof + weight[0] * delta - y
            return np.mean(error ** 2) / scale

        def gradient(weight):
            return np.array([2 * np.mean((log_oof + weight[0] * delta - y) * delta) / scale])

        result = minimize(objective, x0=np.array([0.5]), jac=gradient,
                          bounds=[(0.0, 1.0)], method="L-BFGS-B",
                          options={"ftol": 1e-14, "gtol": 1e-12})
        if not result.success:
            raise RuntimeError(f"Mixture optimization failed: {result.message}")
        self.weights_ = np.array([float(result.x[0]), 1.0 - float(result.x[0])])
        self.inner_oof_ = np.column_stack([raw_oof, log_oof])
        self.inner_mse_ = float(mean_squared_error(y, self.inner_oof_ @ self.weights_))
        self.raw_model_ = clone(self.raw_estimator).fit(X, y)
        self.log_model_ = clone(log_est).fit(X, np.log1p(y))
        self.n_features_in_ = X.shape[1]
        return self

    def predict_components(self, X):
        check_is_fitted(self, "weights_")
        return np.column_stack([finite_prediction(self.raw_model_.predict(X)),
                                inverse_log(self.log_model_.predict(X))])

    def predict(self, X):
        return finite_prediction(self.predict_components(X) @ self.weights_)


def smiles_features(smiles, descriptor_transform="raw"):
    """Deterministic RDKit extraction; takes structures only, never responses.

    Retain historical generic BM grouping, merging groups if canonical tautomer
    identities would otherwise cross a split. Acyclic molecules share one group.
    """
    smiles = list(smiles)
    keys = [canonical(s, True)[0] for s in smiles]
    original_groups = [standardize(s)[1] for s in smiles]
    parents = {g: g for g in original_groups}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    seen = {}
    for key, group in zip(keys, original_groups):
        if key in seen:
            a, b = sorted([find(group), find(seen[key])])
            parents[b] = a
        seen[key] = group
    groups = np.array([find(g) for g in original_groups])
    fingerprints, descriptors = featurize(keys)
    if descriptor_transform == "signedlog":
        # Stateless stabilization of RDKit dynamic range; target and predictions
        # remain on their original scales. This does not cap feature magnitudes.
        descriptors = np.sign(descriptors) * np.log1p(np.abs(descriptors))
    elif descriptor_transform != "raw":
        raise ValueError(f"Unknown descriptor transform: {descriptor_transform}")
    return {"fingerprints": fingerprints, "descriptors": descriptors}, keys, groups


def masked_mean_pool(hidden, attention_mask, special_tokens_mask):
    """Exclude padding and BOS/EOS tokens from the frozen representation."""
    mask = attention_mask.bool() & ~special_tokens_mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("An input has no molecular tokens")
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1)


def extract_chemberta_embeddings(smiles, *, cache_dir, revision=None,
                                 batch_size=16, device="cpu", threads=2):
    """Frozen ChemBERTa mean embeddings from SMILES ONLY; no fine tuning.

    Resolve and record a concrete HF revision. Reuse a cache keyed by ordered
    structures, immutable model revision and pooling. Long inputs fail explicitly
    rather than silently truncating. No scaler/PCA is learned here.
    """
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModel, AutoTokenizer

    smiles = list(smiles)
    if not smiles:
        raise ValueError("At least one SMILES is required")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if revision is None:
        revision = HfApi().model_info(CHEMBERTA_MODEL).sha
    # Branch names are resolved once as well; a 40-digit commit works offline.
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        revision = HfApi().model_info(CHEMBERTA_MODEL, revision=revision).sha
    metadata = {"model": CHEMBERTA_MODEL, "revision": revision,
                "pooling": "last_hidden_mean_excluding_special_and_padding_v1",
                "structures_sha256": hashlib.sha256(json.dumps(smiles).encode()).hexdigest(),
                "rows": len(smiles), "frozen": True, "truncation": False}
    cache_key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cache_path = cache_dir / f"{cache_key}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            saved_metadata = json.loads(str(cached["metadata"]))
            embeddings = cached["embeddings"]
        if any(saved_metadata.get(k) != v for k, v in metadata.items()):
            raise ValueError("Embedding cache metadata mismatch")
        if embeddings.ndim != 2 or len(embeddings) != len(smiles):
            raise ValueError("Invalid cached embedding shape")
        return finite_prediction(embeddings), saved_metadata
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL, revision=revision,
                                               cache_dir=str(cache_dir / "weights"))
    model = AutoModel.from_pretrained(CHEMBERTA_MODEL, revision=revision,
                                     cache_dir=str(cache_dir / "weights"), add_pooling_layer=False)
    model.requires_grad_(False).eval().to(device)
    max_tokens = min(model.config.max_position_embeddings - 2, tokenizer.model_max_length)
    lengths = tokenizer(smiles, truncation=False, return_length=True)["length"]
    too_long = [i for i, length in enumerate(lengths) if length > max_tokens]
    if too_long:
        raise ValueError(f"SMILES exceed {max_tokens} tokens at indices {too_long}; no truncation")
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(smiles), batch_size):
            tokens = tokenizer(smiles[start:start + batch_size], return_tensors="pt",
                               padding=True, truncation=False, return_special_tokens_mask=True)
            special = tokens.pop("special_tokens_mask").to(device)
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            chunks.append(masked_mean_pool(hidden, tokens["attention_mask"], special)
                          .cpu().numpy().astype(np.float32))
            print(f"ChemBERTa {min(start + batch_size, len(smiles))}/{len(smiles)}", flush=True)
    embeddings = finite_prediction(np.concatenate(chunks))
    metadata.update({"dimensions": embeddings.shape[1], "max_observed_tokens": max(lengths),
                     "max_allowed_tokens": int(max_tokens)})
    np.savez_compressed(cache_path, embeddings=embeddings, metadata=json.dumps(metadata))
    return embeddings, metadata


def training_weights(y, mode="uniform", strength=0.0):
    """Mild target-dependent weighting; normalization uses this training set only."""
    y = check_response(y)
    if strength < 0:
        raise ValueError("Weight strength must be nonnegative")
    if mode == "uniform":
        return np.ones_like(y)
    if mode not in {"raw", "log"}:
        raise ValueError(f"Unknown weighting mode: {mode}")
    signal = y if mode == "raw" else np.log1p(y)
    scale = float(np.mean(signal))
    weights = 1.0 + strength * signal / scale if scale > 0 else np.ones_like(y)
    return weights / weights.mean()


def boosting(seed=2718):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(objective="regression_l2", n_estimators=350,
                         learning_rate=0.03, max_depth=8, num_leaves=31,
                         min_child_samples=12, reg_lambda=100.0, reg_alpha=1.0,
                         max_bin=63, n_jobs=2, random_state=seed,
                         deterministic=True, force_col_wise=True, verbosity=-1)


def ridge_pipeline(alpha):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=alpha))


def tuned_ridge(X, y, groups, X_valid, *, inner_splits=3, seed=2718):
    scores = []
    for alpha in [100.0, 1000.0, 10000.0]:
        oof = np.empty(len(y))
        for train, valid in scaffold_folds(groups, inner_splits, seed):
            model = ridge_pipeline(alpha).fit(X[train], y[train])
            oof[valid] = finite_prediction(model.predict(X[valid]))
        scores.append({"alpha": alpha, "inner_mse": float(mean_squared_error(y, oof))})
    selected = min(scores, key=lambda row: row["inner_mse"])
    model = ridge_pipeline(selected["alpha"]).fit(X, y)
    return finite_prediction(model.predict(X_valid)), {"selected": selected, "candidates": scores}


def fit_outer_fold(train_features, y_train, groups_train, valid_features, *,
                   inner_splits=3, seed=2718):
    """Training boundary: validation responses are deliberately absent from API."""
    y_train = check_response(y_train)
    predictions, diagnostics = {}, {}
    ensemble = RawLogEnsemble(TanimotoRegressor(alpha=1.0),
                              inner_splits=inner_splits, seed=seed).fit(
                                  train_features["fingerprints"], y_train, groups_train)
    components = ensemble.predict_components(valid_features["fingerprints"])
    predictions["mean_raw"] = np.full(len(components), y_train.mean())
    predictions["reference_krr_raw"] = components[:, 0]
    predictions["reference_krr_log1p"] = components[:, 1]
    predictions["H1_raw_log_ensemble"] = components @ ensemble.weights_
    diagnostics["H1_raw_log_ensemble"] = {"w_raw": float(ensemble.weights_[0]),
                                            "w_log": float(ensemble.weights_[1]),
                                            "inner_mse": ensemble.inner_mse_}
    for kind in ["descriptors", "embeddings", "combined"]:
        if kind not in train_features:
            continue
        name = f"H2_{kind}_ridge" if kind != "descriptors" else "control_descriptors_ridge"
        predictions[name], diagnostics[name] = tuned_ridge(
            train_features[kind], y_train, groups_train, valid_features[kind],
            inner_splits=inner_splits, seed=seed)
    if "combined" in train_features:
        pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), boosting(seed))
        predictions["H2_combined_lgbm"] = finite_prediction(pipeline.fit(
            train_features["combined"], y_train).predict(valid_features["combined"]))

    # Same classical features and model for the unweighted and weighted controls.
    X = np.column_stack([train_features["fingerprints"], train_features["descriptors"]])
    V = np.column_stack([valid_features["fingerprints"], valid_features["descriptors"]])
    policies = [("uniform", 0.0)] + [(mode, strength) for mode in ["log", "raw"]
                                    for strength in [0.5, 1.0, 2.0]]
    policy_scores = []
    for mode, strength in policies:
        oof = np.empty(len(y_train))
        for train, valid in scaffold_folds(groups_train, inner_splits, seed):
            weights = training_weights(y_train[train], mode, strength)
            model = boosting(seed).fit(X[train], y_train[train], sample_weight=weights)
            oof[valid] = finite_prediction(model.predict(X[valid]))
        policy_scores.append({"mode": mode, "strength": strength,
                              "inner_mse": float(mean_squared_error(y_train, oof))})
    selected = min(policy_scores, key=lambda row: row["inner_mse"])
    diagnostics["H3_nested_weighting"] = {"selected": selected, "candidates": policy_scores}
    # Fixed policies expose whether training weighting harms the small-SI region.
    fixed = {"control_lgbm_raw": ("uniform", 0.0),
             "H3_fixed_log_weight": ("log", 1.0),
             "H3_fixed_raw_weight": ("raw", 0.5),
             "H3_nested_weighting": (selected["mode"], selected["strength"])}
    fitted = {}
    for name, (mode, strength) in fixed.items():
        if (mode, strength) not in fitted:
            weights = training_weights(y_train, mode, strength)
            model = boosting(seed).fit(X, y_train, sample_weight=weights)
            fitted[mode, strength] = finite_prediction(model.predict(V))
        predictions[name] = fitted[mode, strength]
    return predictions, diagnostics, ensemble.inner_fold_


def score_predictions(y, prediction, *, low_threshold=100.0, tail_threshold=10000.0):
    """All metrics unweighted, including the low/tail diagnostic subsets."""
    y, prediction = check_response(y), finite_prediction(prediction)
    if prediction.shape != y.shape:
        raise ValueError("Prediction shape must match the 1D response")
    error = prediction - y
    low, tail = y <= low_threshold, y >= tail_threshold
    row = {"MSE": float(np.mean(error ** 2)), "MAE": float(np.mean(np.abs(error))),
           "negative_predictions": int((prediction < 0).sum()),
           "low_n": int(low.sum()), "tail_n": int(tail.sum()),
           "pred_max": float(prediction.max())}
    for name, mask in [("low", low), ("tail", tail)]:
        row[f"{name}_MSE"] = float(np.mean(error[mask] ** 2)) if mask.any() else None
        row[f"{name}_bias"] = float(error[mask].mean()) if mask.any() else None
    row["low_pred_gt_1000"] = int((prediction[low] > 1000).sum())
    row["low_pred_p95"] = float(np.quantile(prediction[low], 0.95)) if low.any() else None
    return row
