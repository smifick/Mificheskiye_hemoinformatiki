"""Mean prediction, structure-dependent smearing and leakage-safe nested selection.

Public fitting interfaces receive training structures/targets/groups only.
There is no upper prediction cap or target winsorization in these models.
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import solve
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from si_tail import (TanimotoRegressor, boosting, check_response, finite_prediction,
                     scaffold_folds, smiles_features)
from train_model import tanimoto


def structures(smiles):
    features, keys, groups = smiles_features(smiles, 'signedlog')
    torsion = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
    feature = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=8192, includeChirality=True,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    mols = [Chem.MolFromSmiles(key) for key in keys]
    features['torsion'] = np.array([torsion.GetCountFingerprintAsNumPy(m) for m in mols], float)
    features['feature'] = np.array([feature.GetFingerprintAsNumPy(m) for m in mols], float)
    features['identity'] = np.asarray(keys, dtype=object)
    return features, keys, groups


def subset(features, indices):
    return {key: value[indices] for key, value in features.items()}


def minmax(a, b):
    result = np.empty((len(a), len(b)))
    for i, row in enumerate(a):
        denominator = np.maximum(row, b).sum(axis=1)
        result[i] = np.divide(np.minimum(row, b).sum(axis=1), denominator,
                              out=np.zeros(len(b)), where=denominator > 0)
    return result


def _direct_kernel(a, b):
    return .5 * (tanimoto(a['fingerprints'], b['fingerprints']) + minmax(a['torsion'], b['torsion']))


_STRUCTURE_REGISTRY = {}


@lru_cache(maxsize=384)
def _cached_kernel(left, right):
    a = {name: np.array([_STRUCTURE_REGISTRY[key][j] for key in left])
         for j, name in enumerate(['fingerprints', 'torsion'])}
    b = {name: np.array([_STRUCTURE_REGISTRY[key][j] for key in right])
         for j, name in enumerate(['fingerprints', 'torsion'])}
    result = _direct_kernel(a, b)
    result.setflags(write=False)
    return result


def kernel(a, b):
    # Memoization contains deterministic SMILES similarities only, never targets,
    # fitted statistics or model parameters. It also works after model reload.
    if 'identity' not in a or 'identity' not in b:
        return _direct_kernel(a, b)
    for features in [a, b]:
        for i, key in enumerate(features['identity']):
            _STRUCTURE_REGISTRY.setdefault(key, (features['fingerprints'][i], features['torsion'][i]))
    return _cached_kernel(tuple(a['identity']), tuple(b['identity']))


def centered_fit(k, y, alpha, weights=None):
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    mean = float(np.average(y, weights=weights))
    dual = solve(k + alpha * np.diag(1 / weights), y - mean, assume_a='pos')
    return mean, dual


class MeanGLM:
    """Regularized log-link conditional mean, not OLS of a log-transformed target."""

    def __init__(self, power=1.5, alpha=10.0):
        self.power, self.alpha = power, alpha

    def fit(self, features, y, groups=None):
        y = check_response(y)
        # Shift supports Gamma with the two numerical zeros; no label is dropped.
        self.response_scale_ = float(np.mean(y + 1))
        self.pipeline_ = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                       StandardScaler(),
                                       TweedieRegressor(power=self.power, alpha=self.alpha, link='log',
                                                        max_iter=1000, tol=1e-7))
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            self.pipeline_.fit(features['descriptors'], (y + 1) / self.response_scale_)
        return self

    def predict(self, features):
        with np.errstate(over='raise', invalid='raise'):
            return finite_prediction(self.response_scale_ * self.pipeline_.predict(features['descriptors']) - 1)


class RegimeSmearing:
    """Known code regime + physical SI; optionally conditional cross-fitted smearing.

    This is not a learned extreme-value regime: the physical tail has too few
    examples. The gate models the known numerical code, using SMILES alone.
    """

    def __init__(self, correction='conditional', physical_alpha=3.0, prior_count=10.0,
                 inner_splits=3, seed=2718):
        self.correction, self.physical_alpha, self.prior_count = correction, physical_alpha, prior_count
        self.inner_splits, self.seed = inner_splits, seed

    def fit(self, features, y, groups, external=None):
        y, groups = check_response(y), np.asarray(groups)
        self.gate_reference_ = features
        self.zero_conditional_rate_ = float(np.mean(y[y != 10000] == 0))
        code = (y == 10000).astype(float)
        self.code_mean_, self.code_dual_ = centered_fit(kernel(features, features), code, .5)
        self.feature_mean_, self.feature_dual_ = centered_fit(tanimoto(features['feature'], features['feature']), code, .5)
        physical = np.flatnonzero((y > 0) & (y != 10000))
        if len(physical) < 3 or len(set(groups[physical])) < 2:
            raise ValueError('Insufficient physical observations/groups')
        own = subset(features, physical)
        if external is not None:
            external_features, external_y, external_groups, external_weight = external
            # Caller must first remove outer validation overlap. This additional
            # check forbids conflicting group aliases for a canonical structure.
            all_features = {key: np.concatenate([own[key], external_features[key]], axis=0) for key in own}
            all_y = np.r_[y[physical], external_y]
            weights = np.r_[np.ones(len(physical)), np.full(len(external_y), external_weight)]
        else:
            all_features, all_y, weights = own, y[physical], np.ones(len(physical))
        self.physical_reference_ = all_features
        self.physical_mean_, self.physical_dual_ = centered_fit(
            kernel(all_features, all_features), np.log1p(all_y), self.physical_alpha, weights)
        self.ratio_reference_ = own
        if self.correction == 'none':
            self.ratios_ = np.ones(len(physical))
        else:
            oof = np.full(len(physical), np.nan)
            for train, valid in scaffold_folds(groups[physical], self.inner_splits, self.seed):
                fit_features = subset(own, train)
                fit_y = y[physical[train]]
                fit_weights = np.ones(len(train))
                if external is not None:
                    safe = ~np.isin(external_groups, groups[physical[valid]])
                    safe &= ~np.isin(external_features['identity'], own['identity'][valid])
                    fit_features = {key: np.concatenate([fit_features[key], external_features[key][safe]], axis=0)
                                    for key in fit_features}
                    fit_y = np.r_[fit_y, external_y[safe]]
                    fit_weights = np.r_[fit_weights, np.full(int(safe.sum()), external_weight)]
                mean, dual = centered_fit(kernel(fit_features, fit_features), np.log1p(fit_y),
                                           self.physical_alpha, fit_weights)
                oof[valid] = mean + kernel(subset(own, valid), fit_features) @ dual
            with np.errstate(over='raise', invalid='raise'):
                self.ratios_ = finite_prediction(np.exp(np.log1p(y[physical]) - oof))
        self.global_factor_ = float(self.ratios_.mean())
        self.diagnostics_ = {'physical_n': len(physical), 'physical_max': float(y[physical].max()),
                             'global_factor': self.global_factor_, 'ratio_max': float(self.ratios_.max()),
                             'ratio_q95': float(np.quantile(self.ratios_, .95)),
                             'external_rows': len(all_y) - len(physical)}
        return self

    def predict_details(self, features):
        p0 = self.code_mean_ + kernel(features, self.gate_reference_) @ self.code_dual_
        p1 = self.feature_mean_ + tanimoto(features['feature'], self.gate_reference_['feature']) @ self.feature_dual_
        # Bounds on probabilities are distinct from caps on SI predictions.
        p_code = .5 * (np.clip(p0, 0, 1) + np.clip(p1, 0, 1))
        p_physical = (1 - p_code) * (1 - self.zero_conditional_rate_)
        physical_log = self.physical_mean_ + kernel(features, self.physical_reference_) @ self.physical_dual_
        if self.correction == 'conditional':
            similarities = kernel(features, self.ratio_reference_) ** 2
            factor = (similarities @ self.ratios_ + self.prior_count * self.global_factor_) / (
                similarities.sum(axis=1) + self.prior_count)
        elif self.correction in {'none', 'global'}:
            factor = np.full(len(p_code), self.global_factor_)
        else:
            raise ValueError(f'Unknown correction: {self.correction}')
        with np.errstate(over='raise', invalid='raise'):
            physical = finite_prediction(np.exp(physical_log) * factor - 1)
        prediction = finite_prediction(10000 * p_code + p_physical * physical)
        return prediction, {'p_code': p_code, 'physical_SI': physical,
                            'smearing_factor': factor, 'physical_log': physical_log}

    def predict(self, features):
        return self.predict_details(features)[0]


class PlainBaseline:
    def __init__(self, kind='lgbm', seed=2718):
        self.kind, self.seed = kind, seed

    def _matrix(self, features):
        return np.column_stack([features['fingerprints'], features['descriptors']])

    def fit(self, features, y, groups=None):
        if self.kind == 'krr':
            self.model_ = TanimotoRegressor().fit(features['fingerprints'], y)
        else:
            self.model_ = boosting(self.seed).fit(self._matrix(features), y)
        return self

    def predict(self, features):
        X = features['fingerprints'] if self.kind == 'krr' else self._matrix(features)
        return finite_prediction(self.model_.predict(X))


class NonnegativeProjection:
    """Known target support [0, infinity); no upper cap and no fitted threshold."""
    def __init__(self, model):
        self.model = model

    def predict(self, features):
        return np.maximum(self.model.predict(features), 0.0)


def candidate_specs():
    specs = [{'name': 'baseline_krr', 'family': 'baseline', 'kind': 'krr'},
             {'name': 'baseline_lgbm', 'family': 'baseline', 'kind': 'lgbm'}]
    specs += [{'name': f'glm_p{power:g}_a{alpha:g}', 'family': 'mean_glm', 'power': power, 'alpha': alpha}
              for power in [1.0, 1.5, 2.0] for alpha in [1.0, 10.0, 100.0]]
    for alpha in [3.0, 10.0]:
        for correction, prior in [('none', 10.0), ('global', 10.0), ('conditional', 5.0), ('conditional', 25.0)]:
            specs.append({'name': f'regime_{correction}_a{alpha:g}_prior{prior:g}', 'family': 'smearing',
                          'correction': correction, 'physical_alpha': alpha, 'prior_count': prior})
    return specs


def fit_spec(spec, features, y, groups, seed=2718, external=None):
    if spec['family'] == 'baseline':
        return PlainBaseline(spec['kind'], seed).fit(features, y, groups)
    if spec['family'] == 'mean_glm':
        return MeanGLM(spec['power'], spec['alpha']).fit(features, y, groups)
    return RegimeSmearing(spec['correction'], spec['physical_alpha'], spec['prior_count'],
                         inner_splits=3, seed=seed).fit(features, y, groups, external=external)


def nested_selection(features, y, groups, valid_features, seed=2718):
    """Tune families and the overall selector with NO validation responses."""
    y, groups = check_response(y), np.asarray(groups)
    specs = candidate_specs()
    folds = scaffold_folds(groups, 3, seed)
    scores = []
    for spec in specs:
        oof = np.full(len(y), np.nan)
        failure = None
        try:
            for fold, (train, valid) in enumerate(folds):
                model = fit_spec(spec, subset(features, train), y[train], groups[train], seed + fold + 100)
                oof[valid] = model.predict(subset(features, valid))
            mse = float(np.mean((oof-y)**2))
        except (FloatingPointError, ValueError, ConvergenceWarning, np.linalg.LinAlgError) as exc:
            mse, failure = float('inf'), f'{type(exc).__name__}: {exc}'
        scores.append({'spec': spec, 'inner_MSE': mse, 'failure': failure})
    choices = {'baseline_krr': specs[0], 'baseline_lgbm': specs[1]}
    for family in ['mean_glm', 'smearing']:
        selected = min((r for r in scores if r['spec']['family'] == family), key=lambda r: r['inner_MSE'])
        if not np.isfinite(selected['inner_MSE']):
            raise RuntimeError(f'Every candidate failed in {family}')
        choices[family] = selected['spec']
    choices['nested_selector'] = min(scores, key=lambda r: r['inner_MSE'])['spec']
    choices['smearing_none_control'] = next(s for s in specs if s['name'] == 'regime_none_a3_prior10')
    choices['smearing_global_control'] = next(s for s in specs if s['name'] == 'regime_global_a3_prior10')
    fitted, predictions, details = {}, {}, {}
    for name, spec in choices.items():
        if spec['name'] not in fitted:
            fitted[spec['name']] = fit_spec(spec, features, y, groups, seed)
        model = fitted[spec['name']]
        predictions[name] = model.predict(valid_features)
        details[name] = {'spec': spec, 'fit_diagnostics': getattr(model, 'diagnostics_', {})}
    return predictions, details, scores, fitted
