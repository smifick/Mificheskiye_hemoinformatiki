"""Prespecified paired scaffold and high-range diagnostics; SMILES-only final artifact."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_si_tail_evidence import diagnostic_kinds
from si_tail import scaffold_folds
from si_tail_evidence import (candidate_specs, fit_spec, nested_selection, structures, subset)
from train_model import tanimoto


def metrics(y, p, mask=None):
    mask = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    error = np.asarray(p)[mask] - y[mask]
    return {'n': int(mask.sum()), 'MSE': float(np.mean(error**2)) if mask.any() else None,
            'RMSE': float(np.sqrt(np.mean(error**2))) if mask.any() else None,
            'MAE': float(np.mean(np.abs(error))) if mask.any() else None,
            'bias': float(error.mean()) if mask.any() else None,
            'SSE': float(np.sum(error**2))}


def strata(y, kinds, thresholds):
    physical = kinds == 'measured_ratio'
    return {'all': np.ones(len(y), bool), 'physical_all': physical,
            'physical_low': physical & (y <= 100),
            'physical_medium': physical & (y > 100) & (y <= thresholds),
            'physical_tail': physical & (y > thresholds),
            'technical_10000': kinds == 'technical_10000', 'technical_zero': kinds == 'technical_zero'}


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_json(val) for val in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--out', default='results/si_tail_evidence/benchmark')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2718, 7349])
    parser.add_argument('--diagnostic', action='store_true', help='Remove high physical training scaffold groups')
    parser.add_argument('--no-final', action='store_true')
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    allowed = pd.read_csv(root/'data/raw/train.csv', usecols=['SMILES', 'SI VACV'])
    # Separate audit-only reader, never supplied to estimator fitting.
    audit = pd.read_csv(root/'data/raw/train.csv')
    kinds = diagnostic_kinds(audit)
    y = allowed['SI VACV'].to_numpy(float)
    features, keys, groups = structures(allowed.SMILES)
    protocol = {'seeds': args.seeds, 'outer_splits': 5, 'inner_splits': 3,
                'train_sha256': hashlib.sha256((root/'data/raw/train.csv').read_bytes()).hexdigest(),
                'model_columns': ['SMILES', 'SI VACV'], 'audit_metadata_used_as_features': False,
                'physical_tail_threshold': '95th percentile of outer training measured SI; codes excluded',
                'diagnostic': args.diagnostic,
                'diagnostic_policy': 'Remove entire training scaffold groups containing physical y above training physical q90; evaluate original validation groups',
                'candidates': candidate_specs(), 'predictions_capped': False,
                'external_data': False,
                'final_acceptance': 'Retain fixed strongest eligible baseline LightGBM unless a new family improves overall, physical-low and physical-medium MSE on every prespecified seed',
                'source_sha256': {name: hashlib.sha256((root/'src'/name).read_bytes()).hexdigest()
                                  for name in ['si_tail_evidence.py', 'benchmark_si_tail_evidence.py']},
                'versions': {name: importlib.metadata.version(name) for name in
                             ['numpy', 'pandas', 'scipy', 'scikit-learn', 'rdkit', 'lightgbm', 'joblib']}}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows, fold_rows, selections, candidate_rows, prediction_rows, split_rows = [], [], [], [], [], []
    sse_rows, extreme_rows, component_rows, inner_rows, calibration_rows = [], [], [], [], []
    with threadpool_limits(limits=2):
        for seed in args.seeds:
            predictions, thresholds = {}, np.zeros(len(y))
            folds = scaffold_folds(groups, 5, seed)
            for fold, (initial_train, valid) in enumerate(folds):
                train = initial_train.copy()
                physical_train = train[kinds[train] == 'measured_ratio']
                q95 = float(np.quantile(y[physical_train], .95))
                removed_groups = []
                q90 = float(np.quantile(y[physical_train], .9))
                if args.diagnostic:
                    removed_groups = np.unique(groups[physical_train[y[physical_train] > q90]])
                    train = train[~np.isin(groups[train], removed_groups)]
                if not set(groups[train]).isdisjoint(groups[valid]):
                    raise AssertionError('Scaffold leakage')
                if not set(np.array(keys)[train]).isdisjoint(np.array(keys)[valid]):
                    raise AssertionError('Molecule leakage')
                outer_training_seed=seed+fold+1
                for inner_fold,(inner_train,inner_valid) in enumerate(scaffold_folds(groups[train],3,outer_training_seed)):
                    inner_rows.extend({'seed':seed,'outer_fold':fold,'inner_fold':inner_fold,
                                       'row_index':int(train[i]),'scaffold':groups[train[i]]} for i in inner_valid)
                    inner_physical=train[inner_train][(y[train[inner_train]]>0)&(y[train[inner_train]]!=10000)]
                    for subfold,(_,subvalid) in enumerate(scaffold_folds(groups[inner_physical],3,outer_training_seed+inner_fold+100)):
                        calibration_rows.extend({'seed':seed,'outer_fold':fold,'inner_selection_fold':inner_fold,
                                                 'calibration_fold':subfold,'row_index':int(inner_physical[i]),
                                                 'scaffold':groups[inner_physical[i]]} for i in subvalid)
                outer_physical=train[(y[train]>0)&(y[train]!=10000)]
                for subfold,(_,subvalid) in enumerate(scaffold_folds(groups[outer_physical],3,outer_training_seed)):
                    calibration_rows.extend({'seed':seed,'outer_fold':fold,'inner_selection_fold':-1,
                                             'calibration_fold':subfold,'row_index':int(outer_physical[i]),
                                             'scaffold':groups[outer_physical[i]]} for i in subvalid)
                prediction, chosen, candidates, fitted = nested_selection(
                    subset(features, train), y[train], groups[train], subset(features, valid),
                    seed=seed + fold + 1)
                thresholds[valid] = q95
                for name, values in prediction.items():
                    predictions.setdefault(name, np.full(len(y), np.nan))[valid] = values
                    model=fitted[chosen[name]['spec']['name']]
                    if hasattr(model,'predict_details'):
                        _,components=model.predict_details(subset(features,valid))
                        component_rows.extend({'seed':seed,'fold':fold,'model':name,'row_index':int(i),
                                               'observed':y[i],'prediction':values[j],
                                               **{key:float(value[j]) for key,value in components.items()}}
                                              for j,i in enumerate(valid))
                    for stratum, mask in strata(y[valid], kinds[valid], np.full(len(valid), q95)).items():
                        fold_rows.append({'seed': seed, 'fold': fold, 'model': name, 'stratum': stratum,
                                          'train_n': len(train), 'train_max': float(y[train].max()),
                                          'train_physical_max': float(y[train[kinds[train] == 'measured_ratio']].max()),
                                          'valid_max': float(y[valid].max()), 'pred_max': float(values.max()),
                                          **metrics(y[valid], values, mask)})
                    max_index = valid[np.argmax(y[valid])]
                    local = int(np.flatnonzero(valid == max_index)[0])
                    extreme_rows.append({'seed': seed, 'fold': fold, 'model': name, 'row_index': int(max_index),
                                         'observed': float(y[max_index]), 'prediction': float(values[local]),
                                         'train_max': float(y[train].max()),
                                         'above_training_range': bool(y[max_index] > y[train].max())})
                selections.append({'seed': seed, 'fold': fold, 'models': chosen})
                candidate_rows.extend({'seed': seed, 'fold': fold, **row} for row in candidates)
                split_rows.extend({'seed': seed, 'fold': fold, 'row_index': int(i), 'role': 'validation',
                                   'scaffold': groups[i]} for i in valid)
                split_rows.extend({'seed': seed, 'fold': fold, 'row_index': int(i),
                                   'role': 'training' if i in set(train) else 'removed_diagnostic_group',
                                   'scaffold': groups[i]} for i in initial_train)
                (out/'training_selections.json').write_text(json.dumps(clean_json(selections), indent=2))
                print(f'Finished seed={seed} fold={fold}: training={len(train)}, validation={len(valid)}', flush=True)
            masks = strata(y, kinds, thresholds)
            baseline_sse = (predictions['baseline_lgbm'] - y) ** 2
            for name, p in predictions.items():
                if not np.isfinite(p).all():
                    raise FloatingPointError(f'Incomplete OOF for {name}')
                for stratum, mask in masks.items():
                    rows.append({'seed': seed, 'model': name, 'stratum': stratum, **metrics(y, p, mask)})
                    delta = (p-y)**2 - baseline_sse
                    sse_rows.append({'seed': seed, 'model': name, 'stratum': stratum, 'n': int(mask.sum()),
                                     'SSE_removed': float(-delta[mask & (delta < 0)].sum()),
                                     'SSE_added': float(delta[mask & (delta > 0)].sum()),
                                     'net_MSE_change_contribution': float(delta[mask].sum()/len(y))})
                prediction_rows.extend({'seed': seed, 'model': name, 'row_index': i, 'observed': y[i],
                                         'prediction': p[i], 'label_kind': kinds[i],
                                         'tail_threshold_from_training': thresholds[i],
                                         'squared_error': float((p[i]-y[i])**2)} for i in range(len(y)))
            pd.DataFrame(rows).to_csv(out/'metrics_by_stratum.csv', index=False)
            pd.DataFrame(prediction_rows).to_csv(out/'oof_predictions.csv', index=False)
            pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv', index=False)
            pd.DataFrame(split_rows).to_csv(out/'validation_splits.csv', index=False)
            print(pd.DataFrame(rows).query('stratum == "all"')[['seed','model','MSE']].to_string(index=False), flush=True)
    pd.DataFrame(extreme_rows).to_csv(out/'fold_extreme_predictions.csv', index=False)
    pd.DataFrame(sse_rows).to_csv(out/'error_tradeoff.csv', index=False)
    pd.DataFrame(component_rows).to_csv(out/'component_predictions.csv',index=False)
    pd.DataFrame(inner_rows).to_csv(out/'inner_selection_splits.csv',index=False)
    pd.DataFrame(calibration_rows).to_csv(out/'residual_calibration_splits.csv',index=False)
    (out/'inner_candidate_scores.json').write_text(json.dumps(clean_json(candidate_rows), indent=2))
    if args.diagnostic or args.no_final:
        return
    scores = pd.DataFrame(rows)
    acceptable = []
    for name in ['mean_glm', 'smearing', 'nested_selector']:
        qualifies = True
        for seed in args.seeds:
            for stratum in ['all', 'physical_low', 'physical_medium']:
                pair = scores.query('seed == @seed and stratum == @stratum').set_index('model').MSE
                qualifies &= bool(pair[name] < pair['baseline_lgbm'])
        if qualifies:
            acceptable.append(name)
    if acceptable:
        selected_name = min(acceptable, key=lambda name: scores.query('model == @name and stratum == "all"').MSE.mean())
        _, chosen, _, fitted = nested_selection(features, y, groups, subset(features, np.array([0])), seed=2719)
        spec = chosen[selected_name]['spec']
        model = fitted[spec['name']]
    else:
        selected_name = 'baseline_lgbm'
        spec = candidate_specs()[1]
        model = fit_spec(spec, features, y, groups, seed=2719)
    bundle = {'format': 'si_tail_evidence_v1', 'model': model, 'spec': spec,
              'features': 'canonical tautomer; Morgan bits r2/2048/chirality; signedlog RDKit; no external inputs'}
    joblib.dump(bundle, out/'model.joblib', compress=3)
    test = pd.read_csv(root/'data/raw/test.csv', usecols=['index','SMILES'])
    sample = pd.read_csv(root/'data/raw/sample_submit.csv', usecols=['index'])
    with threadpool_limits(limits=2):
        test_features, test_keys, _ = structures(test.SMILES)
        prediction = model.predict(test_features)
        repeated = joblib.load(out/'model.joblib')['model'].predict(test_features)
    np.testing.assert_allclose(prediction, repeated, rtol=0, atol=1e-10)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Submission has nonfinite/negative predictions; no silent clipping')
    if not test['index'].is_unique or test['index'].tolist() != sample['index'].tolist():
        raise ValueError('Submission IDs/order mismatch')
    pd.DataFrame({'index': test['index'], 'SI VACV': prediction}).to_csv(out/'submission.csv', index=False)
    similarity = tanimoto(test_features['fingerprints'], features['fingerprints'])
    pd.DataFrame({'index': test['index'], 'prediction': prediction,
                  'max_train_tanimoto': similarity.max(axis=1),
                  'nearest_training_row': similarity.argmax(axis=1),
                  'canonical_structure_seen': np.isin(test_keys, keys)}).to_csv(out/'prediction_audit.csv', index=False)
    selection = {'selected_policy': selected_name, 'selected_spec': spec,
                 'accepted_new_policies': acceptable, 'submission_rows': len(test),
                 'submission_min': float(prediction.min()), 'submission_max': float(prediction.max()),
                 'exact_reload': True, 'external_training_rows': 0,
                 'competition_external_data_rule': 'External collection supported by user-provided Dataset Description screenshot; no external labels in this selected candidate',
                 'public_score': None}
    (out/'final_selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == '__main__':
    main()
