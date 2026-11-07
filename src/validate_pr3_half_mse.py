"""Leakage and regression checks for the frozen PR3 improvement experiment.

This refits every frozen head on one archived outer training fold. It does not
repeat cross-validation, change the pool, or select a different candidate.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import research_heads as H
import run_pr3_half_mse as R
from pr3_gate_candidates import extended_notation_features
from smiles_style import raw_style_features


def array_hash(value):
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str((value.dtype, value.shape)).encode())
    if value.dtype.hasobject:
        digest.update(json.dumps(value.tolist(), ensure_ascii=False).encode())
    else:
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def frame_hash(frame):
    return array_hash(pd.util.hash_pandas_object(frame, index=True).to_numpy())


def input_snapshot(state, embeddings):
    """Hash the actual head inputs, not a serialized estimator or cache file."""
    result = {name: array_hash(state[name]) for name in ('y', 'smiles', 'flag', 'groups')}
    result['world.y'] = array_hash(state['world'].y)
    for name in ('train', 'test'):
        result[name] = frame_hash(state[name])
        result['world.'+name] = frame_hash(getattr(state['world'], name))
    result.update({'embedding.'+name: array_hash(value) for name, value in embeddings.items()})
    return result


def poisoned_copy(state, valid, value):
    """Poison all organizer-label representations, retaining training labels."""
    poisoned = state.copy()
    poisoned['world'] = copy.copy(state['world'])
    poisoned['y'] = state['y'].copy()
    poisoned['world'].y = state['world'].y.copy()
    poisoned['y'][valid] = value
    poisoned['world'].y[valid] = value
    poisoned['train'] = state['train'].copy(deep=True)
    poisoned['world'].train = state['world'].train.copy(deep=True)
    poisoned['train'].iloc[valid, poisoned['train'].columns.get_loc('SI VACV')] = value
    poisoned['world'].train.iloc[valid, poisoned['world'].train.columns.get_loc('SI VACV')] = value
    return poisoned


def run(root, cache, out):
    state, embeddings = R.prepare(cache)
    specs = R.specifications()
    rows = np.arange(len(state['y']))
    train, valid = R.folds(state, rows, 'scaffold', 2718, archived=True)[0]
    assert len(specs) == 18, 'Frozen candidate pool changed; review checks explicitly.'
    assert set(state['groups'][train]).isdisjoint(state['groups'][valid])
    initial_snapshot = input_snapshot(state, embeddings)
    archive = np.load(root/'results/audit_20261107/baseline_oof.npz')
    frozen = pd.read_csv(root/'results/pr3_half_mse/development_oof.csv')
    assert list(frozen.columns[2:]) == [s['name'] for s in specs]
    np.testing.assert_allclose(frozen['observed'], state['y'], rtol=1e-14)
    permutation = np.random.default_rng(20261107).permutation(len(valid))
    checks = []
    for column, spec in enumerate(specs):
        model = R.fit_head(spec, state, embeddings, train)
        probability = R.probability(spec, model, state, embeddings, train, valid)
        unchanged = input_snapshot(state, embeddings) == initial_snapshot
        permuted = R.probability(spec, model, state, embeddings, train, valid[permutation])
        finite = bool(np.isfinite(probability).all() and np.isfinite(permuted).all())
        bounded = bool(((probability >= 0) & (probability <= 1)).all())
        case = dict(model=spec['name'], finite=finite, probability_in_0_1=bounded,
                    training_inputs_unchanged=unchanged,
                    query_permutation_max_abs_error=float(np.max(abs(permuted-probability[permutation]))))
        prediction = H.complete_unbounded_predictions(state, train, valid,
                                                       archive['r'][valid], probability)
        case['frozen_outer_prediction_max_abs_error'] = float(np.max(abs(prediction-frozen[spec['name']].to_numpy()[valid])))
        poison_errors = {}
        for name, value in (('nan', np.nan), ('large_1e12', 1e12)):
            poisoned = poisoned_copy(state, valid, value)
            assert np.array_equal(poisoned['y'][train], state['y'][train])
            assert np.array_equal(poisoned['world'].y[train], state['world'].y[train])
            before = input_snapshot(poisoned, embeddings)
            refit = R.fit_head(spec, poisoned, embeddings, train)
            poisoned_probability = R.probability(spec, refit, poisoned, embeddings, train, valid)
            delta = float(np.max(abs(poisoned_probability-probability)))
            poison_errors[name] = delta
            case['finite'] &= bool(np.isfinite(poisoned_probability).all())
            case['probability_in_0_1'] &= bool(((poisoned_probability >= 0) & (poisoned_probability <= 1)).all())
            case['training_inputs_unchanged'] &= input_snapshot(poisoned, embeddings) == before
        case['outer_label_poison_max_abs_error'] = poison_errors
        case['passed'] = bool(case['finite'] and case['probability_in_0_1']
                              and case['training_inputs_unchanged']
                              and case['query_permutation_max_abs_error'] < 1e-12
                              and case['frozen_outer_prediction_max_abs_error'] < 1e-5
                              and all(v < 1e-12 for v in poison_errors.values()))
        checks.append(case)

    # The features must follow strings exactly when the input batch is reordered.
    raw = state['smiles']
    input_order = np.random.default_rng(314).permutation(len(raw))
    feature_errors = {}
    for name, function in (('style9', raw_style_features), ('notation64', extended_notation_features)):
        original = function(raw)
        shuffled = function(raw[input_order])
        feature_errors[name] = float(np.max(abs(original[input_order]-shuffled)))
    output = H.complete_unbounded_predictions(state, train, valid[:1],
                                              np.array([30000.]), np.array([0.]))
    zero_rate = float(np.mean(state['y'][train] == 0))
    expected = 30000*(1-zero_rate)
    unbounded = dict(physical_input=30000., code_probability=0., training_zero_rate=zero_rate,
                     prediction=float(output[0]), expected_prediction=expected,
                     exceeds_10000=bool(output[0] > 10000),
                     exact_formula_matches=bool(abs(output[0]-expected) < 1e-12))
    final_unchanged = input_snapshot(state, embeddings) == initial_snapshot
    result = dict(status='PASS', scope='18 frozen heads, first archived outer fold; no full CV or new selection',
                  heads_checked=len(checks), training_rows=len(train), validation_rows=len(valid),
                  validation_source_rows=(valid+2).tolist(), heads=checks,
                  raw_smiles_feature_permutation_max_abs_error=feature_errors,
                  complete_unbounded_regression=unbounded,
                  original_training_inputs_unchanged=final_unchanged,
                  physical_poisoning_scope='separate physical-agent check; not rerun here')
    passed = (all(c['passed'] for c in checks) and all(v == 0 for v in feature_errors.values())
              and unbounded['exceeds_10000'] and unbounded['exact_formula_matches'] and final_unchanged)
    result['status'] = 'PASS' if passed else 'FAIL'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('heads', 'validation_source_rows')}, indent=2))
    if not passed:
        raise AssertionError('PR3 half-MSE leakage/regression validation failed; see '+str(out))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='work/pr3_half_mse')
    parser.add_argument('--out', default='results/pr3_half_mse/validation.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with threadpool_limits(limits=2):
        run(root, root/args.cache, root/args.out)
