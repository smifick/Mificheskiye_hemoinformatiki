"""Read-only measurement, scaffold and extrapolation audit; no model features from metadata."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits

from si_tail import boosting, ridge_pipeline, scaffold_folds, smiles_features
from train_model import tanimoto


def diagnostic_kinds(data):
    """Metadata is used for audit strata only, never passed to model.fit."""
    both = data.CC50_uM.notna() & data.IC50_uM_VACV.notna()
    code = data['SI VACV'].eq(10000) & data.IC50_uM_VACV.isna()
    zero = data['SI VACV'].eq(0) & data.CC50_uM.isna()
    return np.select([both, code, zero], ['measured_ratio', 'technical_10000', 'technical_zero'],
                     default='unresolved')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--out', default='results/si_tail_evidence/audit')
    parser.add_argument('--embedding-cache', default='work/chemberta_cache')
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(root / 'data/raw/train.csv')
    y = data['SI VACV'].to_numpy(float)
    kinds = diagnostic_kinds(data)
    features, keys, groups = smiles_features(data.SMILES)
    signed = np.sign(features['descriptors']) * np.log1p(np.abs(features['descriptors']))
    folds = scaffold_folds(groups, 5, 2718)
    data = data.assign(row_index=np.arange(len(y)), source_row=np.arange(len(y)) + 2,
                       canonical_smiles=keys, scaffold_group=groups, diagnostic_kind=kinds,
                       ratio_from_supplied_concentrations=data.CC50_uM / data.IC50_uM_VACV,
                       concentration_unit='uM', doi_status='not supplied by organizer',
                       relation_status='numeric CSV; original comparison operators not supplied')
    data['relative_ratio_discrepancy'] = np.abs(data.ratio_from_supplied_concentrations-y) / np.maximum(y, 1e-12)
    data.to_csv(out / 'label_audit.csv', index=False)
    data[data.duplicated('canonical_smiles', keep=False)].sort_values('canonical_smiles').to_csv(
        out / 'duplicate_measurements.csv', index=False)
    identities=[]
    for key, repeated in data.groupby('canonical_smiles'):
        if len(repeated)>1:
            values=repeated['SI VACV']
            identities.append({'canonical_smiles':key,'n':len(repeated),'min_SI':float(values.min()),
                               'max_SI':float(values.max()),
                               'material_difference_rtol_2pct':not bool(np.isclose(values.min(),values.max(),rtol=.02,atol=.001)),
                               'within_identity_SSE':float(np.sum((values-values.mean())**2))})
    identity_table=pd.DataFrame(identities)
    identity_table.to_csv(out/'contradictory_identities.csv',index=False)
    bit_hashes=[hashlib.sha256(row.tobytes()).hexdigest() for row in features['fingerprints']]
    fingerprint_audit=pd.DataFrame({'row_index':np.arange(len(y)),'fingerprint_sha256':bit_hashes,
                                    'canonical_smiles':keys})
    bit_diversity=fingerprint_audit.groupby('fingerprint_sha256').canonical_smiles.nunique()
    collided=bit_diversity[bit_diversity>1].index
    fingerprint_audit[fingerprint_audit.fingerprint_sha256.isin(collided)].to_csv(out/'fingerprint_collisions.csv',index=False)
    identity_summary={'raw_unique_smiles':data.SMILES.nunique(),'canonical_unique_structures':len(set(keys)),
                      'repeated_identity_groups':len(identities),
                      'materially_conflicting_identity_groups':int(identity_table.material_difference_rtol_2pct.sum()),
                      'empirical_minimum_MSE_for_deterministic_canonical_structure_predictor':float(identity_table.within_identity_SSE.sum()/len(y)),
                      'scope':'Empirical bound for this normalized representation and these rows; not an estimate of population Bayes risk',
                      'morgan_bit_patterns_shared_by_distinct_structures':len(collided),
                      'extreme_fingerprint_collides_with_different_structure':bit_hashes[int(np.argmax(y))] in set(collided)}
    (out/'identity_noise.json').write_text(json.dumps(identity_summary,indent=2))
    physical = kinds == 'measured_ratio'
    top = np.flatnonzero(physical)[np.argsort(y[physical])[-12:][::-1]]
    data.iloc[top].to_csv(out / 'largest_measured_labels.csv', index=False)
    oof = {name: np.full(len(y), np.nan) for name in ['reference_krr_raw', 'control_lgbm_raw', 'H2_combined_ridge']}
    saved = json.loads((root/'results/si_tail_v5_signedlog/training_selections.json').read_text())
    meta = json.loads((root/'results/si_tail_v5_signedlog/protocol.json').read_text())['chemberta']
    from si_tail import extract_chemberta_embeddings
    embeddings, _ = extract_chemberta_embeddings(keys, cache_dir=args.embedding_cache, revision=meta['revision'])
    combined = np.column_stack([signed, embeddings])
    split, stats, neighbors = [], [], []
    with threadpool_limits(limits=2):
        for fold, (train, valid) in enumerate(folds):
            from si_tail import TanimotoRegressor
            model = TanimotoRegressor().fit(features['fingerprints'][train], y[train])
            oof['reference_krr_raw'][valid] = model.predict(features['fingerprints'][valid])
            X = np.column_stack([features['fingerprints'], signed])
            oof['control_lgbm_raw'][valid] = boosting(2718 + fold + 1).fit(X[train], y[train]).predict(X[valid])
            # Reproduce the saved Ridge hyperparameter choice, not a new selection.
            alpha = saved[fold]['models']['H2_combined_ridge']['selected']['alpha']
            oof['H2_combined_ridge'][valid] = ridge_pipeline(alpha).fit(combined[train], y[train]).predict(combined[valid])
            measured_train = train[physical[train]]
            threshold = float(np.quantile(y[measured_train], .95))
            row = {'fold': fold, 'train_n': len(train), 'valid_n': len(valid),
                   'train_max_all': float(y[train].max()), 'valid_max_all': float(y[valid].max()),
                   'physical_train_n': len(measured_train), 'physical_valid_n': int(physical[valid].sum()),
                   'physical_tail_threshold_train_q95': threshold,
                   'physical_tail_train_n': int((y[measured_train] > threshold).sum()),
                   'physical_tail_valid_n': int((physical[valid] & (y[valid] > threshold)).sum()),
                   'valid_above_train_all_max_n': int((y[valid] > y[train].max()).sum()),
                   'valid_physical_above_train_physical_max_n': int((physical[valid] & (y[valid] > y[measured_train].max())).sum())}
            for quantile in [.5, .9, .95, .99, 1.0]:
                row[f'train_q{quantile:g}'] = float(np.quantile(y[train], quantile))
                row[f'physical_train_q{quantile:g}'] = float(np.quantile(y[measured_train], quantile))
            stats.append(row)
            split.extend({'row_index': int(i), 'fold': fold, 'group': groups[i]} for i in valid)
            for index in valid[physical[valid] & (y[valid] > threshold)]:
                similarity = tanimoto(features['fingerprints'][[index]], features['fingerprints'][train])[0]
                for rank, position in enumerate(np.argsort(similarity)[-5:][::-1], 1):
                    neighbor = train[position]
                    neighbors.append({'query_row_index': int(index), 'query_SI': y[index], 'fold': fold,
                                      'rank': rank, 'neighbor_row_index': int(neighbor),
                                      'tanimoto': float(similarity[position]), 'neighbor_SI': y[neighbor],
                                      'neighbor_label_kind': kinds[neighbor],
                                      'query_cell': data.iloc[index].Culture, 'neighbor_cell': data.iloc[neighbor].Culture,
                                      'query_SMILES': keys[index], 'neighbor_SMILES': keys[neighbor],
                                      'same_scaffold': bool(groups[index] == groups[neighbor])})
    pd.DataFrame(stats).to_csv(out/'fold_target_diagnostics.csv', index=False)
    pd.DataFrame(split).to_csv(out/'validation_splits.csv', index=False)
    pd.DataFrame(neighbors).to_csv(out/'tail_neighbors.csv', index=False)
    pd.DataFrame({'row_index': np.arange(len(y)), 'observed': y, **oof}).to_csv(out/'baseline_reproduced_oof.csv', index=False)
    old = pd.read_csv(root/'results/si_tail_v5_signedlog/oof_predictions.csv')
    reproduction = {name: {'MSE': float(mean_squared_error(y, pred)),
                          'max_abs_difference_vs_previous': float(np.max(np.abs(pred-old[name])))}
                    for name, pred in oof.items()}
    # Family-level counterexample: sums of trees are not universally bounded by max(y).
    from lightgbm import LGBMRegressor
    toy_X = np.array([[0., 0.], [1., 0.], [0., 1.]])
    toy_y = np.array([0., 1., 1.])
    toy = LGBMRegressor(n_estimators=200, learning_rate=.1, max_depth=1, num_leaves=2,
                        min_child_samples=1, min_data_in_bin=1, reg_lambda=0,
                        n_jobs=1, verbosity=-1).fit(toy_X, toy_y)
    toy_prediction = float(toy.predict([[1., 1.]])[0])
    summary = {'train_sha256': hashlib.sha256((root/'data/raw/train.csv').read_bytes()).hexdigest(),
               'rows': len(y), 'label_kinds': pd.Series(kinds).value_counts().to_dict(),
               'ratio_agrees_rtol_2pct': int(np.isclose(data.ratio_from_supplied_concentrations[physical], y[physical], rtol=.02).sum()),
               'extreme': data.iloc[int(np.argmax(y))][['row_index','source_row','SI VACV','CC50_uM','IC50_uM_VACV','Culture','doi_status','relation_status']].to_dict(),
               'reproduction': reproduction,
               'boosting_counterexample': {'training_max': float(toy_y.max()), 'prediction': toy_prediction,
                                            'exceeds_training_max': bool(toy_prediction > toy_y.max())}}
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
