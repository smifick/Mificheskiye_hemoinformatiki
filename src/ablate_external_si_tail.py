"""Research-only external-data ablation, with molecule/scaffold exclusion at every fit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_si_tail_evidence import diagnostic_kinds
from benchmark_si_tail_evidence import metrics, strata
from si_tail import scaffold_folds
from si_tail_evidence import RegimeSmearing, structures, subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', default='results/si_tail_evidence/external_audit/research_training_bank.csv')
    parser.add_argument('--out', default='results/si_tail_evidence/external_ablation')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv('data/raw/train.csv', usecols=['SMILES','SI VACV'])
    kinds = diagnostic_kinds(pd.read_csv('data/raw/train.csv'))
    bank = pd.read_csv(args.bank)
    features, keys, groups = structures(train.SMILES)
    external, ext_keys, ext_groups = structures(bank.SMILES)
    y, ext_y = train['SI VACV'].to_numpy(float), bank.SI.to_numpy(float)
    specs = [('none', None, 0.0), ('previous_curated_w0.1', 'previous_curated_vero', .1),
             ('previous_curated_w1', 'previous_curated_vero', 1.0),
             ('new_publication_w0.1', 'new_complete_publication', .1),
             ('combined_w0.1', 'all', .1), ('combined_w1', 'all', 1.0)]
    protocol = {'competition_submission': False,
                'external_rule_status': 'External collection supported by user-provided Dataset Description screenshot; this experiment is an ablation, not an automatically selected final addition',
                'seed': 2718, 'folds': 5, 'frozen_physical_model': 'uncapped regime global smearing; alpha=3',
                'specs': specs, 'test_overlaps_removed_by_bank_curation': True,
                'validation_molecule_and_scaffold_exclusion': True,
                'inner_smearing_molecule_and_scaffold_exclusion': True,
                'no_external_lookup': True}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    predictions = {name: np.full(len(y), np.nan) for name,_,_ in specs}
    thresholds = np.zeros(len(y))
    audit_rows, fold_scores = [], []
    with threadpool_limits(limits=2):
        for fold, (fit, valid) in enumerate(scaffold_folds(groups, 5, 2718)):
            threshold = np.quantile(y[fit[kinds[fit]=='measured_ratio']], .95)
            thresholds[valid] = threshold
            outer_safe = ~np.isin(ext_groups, groups[valid]) & ~np.isin(ext_keys, np.asarray(keys)[valid])
            for name, origin, weight in specs:
                keep = outer_safe & (bank.origin.eq(origin).to_numpy() if origin != 'all' else True)
                if origin is None:
                    keep[:] = False
                ids = np.flatnonzero(keep)
                ext_tuple = (subset(external, ids), ext_y[ids], ext_groups[ids], weight) if len(ids) else None
                if set(ext_groups[ids]) & set(groups[valid]) or set(np.asarray(ext_keys)[ids]) & set(np.asarray(keys)[valid]):
                    raise AssertionError('External leakage')
                model = RegimeSmearing('global', 3.0, 10.0, seed=2719+fold).fit(
                    subset(features, fit), y[fit], groups[fit], external=ext_tuple)
                pred = model.predict(subset(features, valid))
                predictions[name][valid] = pred
                fold_scores.append({'model': name, 'fold': fold, 'external_rows': len(ids), **metrics(y[valid],pred)})
                audit_rows.extend({'model':name,'fold':fold,'bank_row_index':int(i),
                                   'canonical_smiles':ext_keys[i],'scaffold':ext_groups[i]} for i in ids)
            print('External ablation fold', fold, flush=True)
    rows = [{'model': name, 'stratum': stratum, **metrics(y, p, mask)}
            for name,p in predictions.items() for stratum,mask in strata(y,kinds,thresholds).items()]
    pd.DataFrame(rows).to_csv(out/'comparison.csv', index=False)
    pd.DataFrame(fold_scores).to_csv(out/'fold_metrics.csv', index=False)
    pd.DataFrame(audit_rows).to_csv(out/'external_training_rows.csv', index=False)
    pd.DataFrame({'row_index':np.arange(len(y)),'observed':y,**predictions}).to_csv(out/'oof_predictions.csv',index=False)
    # Entire newly curated publication, structure/scaffold-held-out retrospective check.
    query = np.flatnonzero(bank.origin.eq('new_complete_publication').to_numpy())
    hold_features = subset(external,query)
    safe_train = ~np.isin(groups,ext_groups[query]) & ~np.isin(keys,np.asarray(ext_keys)[query])
    safe_ext = bank.origin.eq('previous_curated_vero').to_numpy() & ~np.isin(ext_groups,ext_groups[query])
    safe_ext &= ~np.isin(ext_keys,np.asarray(ext_keys)[query])
    publication_rows=[]
    with threadpool_limits(limits=2):
        for use_external in [False,True]:
            ei=np.flatnonzero(safe_ext) if use_external else np.array([],int)
            ext_tuple=(subset(external,ei),ext_y[ei],ext_groups[ei],.1) if len(ei) else None
            ti=np.flatnonzero(safe_train)
            model=RegimeSmearing('global',3.0,10.0).fit(subset(features,ti),y[ti],groups[ti],external=ext_tuple)
            _,detail=model.predict_details(hold_features)
            for j,i in enumerate(query):
                publication_rows.append({'with_old_external':use_external,'bank_row_index':int(i),
                                         'observed_SI':ext_y[i],'physical_prediction':detail['physical_SI'][j],
                                         'organizer_train_rows':len(ti),'external_train_rows':len(ei)})
    pd.DataFrame(publication_rows).to_csv(out/'publication_holdout.csv',index=False)
    print(pd.DataFrame(rows).query('stratum == "all"')[['model','MSE']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
