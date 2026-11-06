"""Nested validation of the final conservative v6/feature-kernel blend.

Candidate weights 0, 0.5, 1 selected by inner OOF MSE. Both scaffold and
parent-structure folds are reported; no external lookup is used in this test.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import StratifiedGroupKFold
import experiment_v7 as E


def fold_indices(w, rows, scheme, seed):
    if scheme == 'scaffold':
        return E.T.make_folds(w.groups[rows], seed, k=3)
    parents = np.array([Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in w.tau[rows]])
    return [va for _, va in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(rows, w.y[rows] == 10000, parents)]


def components(w, flag, f, kernels, tr, va, external_mask):
    p0 = E.code(w, flag, f, kernels, tr, va, E.p_specs()[0])
    p1 = E.code(w, flag, f, kernels, tr, va, next(s for s in E.p_specs() if s['name'] == 'feature'))
    cfg = E.V.CFG
    ew = np.array([cfg['w'].get(t, 0.) for t in w.ext_tier])
    r = E.T.fit_B(w, tr, external_mask, ew, va, cfg)
    return E.mix(w, tr, p0, r), E.mix(w, tr, p1, r)


def main():
    root = E.T.ROOT
    out = root/'results/v7'
    protocol = {'weights': [0., .5, 1.], 'outer_seed': 2029, 'inner_seed': 781,
                'outer_folds': 5, 'inner_folds': 3,
                'selection': 'Minimize pooled inner MSE; lower feature weight on exact tie.',
                'schemes': ['scaffold', 'parent'],
                'note': 'Nested estimates evaluate this shortlist selection procedure on reused development data, not a new independent dataset.'}
    (out/'nested_protocol.json').write_text(json.dumps(protocol, indent=2))
    w, flag, f, kernels = joblib.load(out/'world.joblib')
    allrows = np.arange(w.n_tr)
    parents = np.array([Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in w.tau[:w.n_tr]])
    records, oof_records = [], []
    for scheme in protocol['schemes']:
        if scheme == 'scaffold':
            outer = E.T.make_folds(w.groups, protocol['outer_seed'], 5)
        else:
            outer = [va for _, va in StratifiedGroupKFold(5,shuffle=True,random_state=protocol['outer_seed']).split(allrows, w.y==10000, parents)]
        for fold, va in enumerate(outer):
            tr = np.setdiff1d(allrows, va)
            assert set(parents[tr]).isdisjoint(parents[va])
            if scheme == 'scaffold':
                assert set(w.scf[tr]).isdisjoint(w.scf[va])
            mask_outer = E.T.ext_allowed(w, va)
            o0, o1 = np.zeros(w.n_tr), np.zeros(w.n_tr)
            for vi in fold_indices(w, tr, scheme, protocol['inner_seed']):
                iv = tr[vi]
                it = np.setdiff1d(tr, iv)
                mask = mask_outer & E.T.ext_allowed(w, iv)
                o0[iv], o1[iv] = components(w, flag, f, kernels, it, iv, mask)
            losses = {a: float(np.mean(((1-a)*o0[tr]+a*o1[tr]-w.y[tr])**2)) for a in protocol['weights']}
            chosen = min(protocol['weights'], key=losses.get)
            a, b = components(w, flag, f, kernels, tr, va, mask_outer)
            nested = (1-chosen)*a+chosen*b
            fixed = .5*(a+b)
            records.append({'scheme':scheme,'fold':fold,'n':len(va),'selected_weight':chosen,
                            'v6_MSE':float(np.mean((a-w.y[va])**2)),
                            'blend50_MSE':float(np.mean((fixed-w.y[va])**2)),
                            'nested_MSE':float(np.mean((nested-w.y[va])**2))})
            oof_records += [dict(scheme=scheme,fold=fold,source_row=int(i+2),observed=float(w.y[i]),
                                 v6=float(a[j]),blend50=float(fixed[j]),nested=float(nested[j])) for j,i in enumerate(va)]
            print('FOLD', records[-1], flush=True)
    df = pd.DataFrame(oof_records)
    summaries = []
    for scheme, d in df.groupby('scheme'):
        summary = {'scheme':scheme}
        for name in ['v6','blend50','nested']:
            summary[name+'_MSE'] = float(np.mean((d[name]-d.observed)**2))
        summaries.append(summary)
    pd.DataFrame(records).to_csv(out/'nested_shortlist_folds.csv',index=False)
    df.to_csv(out/'nested_shortlist_oof.csv',index=False)
    (out/'nested_summary.json').write_text(json.dumps(summaries,indent=2))
    print('FINAL',json.dumps(summaries),flush=True)


if __name__ == '__main__':
    main()
