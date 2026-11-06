"""Reproduce v6, then compare prespecified changes on paired structural folds.

Requires the user-provided train_v5.py and train_v6.py next to this file.
Only SMILES and target from organizer train, and SMILES/index from test are used.
External source-presence flags are computed from the frozen repository snapshot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.linear_model import LogisticRegression

import train_v5 as T
import train_v6 as V
import train_kernel_v7 as F


def get_world(out):
    cache = out / 'world.joblib'
    if cache.exists():
        return joblib.load(cache)
    w = T.World()
    # The legacy loader reads full tables. Remove all fields outside the allowlist
    # before any candidate training. The numerical labels are w.y only.
    w.train = w.train[['SMILES', 'SI VACV']].copy()
    w.test = w.test[['index', 'SMILES']].copy()
    flag = V.chembl_flags(w)
    fingerprints, _ = F.features(pd.concat([w.train.SMILES, w.test.SMILES]))
    kernels = F.kernels(fingerprints)
    kernels['v6'] = w.K[:w.n_tr+w.n_te, :w.n_tr+w.n_te]
    # A local similarity emphasizes close neighbors while retaining PSD.
    kernels['v6_squared'] = kernels['v6']**2
    joblib.dump((w, flag, fingerprints, kernels), cache, compress=3)
    return w, flag, fingerprints, kernels


def p_specs():
    return [
        dict(name='v6', kind='krr', kernel='v6', alpha=.5),
        dict(name='v6_a0.1', kind='krr', kernel='v6', alpha=.1),
        dict(name='v6_a2', kind='krr', kernel='v6', alpha=2.),
        dict(name='squared', kind='krr', kernel='v6_squared', alpha=.5),
        dict(name='morgan_multi', kind='krr', kernel='morgan_multi', alpha=.5),
        dict(name='counts_multi', kind='krr', kernel='counts_multi', alpha=.5),
        dict(name='atom_pair', kind='krr', kernel='ap', alpha=.5),
        dict(name='torsion', kind='krr', kernel='tt', alpha=.5),
        dict(name='feature', kind='krr', kernel='feature', alpha=.5),
        dict(name='all_multi', kind='krr', kernel='all_multi', alpha=.5),
        dict(name='logistic_C0.1', kind='logistic', C=.1),
        dict(name='logistic_C1', kind='logistic', C=1.),
        dict(name='logistic_C10', kind='logistic', C=10.),
        dict(name='kernel_ensemble', kind='ensemble'),
    ]


def code(w, flag, fingerprints, kernels, tr, va, spec):
    c = (w.y == 10000).astype(float)
    flagged = tr[flag[tr]]
    pf = (c[flagged].sum()+.5)/(len(flagged)+2.)
    unflagged = tr[~flag[tr]]
    if spec['kind'] == 'ensemble':
        choices = [p_specs()[0], p_specs()[4], p_specs()[5]]
        pp = np.mean([code(w, flag, fingerprints, kernels, tr, va, s) for s in choices], axis=0)
    elif spec['kind'] == 'logistic':
        est = LogisticRegression(C=spec['C'], max_iter=2000, solver='lbfgs', random_state=0)
        if np.unique(c[unflagged]).size < 2:
            pp = np.full(len(va), c[unflagged].mean())
        else:
            est.fit(fingerprints['m2'][unflagged], c[unflagged])
            pp = est.predict_proba(fingerprints['m2'][va])[:, 1]
    else:
        k = kernels[spec['kernel']]
        mu = c[unflagged].mean()
        dual = solve(k[np.ix_(unflagged, unflagged)]+spec['alpha']*np.eye(len(unflagged)), c[unflagged]-mu, assume_a='pos')
        pp = np.clip(mu+k[np.ix_(va, unflagged)]@dual, 0., 1.)
    return np.where(flag[va], pf, pp)


def mix(w, tr, p, r):
    p0 = np.mean(w.y[tr] == 0)
    return np.clip(10000*p + np.clip(1-p-p0, 0, 1)*r, 0, 10000)


def b_specs():
    return [
        ('v6_B', V.CFG.copy()),
        ('raw_2000', T.C(B='raw', cap=2000., w={'V': 1., 'C': 1.})),
        ('raw_10000', T.C(B='raw', cap=10000., w={'V': 1., 'C': 1.})),
        ('log_exp', T.C(B='log_exp', w={'V': 1., 'C': 1.})),
        ('log_cal', T.C(B='log_cal', w={'V': 1., 'C': 1.})),
    ]


def score_all(w, flag, fingerprints, kernels, tr, va, ext_mask):
    probs = {s['name']: code(w, flag, fingerprints, kernels, tr, va, s) for s in p_specs()}
    physical = {}
    for name, cfg in b_specs():
        ew = np.array([cfg['w'].get(t, 0.) for t in w.ext_tier])
        physical[name] = T.fit_B(w, tr, ext_mask, ew, va, cfg)
    result = {name: mix(w, tr, p, physical['v6_B']) for name, p in probs.items()}
    # Single changes in B retain the original code classifier for clear ablations.
    for name, r in physical.items():
        if name != 'v6_B':
            result[name] = mix(w, tr, probs['v6'], r)
    return result, probs, physical


def lookup(w, pred, val):
    lt = T.lookup_table(w)
    e = np.array([lt.get(w.tau[i], np.nan) for i in val])
    return np.where(np.isfinite(e), np.clip(np.nan_to_num(e), 0, 10000), pred)


def run_seed(w, flag, fingerprints, kernels, seed):
    preds, with_lookup = {}, {}
    records = []
    for fold, va in enumerate(T.make_folds(w.groups, seed)):
        tr = np.setdiff1d(np.arange(w.n_tr), va)
        assert set(w.scf[tr]).isdisjoint(w.scf[va])
        assert set(w.tau[tr]).isdisjoint(w.tau[va])
        em = T.ext_allowed(w, va)
        assert set(w.tau[w.ext_ids[em]]).isdisjoint(w.tau[va])
        candidate, _, _ = score_all(w, flag, fingerprints, kernels, tr, va, em)
        for name, p in candidate.items():
            if name not in preds:
                preds[name] = np.zeros(w.n_tr)
                with_lookup[name] = np.zeros(w.n_tr)
            preds[name][va] = p
            with_lookup[name][va] = lookup(w, p, va)
        records += [dict(seed=seed, fold=fold, source_row=int(i+2), group=int(w.groups[i])) for i in va]
    return preds, with_lookup, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/v7')
    ap.add_argument('--reference', default='submission_v6.csv')
    args = ap.parse_args()
    out = T.ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'development_seeds': [0, 1, 2, 3], 'confirmation_seeds': [101, 102, 103, 104, 105, 106, 107, 108],
                'candidates_A': p_specs(), 'candidates_B': [dict(name=n, config=c) for n, c in b_specs()],
                'selection': 'Minimum mean development scaffold MSE without lookup; final candidate frozen before confirmation.',
                'external_lookup': 'Reported separately, allowed only as a frozen external mapping, not fitted on organizer validation labels.',
                'public_v6_mse_user_reported': 9649437.79854,
                'source_commit': '5eecc1ca55bd7e8a675a0c8939f4c57717e04c41'}
    path = out / 'protocol.json'
    if not path.exists():
        path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    w, flag, fingerprints, kernels = get_world(out)
    rows, splits, saved = [], [], []
    for seed in protocol['development_seeds']:
        p, pl, ss = run_seed(w, flag, fingerprints, kernels, seed)
        splits += ss
        for name in p:
            rows.append(dict(seed=seed, model=name, MSE=float(np.mean((p[name]-w.y)**2)), MSE_lookup=float(np.mean((pl[name]-w.y)**2))))
            saved.extend(dict(seed=seed, source_row=i+2, model=name, prediction=float(p[name][i]), with_lookup=float(pl[name][i])) for i in range(w.n_tr))
        print('DEVELOPMENT', seed, sorted([(r['model'],r['MSE']) for r in rows if r['seed']==seed], key=lambda z:z[1])[:5], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out/'development.csv', index=False)
    pd.DataFrame(saved).to_csv(out/'development_oof.csv', index=False)
    pd.DataFrame(splits).to_csv(out/'splits.csv', index=False)
    means = df.groupby('model')[['MSE', 'MSE_lookup']].mean().sort_values('MSE')
    means.to_csv(out/'development_summary.csv')
    selected = means.index[0]
    (out/'selection_frozen.json').write_text(json.dumps({'selected': selected, 'development_summary': means.reset_index().to_dict('records')}, indent=2))
    print('FROZEN', selected, means.head(8).to_string(), flush=True)
    confirm, predictions = [], []
    for seed in protocol['confirmation_seeds']:
        p, pl, _ = run_seed(w, flag, fingerprints, kernels, seed)
        for name in dict.fromkeys(['v6', selected]):
            confirm.append(dict(seed=seed, model=name, MSE=float(np.mean((p[name]-w.y)**2)), MSE_lookup=float(np.mean((pl[name]-w.y)**2))))
            predictions.extend(dict(seed=seed, source_row=i+2, model=name, observed=float(w.y[i]), prediction=float(p[name][i]), with_lookup=float(pl[name][i])) for i in range(w.n_tr))
        print('CONFIRM', seed, [(r['model'], r['MSE']) for r in confirm if r['seed']==seed], flush=True)
    pd.DataFrame(confirm).to_csv(out/'confirmation.csv', index=False)
    pd.DataFrame(predictions).to_csv(out/'confirmation_oof.csv', index=False)
    tr = np.arange(w.n_tr)
    te = np.arange(w.n_tr, w.n_tr+w.n_te)
    p, _, _ = score_all(w, flag, fingerprints, kernels, tr, te, np.ones(w.n_ext, bool))
    baseline = lookup(w, p['v6'], te)
    supplied = pd.read_csv(T.ROOT / args.reference)
    assert supplied['index'].to_list() == w.test['index'].to_list()
    delta = float(np.max(np.abs(baseline-supplied['SI VACV'].to_numpy())))
    assert delta < 1e-6, f'v6 reproduction discrepancy {delta}'
    final = lookup(w, p[selected], te)
    sub = pd.DataFrame({'index':w.test['index'], 'SI VACV':final})
    assert len(sub)==597 and sub['index'].is_unique and np.isfinite(final).all()
    sub.to_csv(out/'submission_v7.csv', index=False)
    conf = pd.DataFrame(confirm).pivot(index='seed', columns='model', values='MSE')
    change = conf[selected]-conf['v6']
    summary = {'selected': selected, 'development':means.reset_index().to_dict('records'),
               'confirmation_v6_mean':float(conf['v6'].mean()), 'confirmation_v7_mean':float(conf[selected].mean()),
               'confirmation_change_pct':float(100*change.mean()/conf['v6'].mean()),
               'confirmation_seeds_better':int((change<0).sum()), 'v6_csv_max_abs_error':delta,
               'test_changed_rows':int(np.sum(np.abs(final-baseline)>1e-7)),
               'public_v7_mse':None,
               'warning':'Repeated folds reuse the 320-row development dataset and are not an independent dataset. Winner selected before confirmation split results. External-assisted lookup metric differs from strict structural generalization.'}
    (out/'metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print('FINAL', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
