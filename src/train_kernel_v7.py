"""SMILES-only kernel search with nested grouped validation; no test labels.

Run: python src/train_kernel_v7.py --root . --out results/kernel_v7
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import joblib
import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

TARGET = 'SI VACV'
ALPHAS = [.1, .3, 1., 3.]
RDLogger.DisableLog('rdApp.warning')


def standardize(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        raise ValueError('Invalid SMILES')
    parts = Chem.GetMolFrags(m, asMols=True)
    m = max(parts, key=lambda p: (any(a.GetAtomicNum() == 6 for a in p.GetAtoms()), p.GetNumHeavyAtoms()))
    m = rdMolStandardize.Uncharger().uncharge(m)
    Chem.SanitizeMol(m)
    sc = MurckoScaffold.GetScaffoldForMol(m)
    group = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(sc), isomericSmiles=False) if sc.GetNumAtoms() else 'ACYCLIC'
    tau = rdMolStandardize.TautomerEnumerator().Canonicalize(m)
    return Chem.MolToSmiles(m), Chem.MolToSmiles(tau), group


def features(smiles):
    std = [standardize(s) for s in smiles]
    mols = [Chem.MolFromSmiles(s[0]) for s in std]
    taut = [Chem.MolFromSmiles(s[1]) for s in std]
    generators = {}
    for r in [1, 2, 3]:
        generators[f'm{r}'] = (rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=8192, includeChirality=True), False, mols)
        generators[f'c{r}'] = (rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=8192, includeChirality=True), True, mols)
    generators['legacy'] = (rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True), False, mols)
    generators['taut'] = (rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True), False, taut)
    generators['feature'] = (rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=8192, includeChirality=True, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()), False, mols)
    generators['ap'] = (rdFingerprintGenerator.GetAtomPairGenerator(fpSize=8192, includeChirality=True), True, mols)
    generators['tt'] = (rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=8192, includeChirality=True), True, mols)
    generators['rdk'] = (rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=8192), False, mols)
    result = {}
    for key, (gen, count, ms) in generators.items():
        a = np.zeros((len(ms), gen.GetOptions().fpSize), dtype=np.float32)
        for i, m in enumerate(ms):
            fp = gen.GetCountFingerprint(m) if count else gen.GetFingerprint(m)
            DataStructs.ConvertToNumpyArray(fp, a[i])
        if count:
            a = np.sqrt(a)
        result[key] = csr_matrix(a)
    a = np.zeros((len(mols), 167), dtype=np.float32)
    for i, m in enumerate(mols):
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), a[i])
    result['maccs'] = csr_matrix(a)
    return result, std


def kernels(f, reference=None):
    ref = f if reference is None else reference
    k = {}
    for key, a in f.items():
        b = ref[key]
        dot = (a @ b.T).toarray().astype(np.float64)
        aa = np.asarray(a.multiply(a).sum(axis=1)).ravel()
        bb = np.asarray(b.multiply(b).sum(axis=1)).ravel()
        k[key] = dot / np.maximum(aa[:, None] + bb[None, :] - dot, 1e-12)
    for name, keys in {
        'morgan_multi': ['m1', 'm2', 'm3'],
        'counts_multi': ['c1', 'c2', 'c3'],
        'm2_ap': ['m2', 'ap'],
        'm2_feature': ['m2', 'feature'],
        'all_multi': ['m2', 'c2', 'ap', 'tt', 'rdk'],
    }.items():
        k[name] = sum(k[x] for x in keys) / len(keys)
    return k


def predict(k, y, tr, va, alpha):
    a = k[np.ix_(tr, tr)] + alpha * np.eye(len(tr))
    dual = scipy.linalg.solve(a, y[tr], assume_a='pos', check_finite=False)
    return np.maximum(k[np.ix_(va, tr)] @ dual, 0.)


def cv_all(kernels_, y, ids, folds):
    predictions = {}
    for key, k in kernels_.items():
        result = np.zeros((len(ids), len(ALPHAS)))
        for tr0, va0 in folds:
            tr, va = ids[tr0], ids[va0]
            eig, vectors = eigh(k[np.ix_(tr, tr)], check_finite=False)
            dual = vectors @ ((vectors.T @ y[tr])[:, None] / (eig[:, None] + np.asarray(ALPHAS)[None, :]))
            result[va0] = np.maximum(k[np.ix_(va, tr)] @ dual, 0.)
        for j, alpha in enumerate(ALPHAS):
            predictions[f'{key}__{alpha}'] = result[:, j]
    return predictions


def ranking(preds, y):
    return pd.DataFrame([{'model': name, 'MSE': float(np.mean((p - y)**2))} for name, p in preds.items()]).sort_values('MSE')


def spec(name):
    key, a = name.rsplit('__', 1)
    return key, float(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='results/kernel_v7')
    args = ap.parse_args()
    root = Path(args.root)
    out = root / args.out
    out.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(root / 'data/raw/train.csv', usecols=['SMILES', TARGET])
    test = pd.read_csv(root / 'data/raw/test.csv', usecols=['index', 'SMILES'])
    sample = pd.read_csv(root / 'data/raw/sample_submit.csv', usecols=['index'])
    y = train[TARGET].to_numpy(float)
    f, std = features(train.SMILES)
    ft, st = features(test.SMILES)
    k = kernels(f)
    kt = kernels(ft, f)
    ids = np.arange(len(y))
    groups = np.array([s[2] for s in std])
    # Collapse parents and tautomers in interpolation diagnostics.
    parents = np.array([Chem.MolToSmiles(Chem.MolFromSmiles(s[1]), isomericSmiles=False) for s in std])
    folds = list(GroupKFold(5, shuffle=True, random_state=2718).split(ids, y, groups))
    config = {'kernels': list(k), 'alphas': ALPHAS, 'outer_seed': 2718, 'inner_seed': 314,
              'selection': 'Minimum pooled inner scaffold OOF MSE; ties retain table order',
              'features': ['SMILES'], 'target': TARGET, 'external_used': False,
              'train_sha256': hashlib.sha256((root / 'data/raw/train.csv').read_bytes()).hexdigest()}
    (out / 'protocol.json').write_text(json.dumps(config, indent=2))
    allpred = cv_all(k, y, ids, folds)
    table = ranking(allpred, y)
    table.to_csv(out / 'candidate_cv.csv', index=False)
    selected = table.iloc[0].model
    print('SCREEN', table.head(12).to_string(index=False), flush=True)
    nested = np.zeros(len(y))
    rows = []
    for fold, (tr, va) in enumerate(folds):
        assert set(groups[tr]).isdisjoint(groups[va])
        assert set(parents[tr]).isdisjoint(parents[va])
        inner = list(GroupKFold(3, shuffle=True, random_state=314).split(tr, y[tr], groups[tr]))
        scores = ranking(cv_all(k, y, tr, inner), y[tr])
        choice = scores.iloc[0].model
        key, alpha = spec(choice)
        nested[va] = predict(k[key], y, tr, va, alpha)
        row = {'fold': fold, 'selected': choice, 'MSE': float(np.mean((nested[va]-y[va])**2)),
               'legacy_MSE': float(np.mean((allpred['legacy__1.0'][va]-y[va])**2))}
        rows.append(row)
        scores.to_csv(out / f'inner_search_{fold}.csv', index=False)
        print('NESTED', row, flush=True)
    pd.DataFrame(rows).to_csv(out / 'nested_folds.csv', index=False)
    # Fixed winner and baselines on new split seeds. These are stability diagnostics,
    # since this development dataset has also been used by previous experiments.
    stability = []
    eval_names = [selected, 'legacy__1.0', 'taut__1.0']
    for scheme in ['scaffold', 'parent']:
        for seed in [17, 43, 101, 2026, 2027]:
            if scheme == 'scaffold':
                ff = list(GroupKFold(5, shuffle=True, random_state=seed).split(ids, y, groups))
            else:
                ff = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(ids, y == 10000, parents))
            for name in eval_names:
                key, alpha = spec(name)
                p = np.zeros(len(y))
                for tr, va in ff:
                    assert set(parents[tr]).isdisjoint(parents[va])
                    p[va] = predict(k[key], y, tr, va, alpha)
                stability.append({'scheme': scheme, 'seed': seed, 'model': name, 'MSE': float(np.mean((p-y)**2))})
    pd.DataFrame(stability).to_csv(out / 'stability.csv', index=False)
    key, alpha = spec(selected)
    dual = scipy.linalg.solve(k[key] + alpha*np.eye(len(y)), y, assume_a='pos')
    pred = np.maximum(kt[key] @ dual, 0.)
    sub = sample.merge(pd.DataFrame({'index': test['index'], TARGET: pred}), on='index', how='left', validate='one_to_one', sort=False)
    assert sub['index'].tolist() == sample['index'].tolist()
    assert len(sub) == len(test) and np.isfinite(sub[TARGET]).all() and sub[TARGET].ge(0).all()
    sub.to_csv(out / 'submission_kernel_v7.csv', index=False)
    pd.DataFrame({'source_row': ids+2, 'observed': y, 'selected': allpred[selected],
                  'nested': nested, 'legacy': allpred['legacy__1.0'], 'tautomer': allpred['taut__1.0']}).to_csv(out / 'oof_predictions.csv', index=False)
    split_rows = [{'source_row': int(i+2), 'fold': fold, 'scaffold': groups[i], 'parent': parents[i]}
                  for fold, (_, va) in enumerate(folds) for i in va]
    pd.DataFrame(split_rows).to_csv(out / 'splits.csv', index=False)
    joblib.dump({'selected': selected, 'dual': dual, 'train_features': f}, out / 'model.joblib', compress=3)
    metrics = {'selected_model': selected, 'selected_oof_mse': float(table.iloc[0].MSE),
               'nested_procedure_mse': float(np.mean((nested-y)**2)),
               'legacy_same_folds_mse': float(np.mean((allpred['legacy__1.0']-y)**2)),
               'tautomer_same_folds_mse': float(np.mean((allpred['taut__1.0']-y)**2)),
               'candidates': len(table), 'train_rows': len(train), 'test_rows': len(test),
               'versions': {m.__name__: m.__version__ for m in [np, pd, scipy, sklearn, rdkit]},
               'note': 'Winner OOF used for selection; nested estimate evaluates search procedure. Stability checks reuse development data. Kaggle score unknown.'}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print('FINAL', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
