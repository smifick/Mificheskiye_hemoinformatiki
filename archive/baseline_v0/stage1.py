"""Baseline: numeric endpoint regression; no activity thresholds are invented."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def read_table(path):
    return pd.read_excel(path) if Path(path).suffix.lower() in {'.xlsx', '.xls'} else pd.read_csv(path)


def structure(value):
    if pd.isna(value):
        return None
    mol = Chem.MolFromSmiles(str(value).strip())
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    # Conservative policy: exclude multi-fragment records; do not guess active component.
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def group_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Generic scaffold groups heteroatom variants and stereoisomers conservatively.
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return 'ACYCLIC'  # all acyclic molecules stay together
    return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold), isomericSmiles=False)


def features(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, s in enumerate(smiles):
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(s)), result[i])
    return result


def scores(y, pred):
    return {'MAE': float(mean_absolute_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'R2': float(r2_score(y, pred)) if len(y) > 1 else None}


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data', required=True)
    a.add_argument('--smiles-col', default='SMILES')
    a.add_argument('--target', required=True)
    a.add_argument('--transform', choices=['none', 'log10'], default='none')
    a.add_argument('--out', required=True)
    a.add_argument('--library', help='Optional CSV/XLSX with same SMILES column')
    args = a.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)  # protect previous experiment
    config = vars(args) | {'seed': 42, 'radius': 2, 'bits': 2048,
        'python': platform.python_version(), 'rdkit': rdkit.__version__,
        'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
        'input_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest()}
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    df = read_table(args.data)
    for col in [args.smiles_col, args.target]:
        if col not in df:
            raise ValueError(f'Missing column: {col}')
    df['source_row'] = np.arange(len(df)) + 2
    df['canonical_smiles'] = df[args.smiles_col].map(structure)
    df['numeric_target'] = pd.to_numeric(df[args.target], errors='coerce')
    df['rejection_reason'] = ''
    df.loc[df.canonical_smiles.isna(), 'rejection_reason'] = 'invalid_or_multifragment_smiles'
    bad = ~np.isfinite(df.numeric_target)
    if args.transform == 'log10':
        bad |= df.numeric_target <= 0
    df.loc[bad, 'rejection_reason'] += ';missing_non_numeric_nonfinite_or_nonpositive_target'
    df[df.rejection_reason != ''].to_csv(out / 'rejected.csv', index=False)
    valid = df[df.rejection_reason == ''].copy()
    valid.to_csv(out / 'accepted_records.csv', index=False)
    # Repeated exact structures: median on original assay scale, traceable source rows.
    clean = valid.groupby('canonical_smiles', as_index=False).agg(
        target=('numeric_target', 'median'), n_measurements=('numeric_target', 'size'),
        target_min=('numeric_target', 'min'), target_max=('numeric_target', 'max'),
        source_rows=('source_row', lambda x: ','.join(map(str, x))))
    clean['scaffold_group'] = clean.canonical_smiles.map(group_key)
    clean.to_csv(out / 'cleaned.csv', index=False)
    if len(clean) < 20 or clean.scaffold_group.nunique() < 5:
        raise ValueError('Need at least 20 unique structures and 5 scaffold groups; inspect cleaned.csv')
    X = features(clean.canonical_smiles)
    np.savez_compressed(out / 'fingerprints.npz', X=X, smiles=clean.canonical_smiles.to_numpy(dtype=str))
    y = clean.target.to_numpy()
    if args.transform == 'log10':
        y = np.log10(y)
    groups = clean.scaffold_group.to_numpy()
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
    if len(test) < 2 or len(train) < 10 or len(set(groups[train])) < 3:
        raise ValueError('Scaffold holdout too small/imbalanced; inspect group sizes, do not use random split')
    assert not set(groups[train]) & set(groups[test])
    clean['split'] = 'train'
    clean.loc[test, 'split'] = 'test'
    clean.to_csv(out / 'split.csv', index=False)
    models = {'dummy': DummyRegressor(strategy='median'),
              'random_forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                      random_state=42, n_jobs=-1)}
    folds = GroupKFold(n_splits=min(5, len(set(groups[train]))))
    cv, oof = [], []
    for name, model in models.items():
        for fold, (tr, va) in enumerate(folds.split(X[train], y[train], groups[train])):
            ti, vi = train[tr], train[va]
            assert not set(groups[ti]) & set(groups[vi])
            model.fit(X[ti], y[ti])
            pred = model.predict(X[vi])
            cv.append({'model': name, 'fold': fold, 'n_train': len(ti), 'n_valid': len(vi), **scores(y[vi], pred)})
            oof.extend({'model': name, 'fold': fold, 'canonical_smiles': clean.iloc[i].canonical_smiles,
                        'observed': float(y[i]), 'prediction': float(v)} for i, v in zip(vi, pred))
    cvdf = pd.DataFrame(cv)
    cvdf.to_csv(out / 'cv_metrics.csv', index=False)
    pd.DataFrame(oof).to_csv(out / 'cv_predictions.csv', index=False)
    selected = cvdf.groupby('model').MAE.mean().idxmin()
    model = models[selected]
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    report = {'selected_by_train_cv_MAE': selected, 'target_scale': args.transform,
              'train_n': len(train), 'test_n': len(test), 'test': scores(y[test], pred),
              'warning': 'Retrospective assay prediction; no clinical safety or efficacy claim.'}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame({'canonical_smiles': clean.iloc[test].canonical_smiles.to_numpy(),
                  'observed': y[test], 'prediction': pred}).to_csv(out / 'test_predictions.csv', index=False)
    joblib.dump({'model': model, 'config': config, 'selected_model': selected}, out / 'model_train_only.joblib')
    if args.library:
        lib = read_table(args.library)
        lib['canonical_smiles'] = lib[args.smiles_col].map(structure)
        lib[lib.canonical_smiles.isna()].to_csv(out / 'library_rejected.csv', index=False)
        lib = lib.dropna(subset=['canonical_smiles']).copy()
        lib['seen_in_training'] = lib.canonical_smiles.isin(clean.iloc[train].canonical_smiles)
        lib['seen_in_dataset'] = lib.canonical_smiles.isin(clean.canonical_smiles)
        if len(lib):
            lib['prediction'] = model.predict(features(lib.canonical_smiles))
        lib.to_csv(out / 'library_predictions.csv', index=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
