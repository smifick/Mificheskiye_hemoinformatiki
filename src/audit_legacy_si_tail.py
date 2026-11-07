"""Re-evaluate a frozen historical v6/v7 branch on the current paired folds.

External/source-presence models are historical research comparators only. No
external lookup at validation structures is allowed in this audit.
"""
import argparse
import hashlib
import importlib
import json
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from si_tail import scaffold_folds, smiles_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-root', required=True)
    parser.add_argument('--out', default='results/si_tail_evidence/legacy_audit')
    parser.add_argument('--seed', type=int, default=2718)
    args = parser.parse_args()
    legacy = Path(args.legacy_root).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(legacy/'src'))
    T, V = importlib.import_module('train_v5'), importlib.import_module('train_v6')
    F = importlib.import_module('train_kernel_v7')
    original_read = pd.read_csv

    def allowlist_read(path, *positional, **kwargs):
        if Path(path).resolve() == (legacy/'data/raw/train.csv').resolve():
            kwargs['usecols'] = ['SMILES', 'SI VACV']
        elif Path(path).resolve() == (legacy/'data/raw/test.csv').resolve():
            kwargs['usecols'] = ['index', 'SMILES']
        return original_read(path, *positional, **kwargs)

    with patch.object(pd, 'read_csv', side_effect=allowlist_read), threadpool_limits(limits=2):
        w = T.World()
        current=original_read(Path.cwd()/'data/raw/train.csv',usecols=['SMILES','SI VACV'])
        if current.SMILES.tolist()!=w.train.SMILES.tolist() or not np.array_equal(current['SI VACV'].to_numpy(),w.y):
            raise ValueError('Legacy train does not match current benchmark rows/labels')
        flag = V.chembl_flags(w)
        _, keys, groups = smiles_features(w.train.SMILES)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=8192, includeChirality=True,
                        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
        features = np.array([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(F.standardize(s)[0]))
                             for s in w.train.SMILES], float)
        k_feature = T.tanimoto(features, features)
        names = ['legacy_v7_external_capped', 'legacy_v7_external_uncapped',
                 'legacy_v7_no_external_labels_capped', 'legacy_v7_no_external_labels_uncapped']
        predictions = {name: np.zeros(w.n_tr) for name in names}
        records,components = [],[]
        c = (w.y == 10000).astype(float)
        test_keys=set(w.tau[w.n_tr:w.n_tr+w.n_te])
        for fold, (train, valid) in enumerate(scaffold_folds(groups, 5, args.seed)):
            assert set(w.scf[train]).isdisjoint(w.scf[valid])
            assert set(w.tau[train]).isdisjoint(w.tau[valid])
            allowed_ext = (~np.isin(w.tau[w.ext_ids], w.tau[valid]) &
                           ~np.isin(w.scf[w.ext_ids], w.scf[valid]))
            allowed_ext &= ~np.isin(w.tau[w.ext_ids],list(test_keys))
            p_legacy = V.code_prob(w, flag, train, valid, .5)
            unflagged = train[~flag[train]]
            mu = float(c[unflagged].mean())
            dual = solve(k_feature[np.ix_(unflagged, unflagged)] + .5*np.eye(len(unflagged)),
                         c[unflagged]-mu, assume_a='pos')
            p_feature = np.clip(mu+k_feature[np.ix_(valid, unflagged)]@dual, 0, 1)
            flagged = train[flag[train]]
            p_feature[flag[valid]] = (c[flagged].sum()+.5)/(len(flagged)+2)
            zero_rate = float(np.mean(w.y[train] == 0))
            for use_external in [True, False]:
                r = T.fit_B(w, train, allowed_ext if use_external else np.zeros(w.n_ext, bool),
                            np.ones(w.n_ext), valid, V.CFG)
                numerical = [10000*p + np.clip(1-p-zero_rate, 0, 1)*r for p in [p_legacy, p_feature]]
                physical_valid=np.flatnonzero((w.y[valid]>0)&(w.y[valid]!=10000))
                if len(physical_valid):
                    j=physical_valid[np.argmax(w.y[valid[physical_valid]])]
                    components.append({'seed':args.seed,'fold':fold,'row_index':int(valid[j]),
                                       'with_external_labels':use_external,'source_present':bool(flag[valid[j]]),
                                       'p_code_legacy':float(p_legacy[j]),'p_code_feature':float(p_feature[j]),
                                       'physical_SI':float(r[j]),'observed':float(w.y[valid[j]])})
                suffix = 'external' if use_external else 'no_external_labels'
                for bounded in [True, False]:
                    name = f'legacy_v7_{suffix}_{"capped" if bounded else "uncapped"}'
                    pred = np.mean([np.clip(v, 0, 10000) if bounded else v for v in numerical], axis=0)
                    predictions[name][valid] = pred
                    records.append({'fold': fold, 'model': name, 'MSE': float(np.mean((pred-w.y[valid])**2)),
                                    'external_train_rows': int(allowed_ext.sum()) if use_external else 0,
                                    'above_upper_cap_before_projection': int(np.any(np.stack(numerical)>10000, axis=0).sum()),
                                    'max_before_projection': float(np.max(numerical)),
                                    'organizer_max_train': float(w.y[train].max())})
            print('Historical audit fold', fold, flush=True)
    pd.DataFrame(records).to_csv(out/'fold_metrics.csv', index=False)
    pd.DataFrame(components).to_csv(out/'extreme_components.csv',index=False)
    pd.DataFrame({'row_index': np.arange(w.n_tr), 'observed': w.y, **predictions}).to_csv(out/'oof_predictions.csv', index=False)
    metrics = [{'model': name, 'MSE': float(np.mean((pred-w.y)**2)),
                'extreme_prediction': float(pred[np.argmax(w.y)])} for name, pred in predictions.items()]
    pd.DataFrame(metrics).to_csv(out/'comparison.csv', index=False)
    actual_commit=subprocess.check_output(['git','-C',str(legacy),'rev-parse','HEAD'],text=True).strip()
    protocol = {'legacy_commit': actual_commit, 'outer_seed': args.seed, 'outer_folds': 5,
                'validation_external_molecule_or_scaffold_overlap': 0,
                'validation_external_lookup': False,
                'external_competition_test_identities_excluded':True,
                'source_presence_feature': True,
                'competition_eligibility': 'Historical comparator: source-presence metadata used as a feature, outside current structure-only modeling protocol; unreviewed tier C pairs are not automatically admitted',
                'source_sha256': {name: hashlib.sha256((legacy/'src'/name).read_bytes()).hexdigest()
                                  for name in ['train_v5.py', 'train_v6.py', 'train_v7.py', 'train_kernel_v7.py']}}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
