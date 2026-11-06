"""Fit a fixed 50:50 ensemble of v6 and a feature-Morgan code classifier.

The physical component and frozen external lookup follow the supplied v6.
The model bundle supports inference without train/test CSVs or an external API.
Run: python src/train_v7.py --out results/v7_final
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import solve

import train_v5 as T
import train_kernel_v7 as F


@lru_cache(maxsize=16384)
def source_key(smiles):
    m = T.std(str(smiles))
    return None if m is None else T.taut_key(m)


def presence_keys(path):
    ep = pd.read_csv(path, usecols=['SMILES', 'endpoint', 'relation', 'is_vacv_assay']).dropna(subset=['SMILES'])
    if ep.is_vacv_assay.dtype != bool:
        raise ValueError('is_vacv_assay must contain actual booleans')
    good = ep[(ep.endpoint == 'CC50') | (ep.relation.eq('=') & (ep.is_vacv_assay | ep.endpoint.isin(['IC50', 'EC50'])))]
    return {s for v in good.SMILES.unique() if (s := source_key(v)) is not None}


def matrices(smiles):
    smiles = list(smiles)
    mols = [T.std(str(s)) for s in smiles]
    if any(m is None for m in mols):
        raise ValueError('Invalid SMILES')
    bits = np.array([T.mg.GetFingerprintAsNumPy(m) for m in mols], float)
    torsion = np.array([T.tt.GetCountFingerprintAsNumPy(m) for m in mols], float)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=8192, includeChirality=True,
                    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    feature = np.array([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(F.standardize(s)[0])) for s in smiles], float)
    return {'bits': bits, 'torsion': torsion, 'feature': feature}


def legacy_kernel(a, b):
    return .5*(T.tanimoto(a['bits'], b['bits']) + T.minmax(a['torsion'], b['torsion']))


def subset(a, ids):
    return {key: value[ids] for key, value in a.items()}


def infer(model, smiles, details=False):
    smiles = list(smiles)
    a = matrices(smiles)
    keys = np.array([source_key(s) for s in smiles], object)
    flag = np.array([key in model['source_presence'] for key in keys])
    ka = legacy_kernel(a, model['A_reference'])
    kf = T.tanimoto(a['feature'], model['A_reference']['feature'])
    p0 = np.clip(model['A_mean'] + ka@model['A_legacy_dual'], 0, 1)
    p1 = np.clip(model['A_mean'] + kf@model['A_feature_dual'], 0, 1)
    p0[flag] = model['A_flag_probability']
    p1[flag] = model['A_flag_probability']
    p = (1-model['feature_weight'])*p0 + model['feature_weight']*p1
    kb = legacy_kernel(a, model['B_reference'])
    physical = np.maximum(model['B_smear']*np.exp(model['B_mean']+kb@model['B_dual'])-1, 0)
    # Blend the two complete predictions after their v6 clipping step. Mixing
    # probabilities before clipping would be a different, unevaluated model.
    numerical0 = np.clip(10000*p0 + np.clip(1-p0-model['zero_rate'],0,1)*physical,0,10000)
    numerical1 = np.clip(10000*p1 + np.clip(1-p1-model['zero_rate'],0,1)*physical,0,10000)
    numerical = (1-model['feature_weight'])*numerical0 + model['feature_weight']*numerical1
    ext = np.array([model['lookup'].get(key, np.nan) for key in keys])
    covered = np.isfinite(ext)
    final = np.where(covered, np.clip(np.nan_to_num(ext),0,10000), numerical)
    if details:
        return final, {'source_present':flag, 'external_lookup':covered,
                       'p_code_v6':p0,'p_code_feature':p1,'p_code_ensemble':p}
    return final


def train(out):
    import platform
    import rdkit
    import scipy
    import sklearn
    w = T.World()
    w.train = w.train[['SMILES','SI VACV']].copy()
    w.test = w.test[['index','SMILES']].copy()
    org = np.arange(w.n_tr)
    source = presence_keys(T.ROOT/'data/external/external_endpoints.csv')
    flag = np.array([s in source for s in w.tau[:w.n_tr]])
    unflagged = org[~flag]
    c = (w.y==10000).astype(float)
    reference = matrices(w.train.SMILES)
    a_ref = subset(reference, unflagged)
    mu = float(c[unflagged].mean())
    ka = legacy_kernel(a_ref,a_ref)
    kf = T.tanimoto(a_ref['feature'],a_ref['feature'])
    dual_a0 = solve(ka+.5*np.eye(len(unflagged)),c[unflagged]-mu,assume_a='pos')
    dual_a1 = solve(kf+.5*np.eye(len(unflagged)),c[unflagged]-mu,assume_a='pos')
    observed = org[(w.y!=0)&(w.y!=10000)]
    ids = np.r_[observed,w.ext_ids]
    yy = np.r_[w.y[observed],w.ext_si]
    transformed = np.log1p(yy)
    mean_b = float(transformed.mean())
    dual_b = solve(w.K[np.ix_(ids,ids)]+3.*np.eye(len(ids)),transformed-mean_b,assume_a='pos')
    oof = np.full(w.n_tr,np.nan)
    for vi in T.inner_folds(w.scf[observed],5,0):
        va = observed[vi]
        tr = np.setdiff1d(org,va)
        mask = T.ext_allowed(w,va)
        oof[va] = T.fit_B_core(w,tr,mask,np.ones(w.n_ext),va,'log',3.,2000.)
    assert np.isfinite(oof[observed]).all()
    smear = float(np.mean(np.exp(np.log1p(w.y[observed])-oof[observed])))
    b_ref = matrices(list(w.train.SMILES.iloc[observed])+list(w.ext.SMILES))
    # This checks that portable inference uses the same representation as v6.
    assert np.max(np.abs(legacy_kernel(b_ref,b_ref)-w.K[np.ix_(ids,ids)])) < 1e-12
    lookup = {key:float(value) for key,value in T.lookup_table(w).items()}
    model = {'format_version':1,'name':'v7_v6_feature_blend50','feature_weight':.5,
             'A_reference':a_ref,'A_mean':mu,'A_legacy_dual':dual_a0,'A_feature_dual':dual_a1,
             'A_flag_probability':float((c[flag].sum()+.5)/(flag.sum()+2)),
             'B_reference':b_ref,'B_mean':mean_b,'B_dual':dual_b,'B_smear':smear,
             'zero_rate':float(np.mean(w.y==0)),'source_presence':source,'lookup':lookup}
    joblib.dump(model,out/'model.joblib',compress=3)
    pred,audit = infer(model,w.test.SMILES,details=True)
    submission = pd.DataFrame({'index':w.test['index'],'SI VACV':pred})
    sample = pd.read_csv(T.ROOT/'data/raw/sample_submit.csv',usecols=['index'])
    assert submission['index'].tolist()==sample['index'].tolist()
    assert submission['index'].is_unique and len(submission)==597 and np.isfinite(pred).all()
    submission.to_csv(out/'submission_v7.csv',index=False)
    pd.DataFrame({'index':w.test['index'],**audit}).to_csv(out/'prediction_audit.csv',index=False)
    manifest = {'model':'v7_v6_feature_blend50','feature_weight':.5,'code_alpha':.5,'physical_alpha':3.,
                'physical_transform':'log1p with training-only inner scaffold OOF Duan smearing',
                'external_weights':{'V':1.,'C':1.},'external_lookup':'same geometric-mean V/C lookup as supplied v6',
                'train_rows':w.n_tr,'test_rows':w.n_te,'external_training_rows':w.n_ext,
                'source_flag_train':int(flag.sum()),'source_flag_test':int(audit['source_present'].sum()),
                'lookup_test':int(audit['external_lookup'].sum()),'allowed_organizer_train_columns':['SMILES','SI VACV'],
                'allowed_organizer_test_columns':['SMILES','index'],'public_v7_mse':None,
                'python':platform.python_version(),
                'versions':{m.__name__:m.__version__ for m in [np,pd,rdkit,scipy,sklearn]},
                'input_sha256':{str(p.relative_to(T.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [T.ROOT/'data/raw/train.csv',T.ROOT/'data/raw/test.csv',T.ROOT/'data/external/external_endpoints.csv',
                              T.ROOT/'data/external/external_si_database.csv',T.ROOT/'data/external_curated/vacv_si.csv']}}
    (out/'model_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',default='results/v7_final')
    args = ap.parse_args()
    out = T.ROOT/args.out
    out.mkdir(parents=True,exist_ok=False)
    train(out)


if __name__=='__main__':
    main()
