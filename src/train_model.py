"""SMILES-only SI regression. Selection on inner scaffold CV, untouched outer holdout."""
import argparse, hashlib, json, platform, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import rdkit, sklearn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

RDLogger.DisableLog('rdApp.warning')
DESC = Descriptors._descList
TARGET = 'SI VACV'


def standardize(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        raise ValueError(f'Invalid SMILES: {s}')
    parts = Chem.GetMolFrags(m, asMols=True)
    # Prefer a carbon-containing parent, then the greatest heavy atom count.
    m = max(parts, key=lambda x: (any(a.GetAtomicNum()==6 for a in x.GetAtoms()), x.GetNumHeavyAtoms()))
    m = rdMolStandardize.Uncharger().uncharge(m)
    Chem.SanitizeMol(m)
    canonical = Chem.MolToSmiles(m, isomericSmiles=True)
    sc = MurckoScaffold.GetScaffoldForMol(m)
    group = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(sc), isomericSmiles=False) if sc.GetNumAtoms() else 'ACYCLIC'
    return canonical, group, len(parts)


def featurize(smiles):
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    X=np.zeros((len(smiles),2048),dtype=np.float64)
    D=np.zeros((len(smiles),len(DESC)),dtype=np.float64)
    for i,s in enumerate(smiles):
        m=Chem.MolFromSmiles(s)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m),X[i])
        for j,(_,fn) in enumerate(DESC):
            try: D[i,j]=fn(m)
            except Exception: D[i,j]=np.nan
    D[~np.isfinite(D)]=np.nan
    return X,D


def tanimoto(A,B):
    dot=A@B.T
    return dot/np.maximum(A.sum(1)[:,None]+B.sum(1)[None,:]-dot,1)


def metrics(y,p):
    p=np.maximum(p,0)
    return dict(MSE=float(mean_squared_error(y,p)),RMSE=float(np.sqrt(mean_squared_error(y,p))),MAE=float(mean_absolute_error(y,p)),
                R2=float(r2_score(y,p)),RMSLE=float(np.sqrt(mean_squared_error(np.log1p(y),np.log1p(p)))))


def candidate_specs():
    specs=[]
    specs.append(('dummy_mean','raw','bits',DummyRegressor(strategy='mean')))
    specs.append(('dummy_log','log1p','bits',DummyRegressor(strategy='mean')))
    for trans in ['raw','log1p']:
        specs += [(f'rf_{trans}',trans,'both',make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=300,max_depth=12,min_samples_leaf=2,max_features=0.5,n_jobs=2,random_state=42))),
                  (f'extra_{trans}',trans,'both',make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=400,max_depth=14,min_samples_leaf=2,max_features=0.7,n_jobs=2,random_state=42))),
                  (f'hgb_{trans}',trans,'desc',make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=10,l2_regularization=10,learning_rate=.05,random_state=42))),
                  (f'ridge_{trans}',trans,'both',make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100)))]
        for alpha in [.1,1,10]:
            specs.append((f'tanimoto_krr_{alpha}_{trans}',trans,'kernel',KernelRidge(alpha=alpha,kernel='precomputed')))
    return specs


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--test',required=True)
    p.add_argument('--sample',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    train=pd.read_csv(args.train);test=pd.read_csv(args.test);sample=pd.read_csv(args.sample)
    if TARGET not in train or list(sample.columns)!=['index',TARGET]:raise ValueError('Unexpected target/submission schema')
    y=pd.to_numeric(train[TARGET],errors='raise').to_numpy(float)
    if not np.isfinite(y).all() or (y<0).any(): raise ValueError('Invalid target')
    for d in [train,test]:
        std=[standardize(s) for s in d.SMILES]
        d['canonical_smiles']=[x[0] for x in std];d['scaffold_group']=[x[1] for x in std];d['fragment_count']=[x[2] for x in std]
    train['source_row']=np.arange(len(train))+2
    train['SI_origin']='organizer_label'
    train['SI_possible_placeholder']=train[TARGET].eq(10000)&train.IC50_uM_VACV.isna()
    train['doi']='';train['source']='organizer_train';train['source_url']='user-provided train.csv'
    train['IC50_uM']=train['IC50_uM_VACV'];train['SI']=train[TARGET]
    train.to_csv(out/'training_dataset.csv',index=False)
    test.to_csv(out/'test_structure_audit.csv',index=False)
    duplicates=train[train.duplicated('canonical_smiles',keep=False)].sort_values('canonical_smiles')
    duplicates.to_csv(out/'duplicates_audit.csv',index=False)
    ratio=train.CC50_uM/train.IC50_uM_VACV
    audit={'train_rows':len(train),'test_rows':len(test),'train_unique_parent_structures':train.canonical_smiles.nunique(),
           'test_unique_parent_structures':test.canonical_smiles.nunique(),'train_test_parent_overlap':len(set(train.canonical_smiles)&set(test.canonical_smiles)),
           'train_multifragment_rows':int((train.fragment_count>1).sum()),'train_missing_ic50':int(train.IC50_uM_VACV.isna().sum()),
           'SI_10000_rows':int((y==10000).sum()),'SI_10000_and_missing_ic50':int(train.SI_possible_placeholder.sum()),
           'ratio_comparable_rows':int(ratio.notna().sum()),'ratio_agrees_rtol_2pct':int(np.isclose(ratio,y,rtol=.02).sum()),
           'invalid_smiles':0,'generic_scaffold_groups':train.scaffold_group.nunique(),
           'external_used':False,'features':'SMILES only; no CC50, IC50, Title, strain, Culture or test labels'}
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2))
    X,D=featurize(train.canonical_smiles);Xt,Dt=featurize(test.canonical_smiles)
    K=tanimoto(X,X);Kt=tanimoto(Xt,X)
    matrices={'bits':X,'desc':D,'both':np.column_stack([X,D]),'kernel':K}
    test_matrices={'bits':Xt,'desc':Dt,'both':np.column_stack([Xt,Dt]),'kernel':Kt}
    groups=train.scaffold_group.to_numpy()
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(X,y,groups))
    assert set(groups[tr]).isdisjoint(groups[te])
    folds=list(GroupKFold(n_splits=5).split(X[tr],y[tr],groups[tr]))
    splits=train[['source_row','canonical_smiles','scaffold_group']].copy();splits['outer_split']='train';splits.loc[te,'outer_split']='holdout';splits['inner_fold']=-1
    for f,(_,va) in enumerate(folds):splits.loc[tr[va],'inner_fold']=f
    splits.to_csv(out/'validation_splits.csv',index=False)
    specs=candidate_specs();rows=[];oof_rows=[];summaries=[]
    for name,trans,kind,est in specs:
        oof=np.zeros(len(tr));A=matrices[kind]
        for f,(fit,valid) in enumerate(folds):
            a,b=tr[fit],tr[valid];assert set(groups[a]).isdisjoint(groups[b])
            model=clone(est);Y=np.log1p(y[a]) if trans=='log1p' else y[a]
            model.fit(A[np.ix_(a,a)] if kind=='kernel' else A[a],Y)
            pred=model.predict(A[np.ix_(b,a)] if kind=='kernel' else A[b])
            if trans=='log1p':pred=np.expm1(np.clip(pred,-20,20))
            pred=np.maximum(pred,0);oof[valid]=pred
            rows.append(dict(model=name,fold=f,n_train=len(a),n_valid=len(b),**metrics(y[b],pred)))
        summaries.append(dict(model=name,**metrics(y[tr],oof)))
        oof_rows.extend(dict(model=name,source_row=int(train.iloc[idx].source_row),observed=float(y[idx]),prediction=float(pred)) for idx,pred in zip(tr,oof))
        print(name,metrics(y[tr],oof),flush=True)
    summary=pd.DataFrame(summaries).sort_values('MSE');summary.to_csv(out/'cv_summary.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'cv_folds.csv',index=False);pd.DataFrame(oof_rows).to_csv(out/'cv_predictions.csv',index=False)
    selected=summary.iloc[0].model;spec=next(x for x in specs if x[0]==selected)
    _,trans,kind,est=spec;A=matrices[kind]
    model=clone(est);model.fit(A[np.ix_(tr,tr)] if kind=='kernel' else A[tr],np.log1p(y[tr]) if trans=='log1p' else y[tr])
    hp=model.predict(A[np.ix_(te,tr)] if kind=='kernel' else A[te])
    if trans=='log1p': hp=np.expm1(np.clip(hp,-20,20))
    hp=np.maximum(hp,0)
    baseline=np.full(len(te),y[tr].mean())
    report={'selected_model':selected,'selection_metric':'pooled inner scaffold OOF MSE (Kaggle metric)',
            'inner_cv':summary.iloc[0].to_dict(),'holdout':metrics(y[te],hp),'holdout_dummy_mean':metrics(y[te],baseline),
            'outer_train_rows':len(tr),'holdout_rows':len(te),'outer_scaffold_overlap':0,
            'warning':'SI=10000 may encode missing IC50; scores describe organizer labels, not validated biological SI.'}
    pd.DataFrame({'source_row':train.iloc[te].source_row,'canonical_smiles':train.iloc[te].canonical_smiles,'observed':y[te],'prediction':hp}).to_csv(out/'holdout_predictions.csv',index=False)
    # Final refit after evaluation, including all provided training rows. Test labels never used.
    final=clone(est);final.fit(A,np.log1p(y) if trans=='log1p' else y)
    pred=final.predict(test_matrices[kind]);pred=np.maximum(np.expm1(np.clip(pred,-20,20)) if trans=='log1p' else pred,0)
    smap=pd.DataFrame({'index':test['index'],TARGET:pred})
    if not test['index'].is_unique or not sample['index'].is_unique:raise ValueError('Duplicate submission IDs')
    submission=sample[['index']].merge(smap,on='index',how='left',validate='one_to_one',sort=False)
    assert len(submission)==len(sample) and np.isfinite(submission[TARGET]).all() and (submission[TARGET]>=0).all()
    assert submission['index'].tolist()==sample['index'].tolist()
    submission.to_csv(out/'submission.csv',index=False)
    near=Kt.max(axis=1);nearest=Kt.argmax(axis=1)
    pd.DataFrame({'index':test['index'],'prediction':pred,'max_train_tanimoto':near,
                  'nearest_train_smiles':train.iloc[nearest].canonical_smiles.to_numpy(),
                  'exact_parent_seen':test.canonical_smiles.isin(train.canonical_smiles)}).to_csv(out/'prediction_diagnostics.csv',index=False)
    config={'seed':42,'fingerprint_radius':2,'fingerprint_bits':2048,'include_chirality':True,
            'descriptor_names':[n for n,_ in DESC],'python':platform.python_version(),'rdkit':rdkit.__version__,
            'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
            'input_sha256':{Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [args.train,args.test,args.sample]},
            'selected_model':selected,'target_transform':trans,'feature_kind':kind,'external_used':False}
    joblib.dump({'model':final,'config':config,'train_fingerprints':X if kind=='kernel' else None},out/'model.joblib',compress=3)
    (out/'config.json').write_text(json.dumps(config,indent=2));(out/'metrics.json').write_text(json.dumps(report,indent=2))
    print('FINAL',json.dumps(report),flush=True)

if __name__=='__main__':main()
