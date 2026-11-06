"""Nested scaffold CV, external-data ablation, final fit; SMILES-only input."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
from sklearn.model_selection import GroupKFold

from .chemistry import structure_keys
from .features import MolecularFeatures
from .modeling import fit_model,predict_model


def grouped_folds(rows,groups,k,seed):
    count=len(set(groups[rows]))
    n=min(k,count)
    if n<2:raise ValueError('At least two scaffold groups are required in each training CV subset')
    return [(rows[a],rows[b]) for a,b in GroupKFold(n_splits=n,shuffle=True,random_state=seed).split(rows,groups=groups[rows])]


def select_external(ext_groups,ext_parents,groups,parents,blocked_rows):
    # Apply this cumulatively at both levels of nested validation.
    return ~np.isin(ext_groups,groups[blocked_rows]) & ~np.isin(ext_parents,parents[blocked_rows])


def canonical_table(values):
    records=[]
    for i,s in enumerate(values):
        try:records.append(structure_keys(s))
        except ValueError as error:raise ValueError(f'Invalid SMILES at input row {i+2}: {error}') from error
    return np.asarray(records,dtype=object).reshape(-1,3)


def mse(y,p):return float(np.mean((y-p)**2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',required=True)
    parser.add_argument('--external')
    parser.add_argument('--external-smiles-col',default='smiles')
    parser.add_argument('--external-target-col',default='SI')
    parser.add_argument('--out',required=True)
    parser.add_argument('--smiles-col',default='SMILES')
    parser.add_argument('--target-col',default='SI VACV')
    parser.add_argument('--mode',choices=['encoded','ratio'],default='encoded')
    parser.add_argument('--sentinel',type=float,default=10000.)
    parser.add_argument('--families',nargs='+',choices=['ridge','extra','lightgbm','catboost'],default=['ridge','extra','lightgbm','catboost'])
    parser.add_argument('--external-weight',type=float,default=.25)
    parser.add_argument('--outer-folds',type=int,default=5)
    parser.add_argument('--inner-folds',type=int,default=3)
    parser.add_argument('--mordred',action='store_true')
    parser.add_argument('--seed',type=int,default=20261107)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.external_weight<0 or args.sentinel<=0:raise ValueError('Invalid weight or sentinel')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    # Hard allowlist: forbidden organizer columns never enter the training path.
    data=pd.read_csv(args.train,usecols=[args.smiles_col,args.target_col])
    y=pd.to_numeric(data[args.target_col],errors='raise').to_numpy(float)
    if not len(y) or not np.isfinite(y).all() or (y<0).any():raise ValueError('Target must be finite and nonnegative')
    if args.mode=='ratio' and (y<=0).any():raise ValueError('Physical ratio mode requires positive SI labels')
    org=canonical_table(data[args.smiles_col]);groups=org[:,2];parents=org[:,1]
    external=(pd.read_csv(args.external,usecols=[args.external_smiles_col,args.external_target_col])
              .rename(columns={args.external_smiles_col:'smiles',args.external_target_col:'SI'})
              if args.external else pd.DataFrame(columns=['smiles','SI']))
    ey=pd.to_numeric(external.SI,errors='raise').to_numpy(float)
    if not np.isfinite(ey).all() or (ey<=0).any():raise ValueError('External physical SI must be finite and positive')
    ext=canonical_table(external.smiles)
    if args.external and external.smiles.empty:raise ValueError('External file contains no accepted pairs')
    featurizer=MolecularFeatures(mordred=args.mordred).fit(org[:,0])
    X=featurizer.transform(org[:,0])
    EX=featurizer.transform(ext[:,0]) if len(ext) else np.empty((0,X.shape[1]))
    weights=[0.]
    if len(ext) and args.external_weight>0:weights.append(args.external_weight)
    specs=[dict(name=f'{f}_external_{w}',family=f,external_weight=w) for f in args.families for w in weights]
    protocol={'mode':args.mode,'sentinel':args.sentinel,'selection_metric':'pooled MSE in original target units',
              'organizer_columns_loaded':[args.smiles_col,args.target_col],
              'external_columns_loaded':[args.external_smiles_col,args.external_target_col] if args.external else [],
              'outer_folds_requested':args.outer_folds,'inner_folds_requested':args.inner_folds,
              'seed':args.seed,'candidates':specs,'mordred':args.mordred,
              'external_semantics':'Physical SI rows augment the conditional measured-SI branch only in encoded mode.',
              'grouping':'Tautomer-normalized generic Bemis-Murcko scaffold; stereoisomers remain together.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    ids=np.arange(len(y));outer=grouped_folds(ids,groups,args.outer_folds,args.seed)
    nested=np.zeros(len(y));dummy=np.zeros(len(y));allpred={s['name']:np.zeros(len(y)) for s in specs}
    splitrows=[];foldrows=[];innerrows=[]
    for fold,(tr,va) in enumerate(outer):
        assert set(groups[tr]).isdisjoint(groups[va])
        assert set(parents[tr]).isdisjoint(parents[va])
        inner=grouped_folds(tr,groups,args.inner_folds,args.seed+fold+1)
        scores=[]
        for spec in specs:
            ip=np.zeros(len(y))
            for it,iv in inner:
                banned=np.r_[va,iv]
                keep=select_external(ext[:,2],ext[:,1],groups,parents,banned)
                fitted=fit_model(spec,X,y,it,EX,ey,keep,args.mode,args.sentinel,args.seed,args.threads)
                ip[iv]=predict_model(fitted,X[iv])
            loss=mse(y[tr],ip[tr])
            scores.append((loss,spec))
            innerrows.append(dict(outer_fold=fold,model=spec['name'],MSE=loss))
            keep=select_external(ext[:,2],ext[:,1],groups,parents,va)
            fitted=fit_model(spec,X,y,tr,EX,ey,keep,args.mode,args.sentinel,args.seed,args.threads)
            allpred[spec['name']][va]=predict_model(fitted,X[va])
        winner=min(scores,key=lambda t:t[0])[1]
        nested[va]=allpred[winner['name']][va]
        dummy[va]=y[tr].mean()
        foldrows.append(dict(fold=fold,selected=winner['name'],n_valid=len(va),MSE=mse(y[va],nested[va]),dummy_MSE=mse(y[va],dummy[va])))
        splitrows.extend(dict(source_row=int(i+2),fold=fold,parent=parents[i],scaffold=groups[i]) for i in va)
        print('OUTER',json.dumps(foldrows[-1]),flush=True)
    ranking=pd.DataFrame([dict(model=s['name'],MSE=mse(y,allpred[s['name']])) for s in specs]).sort_values('MSE')
    chosen=next(s for s in specs if s['name']==ranking.iloc[0].model)
    final=fit_model(chosen,X,y,ids,EX,ey,np.ones(len(ey),bool),args.mode,args.sentinel,args.seed,args.threads)
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
              'rdkit':rdkit.__version__,'scikit-learn':sklearn.__version__}
    for name in ['scipy','joblib','requests','lightgbm','catboost','mordredcommunity']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    bundle={'format_version':1,'featurizer':featurizer,'model':final,
            'smiles_col':args.smiles_col,'target_col':args.target_col,'versions':versions}
    joblib.dump(bundle,out/'model.joblib',compress=3)
    report={'selected':chosen,'nested_selection_MSE':mse(y,nested),'dummy_same_folds_MSE':mse(y,dummy),
            'selected_OOF_MSE':mse(y,allpred[chosen['name']]),'train_rows':len(y),'external_rows':len(ey),
            'scaffold_groups':len(set(groups)),'feature_count':X.shape[1],
            'warning':'Selected OOF scores are used for final selection; nested scores estimate the search procedure on this dataset. Kaggle score unknown.',
            'versions':versions,'input_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.train]+([args.external] if args.external else [])}}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    (out/'feature_names.json').write_text(json.dumps(list(featurizer.feature_names_),indent=2))
    ranking.to_csv(out/'candidate_cv.csv',index=False)
    pd.DataFrame(foldrows).to_csv(out/'outer_folds.csv',index=False)
    pd.DataFrame(innerrows).to_csv(out/'inner_search.csv',index=False)
    pd.DataFrame(splitrows).to_csv(out/'splits.csv',index=False)
    pd.DataFrame({'source_row':ids+2,'observed':y,'nested_prediction':nested,'selected_prediction':allpred[chosen['name']],'dummy_prediction':dummy}).to_csv(out/'oof_predictions.csv',index=False)
    with open(out/'environment.lock.txt','w') as f:
        for name,version in versions.items():
            if name!='python':f.write(f'{name}=={version}\n')
    print('FINAL',json.dumps(report),flush=True)


if __name__=='__main__':main()
