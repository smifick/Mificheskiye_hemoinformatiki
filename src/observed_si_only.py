"""Isolated complete-concentration SI experiment: one fixed raw KRR model.

Concentrations determine the TRAINING cohort only. Estimator and inference
inputs come strictly from SMILES. No external data, gate, ensemble, inherited
model, log transform or parameter search is involved.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits

TARGET='SI VACV'
EXPERIMENT='observed_si_only_20261107'
ALPHA=1.

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def representations(smiles):
    """Match historical v3 tautomer-Morgan normalization and fingerprints."""
    enumerator=rdMolStandardize.TautomerEnumerator()
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    bits=[];keys=[];groups=[]
    for source in smiles:
        if not isinstance(source,str):raise ValueError('SMILES must be a string')
        mol=Chem.MolFromSmiles(source)
        if mol is None:raise ValueError('Invalid SMILES')
        mol=max(Chem.GetMolFrags(mol,asMols=True),key=lambda m:(any(a.GetAtomicNum()==6 for a in m.GetAtoms()),m.GetNumHeavyAtoms()))
        mol=rdMolStandardize.Uncharger().uncharge(mol);Chem.SanitizeMol(mol)
        scaffold=MurckoScaffold.GetScaffoldForMol(mol)
        groups.append(Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold),isomericSmiles=False)
                      if scaffold.GetNumAtoms() else 'ACYCLIC')
        mol=Chem.MolFromSmiles(Chem.MolToSmiles(mol,isomericSmiles=True))
        key=Chem.MolToSmiles(enumerator.Canonicalize(mol),isomericSmiles=True)
        keys.append(key);bits.append(generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(key)))
    return np.asarray(bits,float),np.asarray(keys,object),np.asarray(groups,object)

def tanimoto(a,b):
    dot=a@b.T
    return dot/np.maximum(a.sum(1)[:,None]+b.sum(1)[None,:]-dot,1.)

def fit_estimator(kernel,y,training_rows):
    return KernelRidge(alpha=ALPHA,kernel='precomputed').fit(kernel[np.ix_(training_rows,training_rows)],y[training_rows])

def predict_estimator(estimator,cross_kernel):
    return np.maximum(estimator.predict(cross_kernel),0.)

def infer(bundle,smiles):
    if bundle['experiment']!=EXPERIMENT:raise ValueError('Wrong experiment bundle')
    x,_,_=representations(smiles)
    return predict_estimator(bundle['estimator'],tanimoto(x,bundle['reference_fingerprints']))

def audit_train(train):
    ic=pd.to_numeric(train.IC50_uM_VACV,errors='raise');cc=pd.to_numeric(train.CC50_uM,errors='raise')
    y=pd.to_numeric(train[TARGET],errors='raise').to_numpy(float)
    if not np.isfinite(y).all() or (y<0).any():raise ValueError('Invalid target')
    missing_ic=ic.isna().to_numpy();missing_cc=cc.isna().to_numpy();keep=~(missing_ic|missing_cc)
    ratio=(cc/ic).to_numpy(float);delta=np.abs(ratio[keep]-y[keep])
    audit=dict(rows=len(train),missing_IC50=int(missing_ic.sum()),missing_CC50=int(missing_cc.sum()),
               missing_both=int((missing_ic&missing_cc).sum()),complete_rows=int(keep.sum()),
               missing_IC50_all_SI_10000=bool(np.all(y[missing_ic]==10000)),
               missing_CC50_all_SI_zero=bool(np.all(y[missing_cc]==0)),
               SI_10000_rows=int((y==10000).sum()),SI_zero_rows=int((y==0).sum()),
               SI_10000_with_IC50_present=int(((y==10000)&~missing_ic).sum()),
               SI_zero_with_CC50_present=int(((y==0)&~missing_cc).sum()),
               complete_concentrations_nonpositive=int(((ic[keep]<=0)|(cc[keep]<=0)).sum()),
               complete_concentrations_nonfinite=int((~np.isfinite(ic[keep])|~np.isfinite(cc[keep])).sum()),
               ratio_max_absolute_difference=float(delta.max()),
               ratio_max_relative_difference=float((delta/np.abs(ratio[keep])).max()),
               ratio_matches_within_half_percent=int(np.isclose(ratio[keep],y[keep],rtol=.005,atol=1e-12).sum()),
               target_recalculated=False,
               conclusion='Exact missingness/code association; absent endpoints do not define a measured SI.')
    return keep,y,missing_ic,missing_cc,audit

def metric_row(name,scope,mask,y,p):
    yy,pp=y[mask],p[mask]
    return dict(training_arm=name,evaluation_scope=scope,n=int(mask.sum()),
                MSE=float(mean_squared_error(yy,pp)),RMSE=float(np.sqrt(mean_squared_error(yy,pp))),
                MAE=float(mean_absolute_error(yy,pp)),mean_prediction=float(pp.mean()))

def train_experiment(root,out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(experiment=EXPERIMENT,model='raw Tanimoto KernelRidge',alpha=ALPHA,
                  features='tautomer-normalized Morgan radius2 bits2048 chirality',
                  filtered_cohort='both organizer IC50_uM_VACV and CC50_uM observed',
                  control='same fixed model, full training portion of each original fold',
                  validation='unchanged archived5 scaffold folds; paired arms predict identical validation rows',
                  primary_metrics=['unweighted raw MSE on all320','unweighted raw MSE on complete237'],
                  final_model='one fit using complete237 only',external_data=False,ensemble=False,
                  target_transform='none',sample_weight='none',hyperparameter_search=False,
                  upper_prediction_cap=None,source_presence=False,mixture_gate=False,
                  feature_columns=['SMILES'],cohort_audit_columns=['IC50_uM_VACV','CC50_uM'],
                  test_columns=['index','SMILES'],independent_holdout=False,
                  note='Historical rows reused; control is CV only and is never blended into filtered model.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    train_path=root/'data/raw/train.csv';test_path=root/'data/raw/test.csv'
    train=pd.read_csv(train_path,usecols=['SMILES',TARGET,'IC50_uM_VACV','CC50_uM'])
    test=pd.read_csv(test_path,usecols=['index','SMILES'])
    sample=pd.read_csv(root/'data/raw/sample_submit.csv',usecols=['index'])
    keep,y,missing_ic,missing_cc,audit=audit_train(train)
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2))
    ids=np.arange(len(train));x,keys,groups=representations(train.SMILES)
    split_path=root/'results/v2/validation_splits.csv'
    splits=pd.read_csv(split_path).sort_values('source_row').reset_index(drop=True)
    if not np.array_equal(splits.source_row.to_numpy(),ids+2):raise ValueError('Archived folds have different rows')
    folds=splits.outer_fold.to_numpy(int);kernel=tanimoto(x,x)
    cohort=pd.DataFrame(dict(source_row=ids+2,SMILES=train.SMILES,observed=y,canonical_smiles=keys,
                             scaffold_group=groups,fold=folds,in_filtered_training=keep))
    cohort.to_csv(out/'cohort_and_splits.csv',index=False)
    cohort.loc[keep,['source_row','SMILES','observed','canonical_smiles','scaffold_group']].rename(columns={'observed':TARGET}).to_csv(out/'training_data_filtered237.csv',index=False)
    excluded=train.loc[~keep].copy();excluded.insert(0,'source_row',ids[~keep]+2)
    excluded['exclusion_reason']=np.where(missing_ic[~keep],'missing_IC50','missing_CC50')
    excluded.to_csv(out/'excluded_rows83.csv',index=False)
    predictions={name:np.full(len(y),np.nan) for name in ['control_all320','filtered237_only']}
    fold_stats=[]
    for fold in sorted(set(folds)):
        tr=ids[folds!=fold];va=ids[folds==fold]
        assert set(groups[tr]).isdisjoint(groups[va])
        assert set(keys[tr]).isdisjoint(keys[va])
        for name in predictions:
            fit=tr if name=='control_all320' else tr[keep[tr]]
            estimator=fit_estimator(kernel,y,fit)
            p=predict_estimator(estimator,kernel[np.ix_(va,fit)])
            predictions[name][va]=p
            for scope,valid in [('all_original_labels',va),('observed_SI_only',va[keep[va]])]:
                fold_stats.append(dict(training_arm=name,fold=int(fold),training_rows=len(fit),
                                       evaluation_scope=scope,validation_rows=len(valid),
                                       MSE=float(mean_squared_error(y[valid],predictions[name][valid]))))
    assert all(np.isfinite(p).all() for p in predictions.values())
    pd.DataFrame(fold_stats).to_csv(out/'fold_metrics.csv',index=False)
    oof=cohort.copy()
    for name,p in predictions.items():oof[name]=p
    oof.to_csv(out/'oof_predictions.csv',index=False)
    metrics=[]
    for name,p in predictions.items():
        for scope,mask in [('all_original_labels',np.ones(len(y),bool)),('observed_SI_only',keep),
                           ('excluded_codes_only',~keep),('observed_SI_le_100',keep&(y<=100)),
                           ('observed_SI_gt_100',keep&(y>100)),('largest_observed_SI',y==y[keep].max())]:
            metrics.append(metric_row(name,scope,mask,y,p))
    comparison=pd.DataFrame(metrics);comparison.to_csv(out/'comparison.csv',index=False)
    effects={}
    for scope in comparison.evaluation_scope.unique():
        part=comparison[comparison.evaluation_scope==scope].set_index('training_arm')
        control=float(part.loc['control_all320','MSE']);filtered=float(part.loc['filtered237_only','MSE'])
        effects[scope]=dict(control_MSE=control,filtered_MSE=filtered,improvement_pct=100*(1-filtered/control))
    (out/'effect_summary.json').write_text(json.dumps(effects,indent=2))
    # This is the only final model; control models exist only inside CV.
    fit=ids[keep];estimator=fit_estimator(kernel,y,fit)
    bundle=dict(experiment=EXPERIMENT,format_version=1,estimator=estimator,
                reference_fingerprints=x[fit],reference_source_rows=(fit+2),
                reference_smiles=train.SMILES.iloc[fit].tolist(),target=TARGET,
                final_training_rows=len(fit),feature_columns=['SMILES'])
    joblib.dump(bundle,out/'model_filtered237.joblib',compress=3)
    p=infer(bundle,test.SMILES)
    submission=pd.DataFrame({'index':test['index'],TARGET:p})
    if not submission['index'].is_unique or not submission['index'].equals(sample['index']):raise ValueError('Submission ID mismatch')
    assert np.isfinite(p).all() and (p>=0).all()
    submission.to_csv(out/'submission_filtered237.csv',index=False)
    manifest=dict(experiment=EXPERIMENT,python=platform.python_version(),
                  packages={s:importlib.metadata.version(s) for s in ['numpy','pandas','scipy','rdkit','scikit-learn','joblib','threadpoolctl']},
                  input_sha256={str(p.relative_to(root)):digest(p) for p in [train_path,test_path,split_path]},
                  model_sha256=digest(out/'model_filtered237.joblib'),
                  submission_sha256=digest(out/'submission_filtered237.csv'),
                  final_training_rows=len(fit),excluded_rows=int((~keep).sum()),submission_rows=len(submission),
                  public_Kaggle_MSE=None,ensemble=False,previous_submissions_modified=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(audit,indent=2));print(pd.DataFrame(metrics).to_string(index=False));print(json.dumps(manifest,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--out',default='results/'+EXPERIMENT)
    p=sub.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if args.command=='train':train_experiment(root,root/args.out)
    else:
        source=pd.read_csv(args.input,usecols=['index','SMILES']);bundle=joblib.load(args.model)
        pd.DataFrame({'index':source['index'],TARGET:infer(bundle,source.SMILES)}).to_csv(args.out,index=False)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
