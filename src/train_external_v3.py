"""Strict SMILES/target training with publication-verified external SI ablation."""
import argparse,json
from functools import lru_cache
from pathlib import Path
import numpy as np,pandas as pd,joblib
from rdkit import Chem,RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold
from train_model import standardize,featurize,tanimoto,metrics,TARGET
RDLogger.DisableLog('rdApp.warning')

@lru_cache(maxsize=16384)
def canonical(s,taut=False):
 s,g,n=standardize(s)
 if taut:s=Chem.MolToSmiles(rdMolStandardize.TautomerEnumerator().Canonicalize(Chem.MolFromSmiles(s)),isomericSmiles=True)
 return s,g

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root);out=root/'results/external_v3';out.mkdir(parents=True,exist_ok=False)
 # Explicit allowlist: no organizer concentrations, missingness or assay metadata loaded.
 train=pd.read_csv(root/'data/raw/train.csv',usecols=['SMILES',TARGET]);test=pd.read_csv(root/'data/raw/test.csv',usecols=['index','SMILES']);sample=pd.read_csv(root/'data/raw/sample_submit.csv',usecols=['index'])
 ext=pd.read_csv(root/'data/external_curated/vacv_si.csv');ext=ext[ext.cell_line.eq('Vero')].copy()
 testkeys={canonical(s,True)[0] for s in test.SMILES};ext['canonical_key']=[canonical(s,True)[0] for s in ext.SMILES]
 blocked=ext[ext.canonical_key.isin(testkeys)].copy();ext=ext[~ext.canonical_key.isin(testkeys)].copy()
 blocked.to_csv(out/'external_test_overlap_excluded.csv',index=False);ext.to_csv(out/'external_training_bank.csv',index=False)
 y=train[TARGET].to_numpy(float);ye=ext.SI.to_numpy(float);groups=np.array([standardize(s)[1] for s in train.SMILES]);ge=np.array([standardize(s)[1] for s in ext.SMILES]);ids=np.arange(len(y))
 folds=list(GroupKFold(n_splits=5,shuffle=True,random_state=2718).split(ids,y,groups));rows=[];preds={};splits=[];matrices={}
 for taut in [False,True]:
  X=featurize([canonical(s,taut)[0] for s in train.SMILES])[0];E=featurize([canonical(s,taut)[0] for s in ext.SMILES])[0];T=featurize([canonical(s,taut)[0] for s in test.SMILES])[0]
  Z=np.vstack([X,E]);K=tanimoto(Z,Z);Kt=tanimoto(T,Z);matrices[taut]=(Z,K,Kt)
  for weight in [0.,.05,.2]:
   name=f'taut_{taut}_external_{weight}';pred=np.zeros(len(y))
   for f,(tr,va) in enumerate(folds):
    keep=np.where(~np.isin(ge,groups[va]))[0] if weight else np.array([],dtype=int)
    fit=np.r_[tr,len(y)+keep];target=np.r_[y[tr],ye[keep]];w=np.r_[np.ones(len(tr)),np.full(len(keep),weight)]
    assert set(groups[va]).isdisjoint(ge[keep]);assert set(groups[va]).isdisjoint(groups[tr])
    model=KernelRidge(alpha=1.,kernel='precomputed').fit(K[np.ix_(fit,fit)],target,sample_weight=w)
    pred[va]=np.maximum(model.predict(K[np.ix_(va,fit)]),0)
    splits.append(dict(model=name,fold=f,train_rows=len(tr),validation_rows=len(va),external_rows=len(keep)))
   preds[name]=pred;rows.append(dict(model=name,tautomer=taut,external_weight=weight,**metrics(y,pred)));print(name,metrics(y,pred)['MSE'],flush=True)
 table=pd.DataFrame(rows).sort_values('MSE');table.to_csv(out/'ablation.csv',index=False);best=table.iloc[0];taut=bool(best.tautomer);weight=float(best.external_weight);Z,K,Kt=matrices[taut]
 keep=np.arange(len(ext)) if weight else np.array([],dtype=int);fit=np.r_[ids,len(y)+keep];targets=np.r_[y,ye[keep]];w=np.r_[np.ones(len(y)),np.full(len(keep),weight)]
 model=KernelRidge(alpha=1.,kernel='precomputed').fit(K[np.ix_(fit,fit)],targets,sample_weight=w);pred=np.maximum(model.predict(Kt[:,fit]),0)
 sub=sample.merge(pd.DataFrame({'index':test['index'],TARGET:pred}),on='index',how='left',validate='one_to_one',sort=False)
 assert len(sub)==597 and sub['index'].tolist()==sample['index'].tolist() and np.isfinite(sub[TARGET]).all()
 sub.to_csv(out/'submission_v3.csv',index=False);joblib.dump({'model':model,'fingerprints':Z[fit],'tautomer':taut},out/'model.joblib',compress=3)
 pd.DataFrame(splits).to_csv(out/'fold_counts.csv',index=False);pd.DataFrame({'source_row':ids+2,'observed':y,**preds}).to_csv(out/'oof_predictions.csv',index=False)
 base=next(x for x in rows if x['model']=='taut_False_external_0.0')
 report={'selected':best.to_dict(),'baseline_same_folds':base,'external_curated_rows':56,'external_vero_rows_before_overlap_filter':len(ext)+len(blocked),'external_rows_excluded_test_overlap':len(blocked),'external_training_bank_rows':len(ext),'external_rows_final_fit':len(keep),'organizer_columns_loaded':['SMILES',TARGET],'test_columns_loaded':['index','SMILES'],'auxiliary_concentration_targets':False,'fold_external_scaffold_overlap':0,'test_external_structure_overlap':0,'public_score':None,'selection_note':'OOF scores used for selection among six prespecified ablations; no independent new holdout claim.'}
 (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
