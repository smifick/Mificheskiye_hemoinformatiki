"""Grouped molecular interpolation benchmark; scaffold extrapolation reported separately."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,joblib
from rdkit import Chem
from sklearn.model_selection import StratifiedGroupKFold,GroupKFold
from train_model import standardize,metrics,TARGET
from train_v2 import extended_features,kernels,stage_fit,stage_predict,cross_validate

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--out',default='results/interpolation_v2');a=p.parse_args();root=Path(a.root);out=root/a.out;out.mkdir(parents=True,exist_ok=False)
    train=pd.read_csv(root/'data/raw/train.csv',usecols=['SMILES',TARGET]);test=pd.read_csv(root/'data/raw/test.csv',usecols=['index','SMILES']);sample=pd.read_csv(root/'data/raw/sample_submit.csv')
    std=[standardize(s) for s in train.SMILES];smiles=[s[0] for s in std];scaffolds=np.array([s[1] for s in std])
    groups=np.array([Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=False) for s in smiles])
    y=train[TARGET].to_numpy(float);codes=np.where(y==10000,1,np.where(y==0,2,0))
    F=extended_features(smiles);Ft=extended_features([standardize(s)[0] for s in test.SMILES]);K=kernels(F);Kt=kernels(Ft,F)
    ids=np.arange(len(y));folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(ids,codes==1,groups))
    specs=[dict(name=f'krr_{key}_{alpha}',family='krr',key=key,alpha=alpha) for key in ['m2','multi'] for alpha in [.003,.01,.03,.1,.3,1.]]
    specs += [dict(name=f'knn_{key}_{k}_{power}',family='knn',key=key,neighbors=k,power=power) for key in ['m2','m3','multi'] for k in [1,3,5] for power in [2,6]]
    rows=[];preds={};splits=[]
    for f,(tr,va) in enumerate(folds):
        assert set(groups[tr]).isdisjoint(groups[va])
        splits.extend(dict(source_row=int(i+2),fold=f,parent_group=groups[i],scaffold_group=scaffolds[i],max_train_tanimoto=float(K['m2'][i,tr].max())) for i in va)
    for spec in specs:
        pred=cross_validate(spec,ids,folds,y,codes,F,K);preds[spec['name']]=pred;rows.append(dict(model=spec['name'],**metrics(y,pred)))
    table=pd.DataFrame(rows).sort_values('MSE');selected=next(s for s in specs if s['name']==table.iloc[0].model);base=next(s for s in specs if s['name']=='krr_m2_1.0')
    sf=list(GroupKFold(n_splits=5,shuffle=True,random_state=2718).split(ids,y,scaffolds))
    sp=cross_validate(selected,ids,sf,y,codes,F,K);bp=cross_validate(base,ids,sf,y,codes,F,K)
    bundle=stage_fit(selected,ids,y,codes,F,K);pred=stage_predict(bundle,np.arange(len(test)),Ft,Kt)
    sub=sample[['index']].merge(pd.DataFrame({'index':test['index'],TARGET:pred}),on='index',validate='one_to_one',how='left',sort=False)
    assert len(sub)==len(sample) and sub['index'].tolist()==sample['index'].tolist() and np.isfinite(sub[TARGET]).all() and sub[TARGET].ge(0).all()
    sub.to_csv(out/'submission_v2.csv',index=False);table.to_csv(out/'candidate_cv.csv',index=False)
    split=pd.DataFrame(splits);split.to_csv(out/'validation_splits.csv',index=False)
    report={'selected_spec':selected,'selection_metric':'Grouped parent-structure OOF MSE, 5 folds, no identical parents across folds',
        'interpolation_selected':metrics(y,preds[selected['name']]),'interpolation_baseline':metrics(y,preds[base['name']]),
        'scaffold_selected':metrics(y,sp),'scaffold_baseline':metrics(y,bp),
        'interpolation_oof_median_similarity':float(split.max_train_tanimoto.median()),'test_median_similarity':float(np.median(Kt['m2'].max(1))),
        'selection_bias_note':'Candidate OOF scores are used for selection and are not independent holdout estimates. Scaffold evaluation of the selected candidate is a secondary diagnostic; Kaggle v2 score is unknown.',
        'public_v2_score':None,'features':'SMILES only; all experimental concentrations and test labels excluded.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2));pd.DataFrame({'source_row':ids+2,'observed':y,'prediction':preds[selected['name']],'baseline':preds[base['name']],'scaffold_prediction':sp,'scaffold_baseline':bp}).to_csv(out/'oof_predictions.csv',index=False)
    joblib.dump({'bundle':bundle,'features':F,'training_smiles':smiles,'spec':selected},out/'model_v2.joblib',compress=3)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
