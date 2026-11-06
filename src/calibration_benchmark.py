"""Focused kernel-intercept and SVM mixture comparison on fixed grouped folds."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from rdkit import Chem
from sklearn.model_selection import GroupKFold,StratifiedGroupKFold
from train_model import metrics,standardize,TARGET
from train_v2 import kernels,cross_validate,stage_fit,stage_predict,extended_features

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root);out=root/'results/calibrated_v2';out.mkdir(parents=True,exist_ok=False)
    saved=joblib.load(root/'results/v2/model_v2.joblib');F=saved['features'];K=kernels(F)
    train=pd.read_csv(root/'data/raw/train.csv');test=pd.read_csv(root/'data/raw/test.csv');sample=pd.read_csv(root/'data/raw/sample_submit.csv')
    y=train[TARGET].to_numpy(float);codes=np.where(train.IC50_uM_VACV.isna(),1,np.where(train.CC50_uM.isna(),2,0));ids=np.arange(len(y))
    std=[standardize(s) for s in train.SMILES];scaf=np.array([s[1] for s in std]);parents=np.array([Chem.MolToSmiles(Chem.MolFromSmiles(s[0]),isomericSmiles=False) for s in std])
    folds={'scaffold':list(GroupKFold(n_splits=5,shuffle=True,random_state=2718).split(ids,y,scaf)),
           'parent':list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(ids,codes==1,parents))}
    specs=[dict(name='reference',family='krr',key='m2',alpha=1.)]
    specs += [dict(name=f'center_{key}_{alpha}',family='centered',key=key,alpha=alpha) for key in ['m2','multi'] for alpha in [.1,1.,10.]]
    specs += [dict(name=f'svm_{key}_{C}_{log}',family='svm_mix',key=key,C=C,log=log) for key in ['m2','multi'] for C in [.1,1.,10.,100.] for log in [False,True]]
    rows=[];allpred={}
    for spec in specs:
        row={'model':spec['name']}
        for regime,ff in folds.items():
            pred=cross_validate(spec,ids,ff,y,codes,F,K);row[regime+'_MSE']=metrics(y,pred)['MSE'];allpred[(spec['name'],regime)]=pred
        rows.append(row)
    table=pd.DataFrame(rows).sort_values('scaffold_MSE');table.to_csv(out/'candidate_cv.csv',index=False)
    selected=next(s for s in specs if s['name']==table.iloc[0].model)
    b=stage_fit(selected,ids,y,codes,F,K);Ft=extended_features([standardize(s)[0] for s in test.SMILES]);Kt=kernels(Ft,F)
    pred=stage_predict(b,np.arange(len(test)),Ft,Kt)
    sub=sample[['index']].merge(pd.DataFrame({'index':test['index'],TARGET:pred}),on='index',validate='one_to_one',sort=False,how='left');sub.to_csv(out/'submission_v2.csv',index=False)
    joblib.dump({'bundle':b,'features':F,'spec':selected},out/'model_v2.joblib',compress=3)
    reference=next(r for r in rows if r['model']=='reference')
    report={'selected_spec':selected,'selected_scores':table.iloc[0].to_dict(),'reference_scores':reference,'public_score':None,
            'selection_note':'Grouped OOF scores used for candidate selection; not an independent holdout. All selected predictions use SMILES only.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    pd.DataFrame({'source_row':ids+2,'observed':y,'scaffold_prediction':allpred[(selected['name'],'scaffold')],'parent_prediction':allpred[(selected['name'],'parent')]}).to_csv(out/'oof_predictions.csv',index=False)

if __name__=='__main__':main()
