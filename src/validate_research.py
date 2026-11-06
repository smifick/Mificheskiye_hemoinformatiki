"""Independent portable I/O checks and explicitly diagnostic context stress."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import research_heads as H
import audit_regression as A


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',default='work/chemberta_snapshots');p.add_argument('--context-stress',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parents[1];out=root/'results/audit_20261107'
    model=out/'selected_model.joblib';reference=pd.read_csv(out/'submission.csv')
    test=pd.read_csv(root/'data/raw/test.csv');sample=pd.read_csv(root/'data/raw/sample_submit.csv',usecols=['index'])
    with tempfile.TemporaryDirectory() as directory:
        work=Path(directory)
        clean=test[['index','SMILES']]
        clean.to_csv(work/'only_smiles.csv',index=False)
        poison=test.assign(CC50_uM=1e99,IC50_uM_VACV=np.nan,strain='ignored',Culture='ignored',Title='ignored')
        poison['SI VACV']=1e99
        poison.iloc[::-1].to_csv(work/'poisoned_reversed.csv',index=False)
        checks={}
        for name in ['only_smiles','poisoned_reversed']:
            command=[sys.executable,str(root/'src/predict_research.py'),'--model',str(model),'--input',str(work/(name+'.csv')),
                     '--out',str(work/(name+'_prediction.csv')),'--cache',args.cache]
            subprocess.run(command,check=True)
            pred=pd.read_csv(work/(name+'_prediction.csv')).sort_values('index').reset_index(drop=True)
            err=float(np.max(abs(pred['SI VACV']-reference['SI VACV'])))
            checks[name+'_max_error']=err
            if err>1e-7:raise AssertionError('Portable prediction mismatch')
        checks.update(rows=len(reference),schema_matches=list(reference)==['index','SI VACV'],
                      IDs_match_sample=reference['index'].equals(sample['index']),finite=bool(np.isfinite(reference['SI VACV']).all()),
                      negative_predictions=int((reference['SI VACV']<0).sum()),
                      model_sha256=A.sha(model),submission_sha256=A.sha(out/'submission.csv'))
        (out/'final_validation.json').write_text(json.dumps(checks,indent=2))
        print('PORTABLE',json.dumps(checks),flush=True)
    # Fixed low-cost syntax probe: diagnostic attribution, not model selection.
    state=joblib.load(out/'state_cache.joblib');y=state['y'];base=np.load(out/'baseline_oof.npz')
    raw=pd.read_csv(root/'data/raw/train.csv');style=[]
    for text in raw.SMILES:
        mol=Chem.MolFromSmiles(text);length=max(len(text),1)
        style.append([text==Chem.MolToSmiles(mol),any(a.GetIsAromatic() for a in mol.GetAtoms()) and not any(c in text for c in 'cnops'),
                      text.count('[')/length,text.count('@')/length,text.count('(')/length,text.count('=')/length,
                      text.count('c')/length,text.count('C')/length,length])
    style=np.asarray(style,float);prediction=np.zeros(len(y))
    for f in sorted(set(base['fold'])):
        va=np.flatnonzero(base['fold']==f);tr=np.flatnonzero(base['fold']!=f);fit=tr[~state['flag'][tr]]
        scaler=StandardScaler().fit(style[fit]);clf=LogisticRegression(C=1.,max_iter=2000).fit(scaler.transform(style[fit]),y[fit]==10000)
        prob=clf.predict_proba(scaler.transform(style[va]))[:,1]
        ff=tr[state['flag'][tr]];pf=((y[ff]==10000).sum()+.5)/(len(ff)+2.)
        prob=np.where(state['flag'][va],pf,prob)
        prediction[va]=H.complete_predictions(state,tr,va,base['r'][va],prob)
    probe={'fixed_SMILES_style_probe_MSE':A.loss(y,prediction),'v7_same_folds_MSE':A.loss(y,base['v7']),
           'status':'Diagnostic only; not used to select final model or to claim causal biology.'}
    (out/'notation_style_probe.json').write_text(json.dumps(probe,indent=2));print('STYLE_PROBE',probe,flush=True)
    if not args.context_stress:return
    # Metadata is used solely to define a stress-test split, never as a feature.
    # Union identical parents so they cannot leak between contextual groups.
    n=len(y);parent=np.arange(n)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):parent[find(a)]=find(b)
    contexts=(raw.strain.astype(str)+' | '+raw.Culture.astype(str)).to_numpy()
    molecules=np.array([Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=False) for s in state['world'].tau[:n]])
    for keys in [contexts,molecules]:
        first={}
        for i,key in enumerate(keys):
            if key in first:union(i,first[key])
            else:first[key]=i
    groups=np.array([find(i) for i in range(n)]);count=len(set(groups))
    if count<2:
        (out/'context_stress.json').write_text(json.dumps({'available':False,'reason':'All context/parent groups are connected'}));return
    results=json.loads((out/'targeted_results.json').read_text());selected=results['selected_for_submission']
    spec=next(s for s in H.candidate_specs() if s['name']==selected.removesuffix('_blend50'))
    emb=H.load_embeddings(out);preds={'v7':np.zeros(n),selected:np.zeros(n)};rows=[]
    for fold,(tr,va) in enumerate(GroupKFold(min(5,count),shuffle=True,random_state=2718).split(np.arange(n),groups=groups)):
        assert set(molecules[tr]).isdisjoint(molecules[va])
        pp=H.all_predictions([spec],state,emb,tr,va)
        for name in preds:preds[name][va]=pp[name]
        rows.extend(dict(source_row=int(i+2),fold=fold,context_component=int(groups[i])) for i in va)
    report={'available':True,'group_components':count,'raw_contexts':len(set(contexts)),
            'scores':{name:A.loss(y,pred) for name,pred in preds.items()},
            'note':'Diagnostic proxy for acquisition context, not true DOI-disjoint validation and not used for model selection. Metadata is not an input feature.'}
    (out/'context_stress.json').write_text(json.dumps(report,indent=2));pd.DataFrame(rows).to_csv(out/'context_stress_splits.csv',index=False)
    print('CONTEXT_STRESS',json.dumps(report),flush=True)


if __name__=='__main__':main()
