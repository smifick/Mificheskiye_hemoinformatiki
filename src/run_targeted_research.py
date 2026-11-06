"""Finite, diagnostic-led study of frozen molecular embeddings/code calibration.

Primary folds are the actual archived v2/v3 assignments. Inner model/ensemble
selection is cross-fitted. Every score is on the original encoded response.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import GroupKFold,StratifiedGroupKFold

import audit_regression as A
import research_heads as H


def folds(state,rows,scheme,seed,n=5,archived=False):
    if archived:
        p=pd.read_csv(Path(__file__).resolve().parents[1]/'results/v2/validation_splits.csv').sort_values('source_row')
        labels=p.outer_fold.to_numpy(int)
        return [(rows[labels!=f],rows[labels==f]) for f in sorted(set(labels))]
    if scheme=='scaffold':
        return [(rows[a],rows[b]) for a,b in GroupKFold(n,shuffle=True,random_state=seed).split(rows,groups=state['groups'][rows])]
    parents=np.array([Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=False) for s in state['world'].tau[rows]])
    return [(rows[a],rows[b]) for a,b in StratifiedGroupKFold(n,shuffle=True,random_state=seed).split(rows,state['y'][rows]==10000,parents)]


def evaluate(state,embeddings,specs,rows,scheme,seed,archived=False,blocked=None,nfolds=5):
    pred={};splits=[]
    for f,(tr,va) in enumerate(folds(state,rows,scheme,seed,nfolds,archived)):
        assert set(state['world'].tau[tr]).isdisjoint(state['world'].tau[va])
        if scheme=='scaffold':assert set(state['groups'][tr]).isdisjoint(state['groups'][va])
        pp=H.all_predictions(specs,state,embeddings,tr,va,blocked)
        for name,p in pp.items():
            if name not in pred:pred[name]=np.zeros(len(state['y']))
            pred[name][va]=p
        splits.extend(dict(scheme=scheme,seed=seed,fold=f,source_row=int(i+2)) for i in va)
    scores={name:A.loss(state['y'][rows],p[rows]) for name,p in pred.items()}
    return scores,pred,splits


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/audit_20261107');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/a.out
    state=joblib.load(out/'state_cache.joblib');emb=H.load_embeddings(out)
    expected=hashlib.sha256(('\n'.join(list(state['train'].SMILES)+list(state['test'].SMILES))).encode()).hexdigest()
    for model in ['mtr','mlm']:
        metadata=json.loads((out/f'embeddings_{model}.json').read_text())
        if metadata['input_smiles_sha256']!=expected:raise ValueError('Embedding rows do not match the current data')
    specs=H.candidate_specs();ids=np.arange(len(state['y']))
    protocol={'primary':'published scaffold folds from results/v2/validation_splits.csv',
              'secondary':'parent grouped 5-fold seed42',
              'candidates':specs,'ensemble_weights':[.5,1.],
              'selection':'Minimum equal-weight relative MSE versus v7 across scaffold and parent inner CV.',
              'nested_outer_seed':2718,'nested_inner_seed':314,'nested_inner_folds':3,
              'confirmation_seeds':[901,902,903],
              'acceptance':'Fixed candidate improves pooled development and both confirmation schemes; nested selection improves both schemes. Otherwise retain deployed v7.',
              'feature_inputs':['SMILES'],'target':'SI VACV','data':'320 unchanged organizer labels; legacy frozen external physical bank only inside fold-safe B',
              'no_independent_holdout_claim':True}
    (out/'targeted_protocol.json').write_text(json.dumps(protocol,indent=2))
    development=[];dev_predictions={};all_splits=[]
    for scheme,seed,archived in [('scaffold',2718,True),('parent',42,False)]:
        scores,pp,ss=evaluate(state,emb,specs,ids,scheme,seed,archived)
        all_splits+=ss;dev_predictions[scheme]=pp
        for name,metric in scores.items():development.append(dict(scheme=scheme,model=name,MSE=metric,relative_to_v7=metric/scores['v7']))
        print('DEVELOPMENT',scheme,sorted(scores.items(),key=lambda x:x[1])[:8],flush=True)
    dev=pd.DataFrame(development);dev.to_csv(out/'targeted_development.csv',index=False)
    ranking=dev.groupby('model').relative_to_v7.mean().sort_values()
    ranking.to_csv(out/'targeted_ranking.csv')
    chosen=ranking.index[0]
    (out/'targeted_selection_frozen.json').write_text(json.dumps({'selected':chosen,'ranking':ranking.to_dict()},indent=2))
    # Evaluate the complete selection procedure. Both inner schemes are computed
    # from outer TRAINING rows; no outer validation response enters selection.
    nested_rows=[];nested_oof=[]
    for scheme,outerseed,archived in [('scaffold',2718,True),('parent',42,False)]:
        for fold,(tr,va) in enumerate(folds(state,ids,scheme,outerseed,archived=archived)):
            inner_scores=[]
            for inner_scheme in ['scaffold','parent']:
                scores,_,_=evaluate(state,emb,specs,tr,inner_scheme,314,blocked=va,nfolds=3)
                inner_scores.append({name:value/scores['v7'] for name,value in scores.items()})
            names=list(inner_scores[0])
            means={name:float(np.mean([s[name] for s in inner_scores])) for name in names}
            selected=min(names,key=means.get)
            pp=H.all_predictions(specs,state,emb,tr,va)
            nested_rows.append(dict(scheme=scheme,fold=fold,selected=selected,n=len(va),
                                   selected_MSE=A.loss(state['y'][va],pp[selected]),baseline_MSE=A.loss(state['y'][va],pp['v7'])))
            nested_oof.extend(dict(scheme=scheme,fold=fold,source_row=int(i+2),observed=float(state['y'][i]),
                                  baseline=float(pp['v7'][j]),prediction=float(pp[selected][j]),
                                  fixed_candidate=float(pp[chosen][j])) for j,i in enumerate(va))
            print('NESTED',nested_rows[-1],flush=True)
    pd.DataFrame(nested_rows).to_csv(out/'targeted_nested_folds.csv',index=False)
    noof=pd.DataFrame(nested_oof);noof.to_csv(out/'targeted_nested_oof.csv',index=False)
    nested_summary=[]
    for scheme,d in noof.groupby('scheme'):
        nested_summary.append(dict(scheme=scheme,baseline_MSE=A.loss(d.observed,d.baseline),
            selection_pipeline_MSE=A.loss(d.observed,d.prediction),fixed_candidate_MSE=A.loss(d.observed,d.fixed_candidate)))
    # The winner is frozen before additional repeats. They reuse development
    # molecules and are stability checks, not an independent dataset.
    confirm=[];confirm_oof=[]
    for scheme in ['scaffold','parent']:
        for seed in protocol['confirmation_seeds']:
            if chosen=='v7':chosen_specs=[]
            elif chosen=='v7_without_upper_cap':chosen_specs=[]
            else:
                base=chosen.removesuffix('_blend50')
                chosen_specs=[s for s in specs if s['name']==base]
            scores,pp,ss=evaluate(state,emb,chosen_specs,ids,scheme,seed)
            all_splits+=ss
            for name in dict.fromkeys(['v7',chosen]):
                confirm.append(dict(scheme=scheme,seed=seed,model=name,MSE=scores[name]))
                confirm_oof.extend(dict(scheme=scheme,seed=seed,source_row=i+2,model=name,observed=float(state['y'][i]),prediction=float(pp[name][i])) for i in ids)
            print('CONFIRM',scheme,seed,{name:scores[name] for name in dict.fromkeys(['v7',chosen])},flush=True)
    conf=pd.DataFrame(confirm);conf.to_csv(out/'targeted_confirmation.csv',index=False)
    coof=pd.DataFrame(confirm_oof);coof.to_csv(out/'targeted_confirmation_oof.csv',index=False)
    pd.DataFrame(all_splits).to_csv(out/'targeted_splits.csv',index=False)
    acceptable=chosen!='v7'
    for summary in nested_summary:
        acceptable &= summary['selection_pipeline_MSE']<summary['baseline_MSE']
    for scheme,d in conf.groupby('scheme'):
        means=d.groupby('model').MSE.mean()
        acceptable &= means[chosen]<means['v7']
    final_name=chosen if acceptable else 'v7'
    # Final fit uses the full fixed external bank, as did deployed v7. No test
    # labels/concentrations are consulted. Exact external lookup stays unchanged.
    w=state['world'];te=np.arange(w.n_tr,w.n_tr+w.n_te);tr=ids
    cfg=A.V6.CFG;ew=np.array([cfg['w'].get(t,0.) for t in w.ext_tier])
    r=A.T.fit_B(w,tr,np.ones(w.n_ext,bool),ew,te,cfg)
    base=H.old_v7(state,tr,te,r)
    if final_name=='v7':pred=base;head=None
    elif final_name=='v7_without_upper_cap':
        prob=H.old_probabilities(state,tr,te)
        pred=.5*(H.complete_unbounded_predictions(state,tr,te,r,prob[:,0])+H.complete_unbounded_predictions(state,tr,te,r,prob[:,1]));head=None
    else:
        spec=next(s for s in specs if s['name']==final_name.removesuffix('_blend50'))
        head=H.fit_head(spec,state,emb,tr)
        pcode=H.predict_head(head,state,emb,te,tr)
        pred=H.complete_predictions(state,tr,te,r,pcode)
        if final_name.endswith('_blend50'):pred=.5*(base+pred)
    lt=A.T.lookup_table(w)
    values=np.array([lt.get(w.tau[i],np.nan) for i in te]);covered=np.isfinite(values)
    lookup=np.maximum(np.nan_to_num(values),0) if final_name=='v7_without_upper_cap' else np.clip(np.nan_to_num(values),0,10000)
    pred=np.where(covered,lookup,pred)
    baseline_final=np.where(covered,np.clip(np.nan_to_num(values),0,10000),base)
    reference=pd.read_csv(root/'submission_candidate_v7.csv')['SI VACV'].to_numpy()
    assert np.max(abs(baseline_final-reference))<1e-6,'Deployed baseline does not reproduce'
    submission=pd.DataFrame({'index':state['test']['index'],'SI VACV':pred})
    sample=pd.read_csv(root/'data/raw/sample_submit.csv',usecols=['index'])
    assert submission['index'].tolist()==sample['index'].tolist() and np.isfinite(pred).all() and (pred>=0).all()
    submission.to_csv(out/'submission.csv',index=False)
    with zipfile.ZipFile(out/'submission.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(out/'submission.csv','submission.csv')
    joblib.dump({'selected':final_name,'head':head,'baseline_portable_model':joblib.load(root/'results/v7_final/model.joblib'),
                 'encoding_batch_size':1,'encoding_threads':3},out/'selected_model.joblib',compress=3)
    result={'frozen_development_winner':chosen,'selected_for_submission':final_name,'acceptance_passed':bool(acceptable),
            'nested':nested_summary,'confirmation':conf.groupby(['scheme','model']).MSE.mean().reset_index().to_dict('records'),
            'baseline_final_max_abs_reproduction_error':float(np.max(abs(baseline_final-reference))),
            'submission_rows':len(submission),'new_candidate_public_MSE':None,
            'known_v7_public_MSE':9421022.05828,'warning':'All local labels were previously used in project development; no new independent validation dataset exists.'}
    (out/'targeted_results.json').write_text(json.dumps(result,indent=2))
    print('FINAL',json.dumps(result),flush=True)


if __name__=='__main__':main()
