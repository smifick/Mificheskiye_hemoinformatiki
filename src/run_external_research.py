"""Bounded source ablations, nested source selection, and portable final fit."""
from pathlib import Path
import argparse,hashlib,json,zipfile
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import external_si_model as M
import run_targeted_research as R

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/external_20261107'

def configurations():
    base={'legacy_V':1.,'legacy_C':1.}
    def cfg(name,w,rho=0.):return dict(name=name,source_weights=w,rho=rho,alpha=3.)
    result=[cfg('clean_legacy',base),cfg('clean_reviewed_V',{'legacy_V':1.}),cfg('organizer_only',{})]
    for weight in [.25,1.]:result.append(cfg('clean_legacy_PMC_'+str(weight),dict(base,PMC10995631=weight)))
    result.append(cfg('reviewed_V_PMC',{'legacy_V':1.,'PMC10995631':1.}))
    for source in ['aux_chembl','aux_pubchem']:
        for rho in [.25,.5]:result.append(cfg('clean_'+source+'_'+str(rho),dict(base,**{source:.25}),rho))
    result.append(cfg('clean_PMC_aux_both',dict(base,PMC10995631=1.,aux_chembl=.25,aux_pubchem=.25),.25))
    return result

def folds(ctx,rows,scheme,seed,archived=False,nfolds=5):
    return R.folds(ctx['state'],rows,scheme,seed,n=nfolds,archived=archived)

def evaluate(ctx,configs,rows,scheme,seed,archived=False,blocked=None,nfolds=5):
    preds={c['name']:np.full(ctx['n'],np.nan) for c in configs};details=[];splits=[]
    for fold,(tr,va) in enumerate(folds(ctx,rows,scheme,seed,archived,nfolds)):
        for cfg in configs:
            p,counts=M.predict_fold(ctx,tr,va,cfg,blocked)
            preds[cfg['name']][va]=p
            details.append(dict(scheme=scheme,seed=seed,fold=fold,model=cfg['name'],n=len(va),
                MSE=float(np.mean((ctx['y'][va]-p)**2)),**counts))
        splits.extend(dict(scheme=scheme,seed=seed,fold=fold,source_row=int(i+2)) for i in va)
    scores={k:float(np.mean((p[rows]-ctx['y'][rows])**2)) for k,p in preds.items()}
    return scores,preds,details,splits

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['development','complete'],default='complete')
    a=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);configs=configurations()
    protocol={'frozen_successful_commit':'b6c2e0c384d043582753993e442be7c042c3139a',
       'successful_public_MSE':6327610.34997,'new_policy':'Exclude all external test-structure labels and disable exact lookup',
       'unchanged':'Notation/MLM code head, source presence, kernel representation, alpha3, log1p, 5-fold training-only smearing, output cap10000',
       'candidates':configs,'primary':'actual archived 5-fold scaffold','secondary':'5-fold parent seed42',
       'nested':'5 outer folds; choose source weights/model using only 3 scaffold inner folds seed827, with all outer held-out scaffolds blocked externally',
       'confirmation_seeds':[901,902,903],
       'acceptance':'Nested source-selection beats clean_legacy on both schemes; fixed development winner beats clean_legacy on both confirmation schemes by at least0.5%. Otherwise clean_legacy.',
       'no_independent_validation_claim':True,'numeric_auxiliary_tasks_separate':True}
    (OUT/'model_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    ctx=M.load_context(ROOT,ROOT/'data/external_research/training_bank.csv');ids=np.arange(ctx['n'])
    print('CONTEXT',len(ctx['bank']),ctx['K'].shape,flush=True)
    development=[];all_oof=[];all_splits=[];all_folds=[];devpred={}
    for scheme,seed,archived in [('scaffold',2718,True),('parent',42,False)]:
        scores,pred,details,split=evaluate(ctx,configs,ids,scheme,seed,archived)
        devpred[scheme]=pred;all_folds+=details;all_splits+=split
        development.extend(dict(scheme=scheme,model=k,MSE=v,relative_to_clean=v/scores['clean_legacy']) for k,v in scores.items())
        for name,pp in pred.items():all_oof.extend(dict(scheme=scheme,model=name,source_row=int(i+2),observed=float(ctx['y'][i]),prediction=float(pp[i])) for i in ids)
        print('DEVELOPMENT',scheme,scores,flush=True)
    dev=pd.DataFrame(development);dev.to_csv(OUT/'source_development.csv',index=False)
    pd.DataFrame(all_oof).to_csv(OUT/'source_development_oof.csv',index=False)
    pd.DataFrame(all_folds).to_csv(OUT/'source_development_folds.csv',index=False)
    rank=dev.groupby('model').relative_to_clean.mean().sort_values();chosen=rank.index[0]
    (OUT/'source_selection_frozen.json').write_text(json.dumps({'selected':chosen,'ranking':rank.to_dict()},indent=2)+'\n')
    # A 2x2 method/data comparison with the previous v7 head.
    factorial=[]
    for scheme,seed,archived in [('scaffold',2718,True),('parent',42,False)]:
        for name in ['clean_legacy','clean_legacy_PMC_1.0']:
            cfg=next(c for c in configs if c['name']==name);old=np.zeros(ctx['n'])
            for tr,va in folds(ctx,ids,scheme,seed,archived):old[va]=M.predict_fold(ctx,tr,va,cfg,old_method=True)[0]
            factorial.extend([dict(scheme=scheme,method='v7',data=name,MSE=float(np.mean((old-ctx['y'])**2))),
                dict(scheme=scheme,method='successful_notation_MLM',data=name,MSE=float(np.mean((devpred[scheme][name]-ctx['y'])**2)))])
    pd.DataFrame(factorial).to_csv(OUT/'method_data_factorial.csv',index=False)
    if a.stage=='development':return
    nested=[];nested_rows=[]
    for scheme,seed,archived in [('scaffold',2718,True),('parent',42,False)]:
        for fold,(tr,va) in enumerate(folds(ctx,ids,scheme,seed,archived)):
            scores,_,_,_=evaluate(ctx,configs,tr,'scaffold',827,blocked=va,nfolds=3)
            selected=min(scores,key=scores.get);cfg=next(c for c in configs if c['name']==selected)
            pp,_=M.predict_fold(ctx,tr,va,cfg)
            baseline=devpred[scheme]['clean_legacy'][va]
            nested.append(dict(scheme=scheme,fold=fold,selected=selected,n=len(va),
                selected_MSE=float(np.mean((pp-ctx['y'][va])**2)),baseline_MSE=float(np.mean((baseline-ctx['y'][va])**2))))
            nested_rows.extend(dict(scheme=scheme,fold=fold,source_row=int(i+2),observed=float(ctx['y'][i]),
                selection_prediction=float(v),clean_prediction=float(b)) for i,v,b in zip(va,pp,baseline))
            print('NESTED',nested[-1],flush=True)
    pd.DataFrame(nested).to_csv(OUT/'source_nested_folds.csv',index=False)
    nested_oof=pd.DataFrame(nested_rows);nested_oof.to_csv(OUT/'source_nested_oof.csv',index=False)
    confirmation=[];confirm_rows=[]
    short=[c for c in configs if c['name'] in set([chosen,'clean_legacy'])]
    for scheme in ['scaffold','parent']:
        for seed in [901,902,903]:
            scores,pred,details,split=evaluate(ctx,short,ids,scheme,seed)
            all_splits+=split
            confirmation.extend(dict(scheme=scheme,seed=seed,model=k,MSE=v) for k,v in scores.items())
            for name,pp in pred.items():confirm_rows.extend(dict(scheme=scheme,seed=seed,model=name,source_row=int(i+2),observed=float(ctx['y'][i]),prediction=float(pp[i])) for i in ids)
            print('CONFIRM',scheme,seed,scores,flush=True)
    conf=pd.DataFrame(confirmation);conf.to_csv(OUT/'source_confirmation.csv',index=False)
    pd.DataFrame(confirm_rows).to_csv(OUT/'source_confirmation_oof.csv',index=False)
    pd.DataFrame(all_splits).to_csv(OUT/'source_splits.csv',index=False)
    acceptance=chosen!='clean_legacy';summary=[]
    for scheme,part in nested_oof.groupby('scheme'):
        clean=float(np.mean((part.clean_prediction-part.observed)**2));selection=float(np.mean((part.selection_prediction-part.observed)**2))
        acceptance &= selection<clean
        summary.append(dict(scheme=scheme,clean_MSE=clean,nested_selection_MSE=selection))
    for scheme,part in conf.groupby('scheme'):
        means=part.groupby('model').MSE.mean();acceptance &= means[chosen]<.995*means['clean_legacy']
    final_name=chosen if acceptance else 'clean_legacy';cfg=next(c for c in configs if c['name']==final_name)
    bundle=M.final_bundle(ctx,cfg);joblib.dump(bundle,OUT/'model.joblib',compress=3)
    # Use the already frozen test embeddings to export; real-CLI encoding is independently checked.
    old=bundle['baseline_portable_model'];a=M.fingerprint(ctx['state']['test'].SMILES)
    r=np.maximum(old['B_smear']*np.exp(old['B_mean']+M.kernel(a,old['B_reference'])@old['B_dual'])-1,0)
    head=bundle['head'];te=np.arange(ctx['n'],ctx['n']+len(ctx['state']['test']))
    p=__import__('research_heads').predict_head(head,ctx['state'],ctx['embeddings'],te,np.arange(ctx['n']))
    pred=np.clip(10000*p+np.clip(1-p-old['zero_rate'],0,1)*r,0,10000)
    sample=pd.read_csv(ROOT/'data/raw/sample_submit.csv',usecols=['index']);ids=ctx['state']['test']['index']
    assert ids.tolist()==sample['index'].tolist() and np.isfinite(pred).all()
    pd.DataFrame({'index':ids,'SI VACV':pred}).to_csv(OUT/'submission.csv',index=False)
    with zipfile.ZipFile(OUT/'submission.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(OUT/'submission.csv','submission.csv')
    result={'development_winner':chosen,'selected_for_submission':final_name,'acceptance_passed':bool(acceptance),
      'nested':summary,'confirmation':conf.groupby(['scheme','model']).MSE.mean().reset_index().to_dict('records'),
      'fit_counts':bundle['physical_fit_counts'],'new_public_MSE':None,'known_successful_public_MSE':6327610.34997,
      'test_lookup_count':len(old['lookup']),'model_sha256':hashlib.sha256((OUT/'model.joblib').read_bytes()).hexdigest(),
      'submission_sha256':hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest(),
      'warning':'Previously used organizer labels; no new independent competition validation. Historical successful model retained unchanged.'}
    (OUT/'model_results.json').write_text(json.dumps(result,indent=2)+'\n');print('FINAL',result,flush=True)

if __name__=='__main__':main()
