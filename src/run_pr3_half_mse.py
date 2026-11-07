"""Finite nested scaffold comparison against PR3; unweighted MSE, no upper cap."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import audit_regression as A
import research_heads as H
import pr3_gate_candidates as G
import pr3_mlm_candidates as M
from smiles_style import raw_style_features
from run_targeted_research import folds

BASE_SPEC=next(s for s in H.candidate_specs() if s['name']=='notation_mlm_probability_average')

def specifications():
    return [dict(name='pr3',owner='baseline',spec=BASE_SPEC)]+[
        dict(name=s['name'],owner='notation',spec=s) for s in G.candidate_specs()]+[
        dict(name=s['name'],owner='mlm',spec=s) for s in M.candidate_specs()]

def prepare(cache):
    state=joblib.load(cache/'state_cache.joblib')
    state['smiles']=np.array(list(state['train'].SMILES)+list(state['test'].SMILES),object)
    manifest=json.loads((cache/'embedding_manifest.json').read_text())
    assert manifest['input_sha256']==hashlib.sha256('\n'.join(state['smiles']).encode()).hexdigest()
    embeddings={'mlm_raw':np.load(cache/'mlm_raw.npy'),'raw_style':raw_style_features(state['smiles'])}
    return state,embeddings

def fit_head(spec,state,emb,tr):
    if spec['owner']=='baseline':return H.fit_head(spec['spec'],state,emb,tr)
    if spec['owner']=='notation':return G.fit_gate(spec['spec'],state,tr)
    return M.fit_gate(spec['spec'],state,emb,tr)

def probability(spec,model,state,emb,tr,va):
    if spec['owner']=='baseline':return H.predict_head(model,state,emb,va,tr)
    if spec['owner']=='notation':return G.predict_gate(model,state,va)
    return M.predict_gate(model,state,emb,va,tr)

def physical(state,tr,va,blocked,cache):
    content=hashlib.sha256(state['world'].K.tobytes()+state['world'].y[tr].tobytes()+state['world'].ext_si.tobytes()).hexdigest()
    key=hashlib.sha256(json.dumps([tr.tolist(),va.tolist(),list(map(int,blocked)),content,A.V6.CFG]).encode()).hexdigest()
    path=cache/f'physical_{key}.npy'
    if path.exists():return np.load(path)
    r=H.physical(state,tr,va,blocked=np.asarray(blocked,dtype=int))
    np.save(path,r);return r

def evaluate(state,emb,specs,rows,splits,blocked,cache):
    pred=np.full((len(state['y']),len(specs)),np.nan)
    for tr,va in splits:
        assert set(state['groups'][tr]).isdisjoint(state['groups'][va])
        assert set(state['world'].tau[tr]).isdisjoint(state['world'].tau[va])
        r=physical(state,tr,va,blocked,cache)
        for col,s in enumerate(specs):
            model=fit_head(s,state,emb,tr)
            p=probability(s,model,state,emb,tr,va)
            pred[va,col]=H.complete_unbounded_predictions(state,tr,va,r,p)
    assert np.isfinite(pred[rows]).all()
    return pred

def convex_weights(x,y):
    def objective(w):
        error=(x@w-y)/10000.
        return float(np.mean(error**2)),2*x.T@error/(len(y)*10000.)
    k=x.shape[1]
    opt=minimize(objective,np.full(k,1/k),jac=True,method='SLSQP',bounds=[(0,1)]*k,
                 constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(k)},
                 options={'ftol':1e-12,'maxiter':1000})
    if not opt.success:raise RuntimeError(opt.message)
    return opt.x

def finalize_outputs(out,root):
    """Summarize the frozen experiment; preserve the unaccepted PR3 baseline."""
    summary=json.loads((out/'summary.json').read_text())
    confirmations=pd.read_csv(out/'confirmation.csv')
    means=confirmations.groupby('model').MSE.mean().to_dict()
    name=summary['fixed_development_winner']
    summary['confirmation_mean_MSE']=means
    summary['confirmation_improvement_pct']=100*(1-means[name]/means['pr3'])
    accepted=(name!='pr3' and summary['nested_selection_MSE']<summary['baseline_MSE']
              and means[name]<means['pr3'])
    summary['decision']='further_review_required' if accepted else 'retain_PR3; no experimental candidate accepted'
    summary['new_hidden_test_score']=None
    source=root/'results/audit_20261107/submission.csv'
    if not accepted:
        shutil.copy2(source,out/'recommended_submission_unchanged_PR3.csv')
        summary['recommended_submission_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    data=pd.read_csv(out/'development_oof.csv');y=data.observed.to_numpy();segments=[]
    for model in dict.fromkeys(['pr3',name]):
        p=data[model].to_numpy()
        for stratum,mask in [('all',np.ones(len(y),bool)),('SI_le_100',y<=100),
                             ('measured_physical',(y>0)&(y!=10000)),('code_10000',y==10000),('largest',y==y.max())]:
            segments.append(dict(model=model,stratum=stratum,n=int(mask.sum()),
                                 MSE=float(np.mean((p[mask]-y[mask])**2)),mean_prediction=float(p[mask].mean())))
    pd.DataFrame(segments).to_csv(out/'segment_metrics.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',default='work/pr3_half_mse')
    parser.add_argument('--out',default='results/pr3_half_mse');parser.add_argument('--confirm',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    cache=root/args.cache;out=root/args.out;out.mkdir(parents=True,exist_ok=True)
    specs=specifications();names=[s['name'] for s in specs]
    protocol=dict(baseline='PR3 fixed notation_mlm_probability_average',baseline_MSE=11341011.015450314,
                  target_MSE=5670505.507725157,candidates=specs,primary='archived v2 scaffold five-fold',
                  inner='three-fold scaffold GroupKFold shuffle seed314',confirmation_seeds=[4401,4402,4403],
                  selection='minimum unweighted raw MSE on inner scaffold OOF',
                  meta_ensemble='convex inner-OOF weights on baseline, best notation, best MLM; evaluated separately',
                  inference_features=['SMILES'],upper_prediction_cap=None,unchanged_training_rows=320,
                  external_policy='exclude inner and outer validation structures and scaffolds; no exact lookup override',
                  limitation='historical molecules; confirmation splits are stability checks, not untouched data')
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Frozen protocol changed')
    path.write_text(json.dumps(protocol,indent=2))
    state,emb=prepare(cache);ids=np.arange(len(state['y']));y=state['y']
    outer=folds(state,ids,'scaffold',2718,archived=True)
    full=evaluate(state,emb,specs,ids,outer,[],cache)
    saved=pd.read_csv(root/'results/audit_20261107/paired_new_model_errors.csv')
    difference=float(np.max(np.abs(full[:,0]-saved.new_prediction)))
    if difference>1e-5:raise AssertionError(f'PR3 reproduction drift: {difference}')
    mse=np.mean((full-y[:,None])**2,axis=0)
    dev=pd.DataFrame(dict(model=names,MSE=mse,improvement_pct=100*(1-mse/mse[0])))
    dev.sort_values('MSE').to_csv(out/'fixed_candidate_comparison.csv',index=False)
    oof=pd.DataFrame(full,columns=names)
    oof.insert(0,'observed',y);oof.insert(0,'source_row',ids+2)
    oof.to_csv(out/'development_oof.csv',index=False)
    print('DEVELOPMENT',dev.sort_values('MSE').head(10).to_dict('records'),flush=True)
    nested=np.zeros(len(y));ensemble=np.zeros(len(y));choices=[]
    for f,(tr,va) in enumerate(outer):
        inner=[(tr[a],tr[b]) for a,b in GroupKFold(3,shuffle=True,random_state=314).split(tr,groups=state['groups'][tr])]
        pp=evaluate(state,emb,specs,tr,inner,va,cache)
        errors=np.mean((pp[tr]-y[tr,None])**2,axis=0);best=int(np.argmin(errors))
        families=[[0],[j for j,s in enumerate(specs) if s['owner']=='notation'],[j for j,s in enumerate(specs) if s['owner']=='mlm']]
        selected=[min(family,key=lambda j:errors[j]) for family in families]
        w=convex_weights(pp[tr][:,selected],y[tr])
        nested[va]=full[va,best];ensemble[va]=full[va][:,selected]@w
        choices.append(dict(fold=f,selected=names[best],inner_MSE=float(errors[best]),
                            family_models=[names[j] for j in selected],weights=w.tolist()))
        print('NESTED',choices[-1],flush=True)
    winner=int(np.argmin(mse))
    pd.DataFrame(dict(source_row=ids+2,observed=y,pr3=full[:,0],fixed=full[:,winner],nested=nested,ensemble=ensemble)).to_csv(out/'nested_oof.csv',index=False)
    (out/'inner_choices.json').write_text(json.dumps(choices,indent=2))
    summary=dict(PR3_reproduction_max_abs_error=difference,baseline_MSE=float(mse[0]),
                 fixed_development_winner=names[winner],fixed_winner_MSE=float(mse[winner]),
                 nested_selection_MSE=A.loss(y,nested),nested_convex_ensemble_MSE=A.loss(y,ensemble),
                 fifty_percent_target_met_fixed=bool(mse[winner]<=mse[0]/2),
                 fifty_percent_target_met_nested=bool(A.loss(y,nested)<=mse[0]/2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',summary,flush=True)
    (out/'selection_frozen.json').write_text(json.dumps(dict(selected=specs[winner],confirmation_seeds=protocol['confirmation_seeds']),indent=2))
    if args.confirm:
        confirmations=[];predictions=[]
        for seed in protocol['confirmation_seeds']:
            splits=folds(state,ids,'scaffold',seed)
            compare=[specs[0]] if winner==0 else [specs[0],specs[winner]]
            pp=evaluate(state,emb,compare,ids,splits,[],cache)
            for j,s in enumerate(compare):
                metric=A.loss(y,pp[:,j]);confirmations.append(dict(seed=seed,model=s['name'],MSE=metric))
                predictions.extend(dict(seed=seed,model=s['name'],source_row=int(i+2),observed=float(y[i]),prediction=float(pp[i,j])) for i in ids)
            print('CONFIRM',confirmations[-len(compare):],flush=True)
        pd.DataFrame(confirmations).to_csv(out/'confirmation.csv',index=False)
        pd.DataFrame(predictions).to_csv(out/'confirmation_oof.csv',index=False)
        finalize_outputs(out,root)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
