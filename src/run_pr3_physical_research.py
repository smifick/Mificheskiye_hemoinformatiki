"""Nested gate-conditional raw/log physical mixture; no outer-label selection."""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import research_heads as H
import pr3_physical_candidates as P
from smiles_style import raw_style_features


def main():
    state=joblib.load(ROOT/'work/pr3_half_mse/state_cache.joblib')
    y=state['y'];n=len(y)
    smi=np.r_[state['train'].SMILES,state['test'].SMILES]
    emb={'mlm_raw':np.load(ROOT/'work/pr3_half_mse/mlm_raw.npy'),'raw_style':raw_style_features(smi)}
    external=pd.read_csv(ROOT/'results/pr3_half_mse/physical/reviewed_external_bank.csv')
    head=[s for s in H.candidate_specs() if s['name']=='notation_mlm_probability_average'][0]
    specs=P.candidate_specs()
    base=np.load(ROOT/'results/audit_20261107/baseline_oof.npz')
    test=np.arange(n,len(smi))
    pred={s['name']:np.full(n,np.nan) for s in specs}
    pred.update(pr3_reproduced=np.full(n,np.nan),strict_log=np.full(n,np.nan),nested=np.full(n,np.nan))
    selected=[];audit=[]
    for fold in np.unique(base['fold']):
        va=np.flatnonzero(base['fold']==fold);tr=np.flatnonzero(base['fold']!=fold)
        innerbase=np.full(n,np.nan);innerdelta={s['name']:np.full(n,np.nan) for s in specs}
        for it,iv in GroupKFold(3,shuffle=True,random_state=314).split(tr,groups=state['groups'][tr]):
            fit,valid=tr[it],tr[iv]
            block=np.r_[va,test]
            model=H.fit_head(head,state,emb,fit)
            p=H.predict_head(model,state,emb,valid,fit)
            z=float(np.mean(y[fit]==0));q=np.maximum(1-p-z,0)
            log=H.physical(state,fit,valid,blocked=block)
            innerbase[valid]=10000*p+q*log
            for spec in specs:
                raw,aa=P.fit_predict(spec,state,fit,valid,blocked=block,external=external)
                innerdelta[spec['name']][valid]=q*(raw-log)
                audit.append(dict(outer_fold=int(fold),scope='inner',model=spec['name'],**aa))
        scores=[]
        for spec in specs:
            delta=innerdelta[spec['name']][tr]
            residual=y[tr]-innerbase[tr]
            w=float(np.clip(delta@residual/max(delta@delta,1e-12),0,1))
            mse=float(np.mean((innerbase[tr]+w*delta-y[tr])**2))
            scores.append(dict(model=spec['name'],raw_weight=w,MSE=mse))
        best=min(scores,key=lambda a:a['MSE'])
        model=H.fit_head(head,state,emb,tr);p=H.predict_head(model,state,emb,va,tr)
        z=float(np.mean(y[tr]==0));q=np.maximum(1-p-z,0)
        log=H.physical(state,tr,va,blocked=test)
        oldlog=H.physical(state,tr,va)
        pred['pr3_reproduced'][va]=10000*p+q*oldlog
        pred['strict_log'][va]=10000*p+q*log
        for spec in specs:
            raw,aa=P.fit_predict(spec,state,tr,va,blocked=test,external=external)
            audit.append(dict(outer_fold=int(fold),scope='outer',model=spec['name'],**aa))
            w=next(a['raw_weight'] for a in scores if a['model']==spec['name'])
            pred[spec['name']][va]=10000*p+q*((1-w)*log+w*raw)
        pred['nested'][va]=pred[best['model']][va]
        selected.append(dict(fold=int(fold),selected=best,inner_scores=scores))
        print(json.dumps(selected[-1]),flush=True)
    out=ROOT/'results/pr3_half_mse/physical'
    rows=[]
    for name,p in pred.items():
        assert np.isfinite(p).all()
        rows.append(dict(model=name,MSE=float(np.mean((p-y)**2)),extreme_prediction=float(p[y.argmax()]),
                         low_SI_MSE=float(np.mean((p[y<=100]-y[y<=100])**2))))
    pd.DataFrame(rows).to_csv(out/'nested_metrics.csv',index=False)
    pd.DataFrame(dict(observed=y,fold=base['fold'],**pred)).to_csv(out/'nested_oof.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'external_count_audit.csv',index=False)
    (out/'nested_selections.json').write_text(json.dumps(selected,indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
