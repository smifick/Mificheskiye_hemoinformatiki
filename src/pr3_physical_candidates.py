"""Finite uncapped physical candidates, with all fitting restricted to tr.

Diagnostic oracles read validation labels ONLY after predictions are frozen.
They never enter candidate fitting or inner model selection.
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
import research_heads as H
import train_v5 as T


def candidate_specs():
    return [dict(name=f'raw_physical_alpha{a:g}_{bank}', alpha=a, bank=bank)
            for a in [3.,30.] for bank in ['none','reviewed_new']]


def fit_predict(spec, state, tr, va, blocked=None, external=None):
    """Return raw KRR r: target never clipped; test/query scaffold excluded.

    external: reviewed bank DataFrame with SMILES, SI; may be None only for
    organizer-only candidates. Validation labels are never read.
    """
    tr,va=np.asarray(tr,int),np.asarray(va,int)
    y=state['y'][tr];keep=(y>0)&(y!=10000)
    ids=tr[keep];target=y[keep];w=np.ones(len(ids))
    world=state['world'];kfit=world.K[np.ix_(ids,ids)];kquery=world.K[np.ix_(va,ids)]
    audit={'physical_rows':len(ids),'external_rows':0}
    if spec['bank']!='none':
        if external is None:raise ValueError('Explicit reviewed exact external bank required')
        smi=np.r_[state['train'].SMILES,state['test'].SMILES]
        extm=[T.std(s) for s in external.SMILES]
        et=np.array([T.taut_key(m) for m in extm],object)
        es=np.array([T.gscaf(m) for m in extm],object)
        block=np.r_[va,[] if blocked is None else blocked,np.arange(len(state['y']),len(smi))].astype(int)
        safe=~np.isin(et,world.tau[block])
        safe&=~np.isin(es,world.scf[block][world.scf[block]!=''])
        safe&=np.isfinite(external.SI)&(external.SI>0)
        ext=external[np.asarray(safe)];em=[m for m,k in zip(extm,safe) if k]
        # Same deterministic molecular kernel as PR3, without fitting on query.
        trainm=[T.std(smi[i]) for i in ids];querym=[T.std(smi[i]) for i in va]
        def arrays(mols):
            return np.array([T.mg.GetFingerprintAsNumPy(m) for m in mols],float),np.array([T.tt.GetCountFingerprintAsNumPy(m) for m in mols],float)
        def kern(a,b):return .5*(T.tanimoto(a[0],b[0])+T.minmax(a[1],b[1]))
        if len(em):
            a,b,c=arrays(trainm),arrays(em),arrays(querym)
            cross=kern(a,b)
            kfit=np.block([[kfit,cross],[cross.T,kern(b,b)]])
            kquery=np.column_stack([kquery,kern(c,b)])
            target=np.r_[target,ext.SI.to_numpy(float)];w=np.r_[w,np.full(len(ext),.1)]
        audit['external_rows']=len(ext)
    mean=float(np.average(target,weights=w))
    dual=solve(kfit+spec['alpha']*np.diag(1/w),target-mean,assume_a='pos')
    return np.maximum(mean+kquery@dual,0),audit


def oracle_audit(root,out):
    path=root/'results/audit_20261107';z=np.load(path/'baseline_oof.npz')
    rows=pd.read_csv(path/'paired_new_model_errors.csv').sort_values('source_row').reset_index(drop=True)
    y,r,z0=z['y'],z['r'],z['p_zero'];pred=rows.new_prediction.to_numpy(float)
    p=np.clip(np.where(pred>=10000*(1-z0),pred/10000,(pred-(1-z0)*r)/(10000-r)),0,1)
    q=np.maximum(1-p-z0,0)
    np.testing.assert_allclose(10000*p+q*r,pred,atol=1e-7,rtol=1e-10)
    code,physical=y==10000,(y>0)&(y!=10000)
    bundle=joblib.load(path/'selected_model.joblib')
    source=bundle['baseline_portable_model']['source_presence']
    flag=np.array([T.taut_key(T.std(s)) in source for s in rows.SMILES])
    meanerr=lambda a:float(np.mean((a-y)**2))
    po=10000*p+q*np.where(physical,y,r)
    floor=pred.copy();floor[physical]=np.maximum(y[physical],10000*p[physical])
    co=10000*code+np.maximum(1-code-z0,0)*r
    largest=int(y.argmax());tail=pred.copy();tail[largest]=y[largest]
    result={'diagnostic_only':True,'used_for_fitting_or_selection':False,
            'selected_MSE':meanerr(pred),'target_MSE':.5*meanerr(pred),
            'true_code_oracle_existing_r_MSE':meanerr(co),
            'true_measured_SI_oracle_unbounded_MSE':meanerr(po),
            'true_measured_SI_oracle_inherited_cap_MSE':meanerr(np.clip(po,0,10000)),
            'best_any_nonnegative_r_measured_only_fixed_p_MSE':meanerr(floor),
            'largest_row_only_perfect_MSE':meanerr(tail),
            'largest_row_source_row':int(rows.source_row.iloc[largest]),
            'largest_row_observed':float(y[largest]),'largest_row_prediction':float(pred[largest]),
            'largest_row_physical_prediction':float(r[largest]),
            'largest_row_MSE_contribution':float((pred[largest]-y[largest])**2/len(y)),
            'source_flag_rows':int(flag.sum()),'source_flag_code_rows':int((flag&code).sum()),
            'interpretation':'Physical oracles modify measured rows only. Undefined physical responses of technical-code/zero rows remain at actual predictions. Per-label oracles are descriptive bounds and cannot be deployed.'}
    budget=[]
    for name,mask in [('measured',physical),('technical_code',code),('zero',y==0),('source_present',flag),('source_absent',~flag),('source_present_excluding_largest',flag&(np.arange(len(y))!=largest))]:
        budget.append(dict(stratum=name,n=int(mask.sum()),MSE=float(np.mean((pred[mask]-y[mask])**2)),contribution_to_total_MSE=float(np.sum((pred[mask]-y[mask])**2)/len(y))))
    out.mkdir(parents=True,exist_ok=True)
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(budget).to_csv(out/'error_budget.csv',index=False)
    rows.assign(code_probability=p,physical_prediction=r,zero_probability=z0,source_present=flag).to_csv(out/'rows.csv',index=False)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--out',type=Path,default=Path('work/pr3_physical_research'))
    args=parser.parse_args()
    with threadpool_limits(limits=2):print(json.dumps(oracle_audit(args.root,args.out),indent=2))


if __name__=='__main__':main()
