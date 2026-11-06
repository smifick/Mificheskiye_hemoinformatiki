"""Bounded technical-code probability models on frozen SMILES features."""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler,normalize
from smiles_style import raw_style_features
import pandas as pd
from pathlib import Path

import audit_regression as A


def candidate_specs():
    specs=[]
    for encoder in ['mtr','mlm']:
        for view in ['canonical','raw','augmented']:
            key=encoder+'_'+view
            for c in [.01,.1]:
                specs.append(dict(name=f'{key}_logistic_{c}',family='logistic',embedding=key,C=c))
            specs.append(dict(name=f'{key}_rbf_fusion',family='fusion',embedding=key,alpha=.5))
    specs += [dict(name='both_canonical_logistic',family='logistic',embedding='both_canonical',C=.1),
              dict(name='both_canonical_rbf_fusion',family='fusion',embedding='both_canonical',alpha=.5),
              dict(name='calibrated_v7_probabilities',family='calibrated',C=1.)]
    style=dict(name='raw_smiles_notation_logistic',family='logistic',embedding='raw_style',C=1.)
    raw_mlm=dict(name='mlm_raw_logistic_0.1',family='logistic',embedding='mlm_raw',C=.1)
    specs += [style,dict(name='notation_mlm_probability_average',family='paired',components=[style,raw_mlm])]
    return specs


def load_embeddings(out):
    result={}
    for encoder in ['mtr','mlm']:
        data=np.load(out/f'embeddings_{encoder}.npz')
        for view in ['raw','canonical','augmented']:
            result[f'{encoder}_{view}']=np.column_stack([data[f'{view}_mean'],data[f'{view}_cls']]).astype(float)
    result['both_canonical']=np.column_stack([result['mtr_canonical'],result['mlm_canonical']])
    root=Path(__file__).resolve().parents[1]
    source=list(pd.read_csv(root/'data/raw/train.csv',usecols=['SMILES']).SMILES)+list(pd.read_csv(root/'data/raw/test.csv',usecols=['SMILES']).SMILES)
    result['raw_style']=raw_style_features(source)
    return result


def old_probabilities(state,tr,va):
    p0=A.V6.code_prob(state['world'],state['flag'],tr,va,.5)
    p1=A.pcode_feature(state,tr,va)
    return np.column_stack([p0,p1])


def calibration_features(p):
    p=np.clip(p,1e-4,1-1e-4)
    return np.log(p/(1-p))


def fit_head(spec,state,embeddings,tr):
    flag=state['flag'];c=(state['y']==10000).astype(int)
    fit=tr if spec.get('training_scope')=='all' else tr[~flag[tr]]
    ff=tr[flag[tr]];pf=float((c[ff].sum()+.5)/(len(ff)+2.))
    model=dict(spec=spec,fit=fit,p_flag=pf)
    if np.unique(c[fit]).size<2:
        model['constant']=float(c[fit].mean())
        return model
    if spec['family']=='paired':
        model['components']=[fit_head(child,state,embeddings,tr) for child in spec['components']]
        return model
    if spec['family']=='calibrated':
        oof=np.zeros((len(state['y']),2))
        groups=state['groups'][tr]
        for it,iv in GroupKFold(3,shuffle=True,random_state=314).split(tr,groups=groups):
            oof[tr[iv]]=old_probabilities(state,tr[it],tr[iv])
        est=LogisticRegression(C=spec['C'],max_iter=2000,random_state=42)
        est.fit(calibration_features(oof[fit]),c[fit])
        model['estimator']=est
        return model
    z=embeddings[spec['embedding']]
    if spec.get('include_source_flag'):z=np.column_stack([z,flag.astype(float)])
    scaler=StandardScaler().fit(z[fit]);zt=scaler.transform(z[fit])
    model['scaler']=scaler
    if spec['family']=='logistic':
        est=LogisticRegression(C=spec['C'],max_iter=2000,random_state=42)
        est.fit(zt,c[fit]);model['estimator']=est
    else:
        zt=normalize(zt)
        distances=cdist(zt,zt,metric='sqeuclidean')
        nonzero=distances[distances>1e-12]
        bandwidth=float(np.median(nonzero)) if len(nonzero) else 1.
        ke=np.exp(-distances/max(bandwidth,1e-12))
        kb=.5*(state['world'].K[np.ix_(fit,fit)]+state['K_feature'][np.ix_(fit,fit)])
        k=.5*(ke+kb)
        mean=float(c[fit].mean())
        dual=solve(k+spec['alpha']*np.eye(len(fit)),c[fit]-mean,assume_a='pos')
        model.update(embedding_reference=zt,bandwidth=bandwidth,mean=mean,dual=dual)
    return model


def predict_head(model,state,embeddings,rows,tr):
    spec=model['spec'];fit=model['fit']
    if 'constant' in model:p=np.full(len(rows),model['constant'])
    elif spec['family']=='paired':p=np.mean([predict_head(child,state,embeddings,rows,tr) for child in model['components']],axis=0)
    elif spec['family']=='calibrated':
        p=model['estimator'].predict_proba(calibration_features(old_probabilities(state,tr,rows)))[:,1]
    else:
        z=embeddings[spec['embedding']][rows]
        if spec.get('include_source_flag'):z=np.column_stack([z,state['flag'][rows].astype(float)])
        z=model['scaler'].transform(z)
        if spec['family']=='logistic':p=model['estimator'].predict_proba(z)[:,1]
        else:
            z=normalize(z)
            ke=np.exp(-cdist(z,model['embedding_reference'],metric='sqeuclidean')/model['bandwidth'])
            kb=.5*(state['world'].K[np.ix_(rows,fit)]+state['K_feature'][np.ix_(rows,fit)])
            p=np.clip(model['mean']+.5*(ke+kb)@model['dual'],0,1)
    return p if spec.get('training_scope')=='all' else np.where(state['flag'][rows],model['p_flag'],p)


def physical(state,tr,va,blocked=None):
    w=state['world'];cfg=A.V6.CFG
    mask=A.T.ext_allowed(w,va)
    if blocked is not None and len(blocked):mask &= A.T.ext_allowed(w,blocked)
    ew=np.array([cfg['w'].get(t,0.) for t in w.ext_tier])
    return A.T.fit_B(w,tr,mask,ew,va,cfg)


def complete_predictions(state,tr,va,r,p):
    zero=float(np.mean(state['y'][tr]==0))
    return np.clip(10000*p+np.clip(1-p-zero,0,1)*r,0,10000)


def complete_unbounded_predictions(state,tr,va,r,p):
    zero=float(np.mean(state['y'][tr]==0))
    return np.maximum(10000*p+np.clip(1-p-zero,0,1)*r,0)


def old_v7(state,tr,va,r):
    p=old_probabilities(state,tr,va)
    return .5*(complete_predictions(state,tr,va,r,p[:,0])+complete_predictions(state,tr,va,r,p[:,1]))


def all_predictions(specs,state,embeddings,tr,va,blocked=None):
    r=physical(state,tr,va,blocked)
    baseline=old_v7(state,tr,va,r)
    old=old_probabilities(state,tr,va)
    unbounded=.5*(complete_unbounded_predictions(state,tr,va,r,old[:,0])+complete_unbounded_predictions(state,tr,va,r,old[:,1]))
    pred={'v7':baseline,'v7_without_upper_cap':unbounded}
    for spec in specs:
        head=fit_head(spec,state,embeddings,tr)
        p=predict_head(head,state,embeddings,va,tr)
        alt=complete_predictions(state,tr,va,r,p)
        pred[spec['name']]=alt
        pred[spec['name']+'_blend50']=.5*(baseline+alt)
    return pred
