"""Fold-isolated numerical multi-output kernel regression."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
import train_v5 as T
import train_v7 as V7
import research_heads as H

def fingerprint(smiles):
    mols=[T.std(str(s)) for s in smiles]
    if any(m is None for m in mols):raise ValueError('Invalid structure')
    return {'bits':np.asarray([T.mg.GetFingerprintAsNumPy(m) for m in mols],float),
            'torsion':np.asarray([T.tt.GetCountFingerprintAsNumPy(m) for m in mols],float)}

def kernel(a,b):return .5*(T.tanimoto(a['bits'],b['bits'])+T.minmax(a['torsion'],b['torsion']))
def select_matrix(m,rows):return {k:v[rows] for k,v in m.items()}

def load_context(root,bank_path):
    root=Path(root);state=joblib.load(root/'results/audit_20261107/state_cache.joblib')
    train=pd.read_csv(root/'data/raw/train.csv',usecols=['SMILES','SI VACV'])
    test=pd.read_csv(root/'data/raw/test.csv',usecols=['SMILES','index'])
    if (list(train.SMILES)!=list(state['train'].SMILES)
        or not np.array_equal(train['SI VACV'],state['y'])
        or list(test.SMILES)!=list(state['test'].SMILES)
        or list(test['index'])!=list(state['test']['index'])):
        raise ValueError('Stale organizer state cache: regenerate the audit inputs')
    expected=hashlib.sha256(('\n'.join(list(train.SMILES)+list(test.SMILES))).encode()).hexdigest()
    for encoder in ['mlm','mtr']:
        meta=json.loads((root/f'results/audit_20261107/embeddings_{encoder}.json').read_text())
        if meta['input_smiles_sha256']!=expected:raise ValueError('Stale embedding input order')
    bank=pd.read_csv(bank_path).fillna({'source':'','task':'','doi':''})
    actual_keys=[V7.source_key(s) for s in bank.SMILES]
    actual_scaffolds=[T.gscaf(T.std(s)) for s in bank.SMILES]
    if actual_keys!=bank.tautomer_parent.tolist() or actual_scaffolds!=bank.scaffold.fillna('').tolist():
        raise ValueError('Bank graph keys do not match the supplied SMILES')
    n=len(state['y']);reps=fingerprint(list(state['train'].SMILES)+list(bank.SMILES))
    k=kernel(reps,reps);test_keys={V7.source_key(s) for s in state['test'].SMILES}
    if bank.tautomer_parent.isin(test_keys).any():raise ValueError('Forbidden external test labels')
    assert np.isfinite(bank.value).all() and (bank.value>0).all()
    return dict(root=root,state=state,bank=bank,y=state['y'],n=n,K=k,representations=reps,
                embeddings=H.load_embeddings(root/'results/audit_20261107'),
                org_keys=np.asarray([V7.source_key(s) for s in state['train'].SMILES],object),
                org_scaffolds=np.asarray(state['world'].scf[:n],object),
                test_keys=test_keys,probability_cache={})

def allowed_bank(ctx,blocked):
    bank=ctx['bank'];blocked=np.asarray(blocked,int)
    if not len(blocked):return np.ones(len(bank),bool)
    keys=set(ctx['org_keys'][blocked]);scaffolds={s for s in ctx['org_scaffolds'][blocked] if s}
    return (~bank.tautomer_parent.isin(keys)&~bank.scaffold.fillna('').isin(scaffolds)).to_numpy()

def training_arrays(ctx,tr,cfg,blocked):
    tr=np.asarray(tr,int);observed=tr[(ctx['y'][tr]!=0)&(ctx['y'][tr]!=10000)]
    bank=ctx['bank'];valid=allowed_bank(ctx,blocked)
    weight=bank.source.map(cfg['source_weights']).fillna(0).to_numpy(float)
    if not cfg.get('rho',0):weight=np.where(bank.task.eq('SI'),weight,0.)
    ext=np.flatnonzero(valid&(weight>0));indices=np.r_[observed,ctx['n']+ext]
    values=np.r_[ctx['y'][observed],bank.value.to_numpy(float)[ext]]
    tasks=np.r_[np.repeat('SI',len(observed)),bank.task.to_numpy(str)[ext]]
    signs=np.r_[np.ones(len(observed)),bank.sign.to_numpy(float)[ext]]
    weights=np.r_[np.ones(len(observed)),weight[ext]]
    if not len(observed):raise ValueError('No measured training response')
    if not np.isfinite(values).all() or (values<=0).any():raise ValueError('Invalid endpoint label')
    return indices,values,tasks,signs,weights

def fit_core(ctx,tr,cfg,blocked):
    indices,values,tasks,signs,weights=training_arrays(ctx,tr,cfg,blocked)
    transformed=np.log1p(values);means={};scales={}
    for task in sorted(set(tasks)):
        mask=tasks==task;means[task]=float(np.average(transformed[mask],weights=weights[mask]))
        scales[task]=max(float(np.sqrt(np.average((transformed[mask]-means[task])**2,weights=weights[mask]))),.25)
    center=np.asarray([means[t] for t in tasks]);scale=np.asarray([scales[t] for t in tasks])
    target=(transformed-center)/scale;rho=cfg.get('rho',0.)
    task_kernel=(1-rho)*(tasks[:,None]==tasks[None,:])+rho*signs[:,None]*signs[None,:]
    gram=ctx['K'][np.ix_(indices,indices)]*task_kernel
    dual=solve(gram+cfg.get('alpha',3.)*np.diag(1/weights),target,assume_a='pos')
    return dict(indices=indices,tasks=tasks,signs=signs,dual=dual,
                cross_task=np.where(tasks=='SI',1.,rho*signs),
                mean=means['SI'],scale=scales['SI'],cfg=cfg,training_count=len(indices),
                auxiliary_rows=int((tasks!='SI').sum()),
                physical_external_rows=int(((indices>=ctx['n'])&(tasks=='SI')).sum()))

def predict_log(ctx,core,rows):
    return core['mean']+core['scale']*(ctx['K'][np.ix_(rows,core['indices'])]@(core['cross_task']*core['dual']))

def fit_physical(ctx,tr,cfg,blocked):
    tr=np.asarray(tr,int);blocked=np.asarray(blocked,int);core=fit_core(ctx,tr,cfg,blocked)
    observed=tr[(ctx['y'][tr]!=0)&(ctx['y'][tr]!=10000)];log_oof=np.full(len(observed),np.nan)
    for local in T.inner_folds(ctx['org_scaffolds'][observed],5,0):
        va=observed[local];inner_tr=np.setdiff1d(tr,va)
        inner=fit_core(ctx,inner_tr,cfg,np.union1d(blocked,va))
        log_oof[local]=predict_log(ctx,inner,va)
    if not np.isfinite(log_oof).all():raise ValueError('Incomplete calibration')
    core['smear']=float(np.mean(np.exp(np.log1p(ctx['y'][observed])-log_oof)))
    return core

def code_probability(ctx,tr,va):
    key=(tuple(tr),tuple(va))
    if key not in ctx['probability_cache']:
        spec=next(s for s in H.candidate_specs() if s['name']=='notation_mlm_probability_average')
        head=H.fit_head(spec,ctx['state'],ctx['embeddings'],tr)
        ctx['probability_cache'][key]=H.predict_head(head,ctx['state'],ctx['embeddings'],va,tr)
    return ctx['probability_cache'][key]

def predict_fold(ctx,tr,va,cfg,blocked=None,old_method=False):
    blocked=np.union1d(va,[] if blocked is None else blocked).astype(int)
    core=fit_physical(ctx,tr,cfg,blocked)
    r=np.maximum(core['smear']*np.exp(predict_log(ctx,core,va))-1,0)
    if old_method:
        p=H.old_probabilities(ctx['state'],tr,va)
        pred=np.mean([H.complete_predictions(ctx['state'],tr,va,r,p[:,j]) for j in range(2)],axis=0)
    else:pred=H.complete_predictions(ctx['state'],tr,va,r,code_probability(ctx,tr,va))
    return pred,dict(physical_external_rows=core['physical_external_rows'],auxiliary_rows=core['auxiliary_rows'])

def final_bundle(ctx,cfg):
    core=fit_physical(ctx,np.arange(ctx['n']),cfg,np.array([],int))
    old=joblib.load(ctx['root']/'results/audit_20261107/selected_model.joblib')
    baseline=old['baseline_portable_model'].copy();baseline['lookup']={}
    baseline['B_reference']=select_matrix(ctx['representations'],core['indices'])
    baseline['B_mean']=core['mean'];baseline['B_dual']=core['scale']*core['cross_task']*core['dual']
    baseline['B_smear']=core['smear']
    ext=ctx['bank'].iloc[np.unique(core['indices'][core['indices']>=ctx['n']]-ctx['n'])]
    assert not set(ext.tautomer_parent)&ctx['test_keys']
    return dict(selected=old['selected'],head=old['head'],baseline_portable_model=baseline,
                encoding_batch_size=1,encoding_threads=3,
                external_policy='No numerical external test labels; no exact lookup',physical_config=cfg,
                physical_fit_counts={k:core[k] for k in ['training_count','physical_external_rows','auxiliary_rows']})
