"""Exact numeric source-override ablation of the fixed PR3 paired head.

Each leaf's individual probability is exported BEFORE routing. PR3 applies
identical source probabilities a second time at its paired wrapper; that
idempotent duplicate is consolidated so alpha acts ONCE, never alpha-squared.
All downstream clipping, zero handling and lookup are retained verbatim.
"""
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import research_heads as H
import train_v7 as V7
from smiles_style import raw_style_features

COMMIT='b6c2e0c384d043582753993e442be7c042c3139a'
GRID=(0.,.025,.05,.1,.2)
SPEC=next(s for s in H.candidate_specs() if s['name']=='notation_mlm_probability_average')
HEADS=('notation','mlm')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def leaf_raw(head,features):
    if 'constant' in head:return np.full(len(features),head['constant'])
    assert head['spec']['family']=='logistic'
    return head['estimator'].predict_proba(head['scaler'].transform(features))[:,1]

def leaf_probabilities(head,emb,rows):
    assert head['spec']['name']==SPEC['name'] and len(head['components'])==2
    raw=[];group=[]
    for child in head['components']:
        assert child['spec'].get('training_scope','unflagged')!='all'
        raw.append(leaf_raw(child,emb[child['spec']['embedding']][rows]))
        group.append(float(child['p_flag']))
    # In this exact baseline the wrapper repeats the SAME rule. Refuse to
    # silently merge different group probabilities from any future model.
    assert group[0]==group[1]==float(head['p_flag'])
    return raw,group

def components(head,emb,rows,source,r,zero,keys,scaffolds,smiles,lookup=None):
    raw,pg=leaf_probabilities(head,emb,rows)
    table=pd.DataFrame(dict(source_present=np.asarray(source,bool),physical_prediction=np.asarray(r,float),
                            zero_rate=float(zero),structure_group=list(keys),scaffold_group=list(scaffolds),SMILES=list(smiles),
                            notation_weight=.5,mlm_weight=.5,legacy_ensemble_weight=0.))
    for label,p,g in zip(HEADS,raw,pg):table['p_molecular_'+label]=p;table['p_group_'+label]=g
    table['p_group_parent']=float(head['p_flag'])
    values=np.full(len(rows),np.nan) if lookup is None else np.array([lookup.get(k,np.nan) for k in keys],float)
    table['lookup_applied']=np.isfinite(values);table['lookup_value']=values
    return table

def apply_alpha(table,alpha,details=False):
    if not 0<=alpha<=1:raise ValueError('alpha outside [0,1]')
    source=table.source_present.to_numpy(bool);probabilities=[]
    for label in HEADS:
        pm=table['p_molecular_'+label].to_numpy(float);pg=table['p_group_'+label].to_numpy(float)
        probabilities.append(np.where(source,(1-alpha)*pg+alpha*pm,pm))
    p=table.notation_weight.to_numpy()*probabilities[0]+table.mlm_weight.to_numpy()*probabilities[1]
    r=table.physical_prediction.to_numpy();z=table.zero_rate.to_numpy()
    numeric=np.clip(10000*p+np.clip(1-p-z,0,1)*r,0,10000)
    final=np.where(table.lookup_applied.to_numpy(bool),np.clip(table.lookup_value.to_numpy(),0,10000),numeric)
    if not np.isfinite(final).all():raise ValueError('Nonfinite prediction')
    return (final,p,np.column_stack(probabilities),numeric) if details else final

def effect_counts(table,alpha=.2):
    p0=apply_alpha(table,0);pa=apply_alpha(table,alpha)
    source=table.source_present.to_numpy(bool)
    different=source&np.any(np.column_stack([np.abs(table['p_molecular_'+h]-table['p_group_'+h])>1e-12 for h in HEADS]),axis=1)
    affected=np.abs(pa-p0)>1e-9
    return dict(rows=len(table),source_present=int(source.sum()),lookup_applied=int(table.lookup_applied.sum()),
                source_and_lookup=int((source&table.lookup_applied.to_numpy(bool)).sum()),
                source_individual_differs=int(different.sum()),effective_rows=int(affected.sum()),
                effective_structures=int(table.loc[affected,'structure_group'].nunique()),
                effective_scaffolds=int(table.loc[affected,'scaffold_group'].nunique()))

def choose_alpha(table,grid=GRID):
    info=effect_counts(table);y=table.observed.to_numpy()
    scores={str(a):float(np.mean((apply_alpha(table,a)-y)**2)) for a in grid}
    enough=info['effective_structures']>=2 and info['effective_scaffolds']>=2
    chosen=min(grid,key=lambda a:(scores[str(a)],a)) if enough else 0.
    return float(chosen),dict(MSE=scores,information_sufficient=bool(enough),effects=info)

def prepare(cache):
    state=joblib.load(cache/'state_cache.joblib')
    smiles=np.array(list(state['train'].SMILES)+list(state['test'].SMILES),object)
    manifest=json.loads((cache/'embedding_manifest.json').read_text())
    assert manifest['input_sha256']==hashlib.sha256('\n'.join(smiles).encode()).hexdigest()
    emb={'raw_style':raw_style_features(smiles),'mlm_raw':np.load(cache/'mlm_raw.npy')}
    return state,emb,smiles

def production_components(bundle,smiles,emb):
    base=bundle['baseline_portable_model'];rows=np.arange(len(smiles))
    assert bundle['selected']==SPEC['name']
    assert bundle['encoding_batch_size']==1 and bundle['encoding_threads']==3
    arrays=V7.matrices(smiles);keys=[V7.source_key(s) for s in smiles]
    flag=np.array([s in base['source_presence'] for s in keys])
    kb=V7.legacy_kernel(arrays,base['B_reference'])
    r=np.maximum(base['B_smear']*np.exp(base['B_mean']+kb@base['B_dual'])-1,0)
    scaffolds=[V7.T.gscaf(V7.T.std(s)) for s in smiles]
    result=components(bundle['head'],emb,rows,flag,r,base['zero_rate'],keys,scaffolds,smiles,base['lookup'])
    ka=V7.legacy_kernel(arrays,base['A_reference'])
    kf=V7.T.tanimoto(arrays['feature'],base['A_reference']['feature'])
    for name,k,dual in [('legacy',ka,base['A_legacy_dual']),('feature',kf,base['A_feature_dual'])]:
        p=np.clip(base['A_mean']+k@dual,0,1)
        result['fixed_unused_'+name+'_probability']=np.where(flag,base['A_flag_probability'],p)
    return result
