"""Frozen PR3/v7 experts at the FINAL SI level; source softening stays zero."""
import hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
import audit_regression as A
import research_heads as H
import train_v7 as V7
from smiles_style import raw_style_features

BASE_COMMIT='b6c2e0c384d043582753993e442be7c042c3139a'
SPEC=next(s for s in H.candidate_specs() if s['name']=='notation_mlm_probability_average')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def prepare(cache):
    state=joblib.load(cache/'state_cache.joblib')
    smiles=np.array(list(state['train'].SMILES)+list(state['test'].SMILES),object)
    manifest=json.loads((cache/'embedding_manifest.json').read_text())
    assert manifest['input_sha256']==hashlib.sha256('\n'.join(smiles).encode()).hexdigest()
    emb=dict(raw_style=raw_style_features(smiles),mlm_raw=np.load(cache/'mlm_raw.npy'))
    state['structural_bits']=np.load(cache/'structural_bits.npy')
    state['similarity_kernel']=V7.T.tanimoto(state['structural_bits'],state['structural_bits'])
    state['cache_namespace']=hashlib.sha256(state['world'].K.tobytes()+state['world'].ext_si.tobytes()+
        state['flag'].tobytes()+state['structural_bits'].tobytes()+emb['mlm_raw'].tobytes()+
        json.dumps([SPEC,A.V6.CFG,BASE_COMMIT]).encode()).hexdigest()
    return state,emb,smiles

def complete(p,r,z):return np.clip(10000*p+np.clip(1-p-z,0,1)*r,0,10000)

def finish(table,pred):
    return np.where(table.lookup_applied.to_numpy(bool),np.clip(table.lookup_value.to_numpy(),0,10000),pred)

def similarity_to_training(state,tr,query):
    reference=np.asarray(tr,int)[~state['flag'][tr]]
    if not len(reference):raise ValueError('Empty structural classifier reference bank')
    assert set(state['groups'][reference]).isdisjoint(state['groups'][query])
    assert set(state['world'].tau[reference]).isdisjoint(state['world'].tau[query])
    return state['similarity_kernel'][np.ix_(query,reference)].max(axis=1),reference

def make_fold(state,emb,smiles,tr,va,blocked,cache,registry,force=False):
    tr=np.asarray(tr,int);va=np.asarray(va,int);blocked=np.asarray(blocked,int);w=state['world']
    assert set(state['groups'][tr]).isdisjoint(state['groups'][va])
    key=hashlib.sha256(json.dumps([tr.tolist(),va.tolist(),blocked.tolist(),state['cache_namespace']]).encode()+state['y'][tr].tobytes()).hexdigest()
    path=cache/('components_'+key+'.csv.gz');meta=cache/('components_'+key+'.json');model=cache/('head_'+key+'.joblib')
    if path.exists() and not force:
        registry[key]=json.loads(meta.read_text());return pd.read_csv(path)
    head=H.fit_head(SPEC,state,emb,tr)
    r=H.physical(state,tr,va,blocked=blocked)
    z=float(np.mean(state['y'][tr]==0));pS=H.old_probabilities(state,tr,va)
    pM=H.predict_head(head,state,emb,va,tr)
    S=.5*(complete(pS[:,0],r,z)+complete(pS[:,1],r,z));M=complete(pM,r,z)
    s,reference=similarity_to_training(state,tr,va)
    table=pd.DataFrame(dict(source_row=va+2,observed=state['y'][va],SMILES=smiles[va],
        structure_group=w.tau[va],scaffold_group=state['groups'][va],source_present=state['flag'][va],
        pred_S=S,pred_M=M,s=s,d=np.abs(S-M)/10000.,lookup_applied=False,lookup_value=np.nan,
        physical_prediction=r,zero_rate=z,p_structural_legacy=pS[:,0],p_structural_feature=pS[:,1],p_M=pM,
        p_group=head['p_flag'],alpha_source=0.,model_id=key,reference_count=len(reference)))
    np.testing.assert_allclose(S[state['flag'][va]],M[state['flag'][va]],atol=1e-9,rtol=1e-14)
    mask=A.T.ext_allowed(w,va)
    if len(blocked):mask &= A.T.ext_allowed(w,blocked)
    for denied in [va,blocked]:
        assert set(w.tau[w.ext_ids[mask]]).isdisjoint(w.tau[denied])
        assert set(w.scf[w.ext_ids[mask]][w.scf[w.ext_ids[mask]]!='']).isdisjoint(w.scf[denied][w.scf[denied]!=''])
        assert set(state['groups'][reference]).isdisjoint(state['groups'][denied])
    joblib.dump(head,model,compress=3)
    provenance=dict(training_source_rows=(tr+2).tolist(),validation_source_rows=(va+2).tolist(),blocked_source_rows=(blocked+2).tolist(),
        reference_source_rows=(reference+2).tolist(),reference_fingerprint_sha256=array_sha(state['structural_bits'][reference]),
        reference_branch='legacy structural CODE classifier A, organizer source-absent only; no external/test',
        molecular_head_training_source_rows=[(c['fit']+2).tolist() for c in head['components']],
        head_sha256=sha(model),external_physical_rows=int(mask.sum()),external_allowed_mask_sha256=array_sha(mask),
        physical_config=A.V6.CFG,alpha_source=0.,p_group=head['p_flag'],zero_rate=z,
        composition='S=average50:50 complete legacy/feature; M=complete paired notation/MLM; mix finalSI then lookup')
    registry[key]=provenance;meta.write_text(json.dumps(provenance,indent=2))
    table.to_csv(path,index=False,compression={'method':'gzip','mtime':0});return table

def paired_probability(head,emb,source):
    if 'constant' in head:p=np.full(len(source),head['constant'])
    elif head['spec']['family']=='paired':p=np.mean([paired_probability(c,emb,source) for c in head['components']],axis=0)
    else:
        assert head['spec']['family']=='logistic'
        p=head['estimator'].predict_proba(head['scaler'].transform(emb[head['spec']['embedding']]))[:,1]
    return np.where(source,head['p_flag'],p)

def production(bundle,smiles,emb):
    base=bundle['baseline_portable_model'];a=V7.matrices(smiles);keys=[V7.source_key(s) for s in smiles]
    assert bundle['selected']==SPEC['name'] and base['feature_weight']==.5
    assert base['A_flag_probability']==bundle['head']['p_flag']
    source=np.array([k in base['source_presence'] for k in keys]);ka=V7.legacy_kernel(a,base['A_reference'])
    kf=V7.T.tanimoto(a['feature'],base['A_reference']['feature'])
    ps=[]
    for k,dual in [(ka,base['A_legacy_dual']),(kf,base['A_feature_dual'])]:
        p=np.clip(base['A_mean']+k@dual,0,1);ps.append(np.where(source,base['A_flag_probability'],p))
    kb=V7.legacy_kernel(a,base['B_reference']);r=np.maximum(base['B_smear']*np.exp(base['B_mean']+kb@base['B_dual'])-1,0)
    z=base['zero_rate'];pM=paired_probability(bundle['head'],emb,source)
    S=.5*(complete(ps[0],r,z)+complete(ps[1],r,z));M=complete(pM,r,z)
    values=np.array([base['lookup'].get(k,np.nan) for k in keys],float)
    s=V7.T.tanimoto(a['bits'],base['A_reference']['bits']).max(axis=1)
    t=pd.DataFrame(dict(SMILES=smiles,structure_group=keys,scaffold_group=[V7.T.gscaf(V7.T.std(x)) for x in smiles],
        source_present=source,pred_S=S,pred_M=M,s=s,d=np.abs(S-M)/10000.,lookup_applied=np.isfinite(values),
        lookup_value=values,physical_prediction=r,zero_rate=z,p_structural_legacy=ps[0],p_structural_feature=ps[1],
        p_M=pM,p_group=bundle['head']['p_flag'],alpha_source=0.,reference_count=len(base['A_reference']['bits'])))
    np.testing.assert_allclose(S[source],M[source],atol=1e-9,rtol=1e-14)
    return t
