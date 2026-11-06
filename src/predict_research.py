"""Portable inference for the selected research candidate: only SMILES + IDs."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

import train_v7 as V7
from smiles_style import raw_style_features


def calibration_features(p):
    p=np.clip(p,1e-4,1-1e-4)
    return np.log(p/(1-p))


def infer(bundle,smiles,cache,batch_size=1,threads=3):
    if batch_size!=bundle.get('encoding_batch_size',1) or threads!=bundle.get('encoding_threads',3):
        raise ValueError('Use the frozen encoder batch/thread settings saved with the model')
    source=list(smiles);baseline=bundle['baseline_portable_model'];name=bundle['selected']
    if name=='v7':return V7.infer(baseline,source)
    matrices=V7.matrices(source)
    keys=[V7.source_key(s) for s in source]
    flag=np.array([s in baseline['source_presence'] for s in keys])
    ka=V7.legacy_kernel(matrices,baseline['A_reference'])
    kf=V7.T.tanimoto(matrices['feature'],baseline['A_reference']['feature'])
    p0=np.clip(baseline['A_mean']+ka@baseline['A_legacy_dual'],0,1)
    p1=np.clip(baseline['A_mean']+kf@baseline['A_feature_dual'],0,1)
    p0[flag]=baseline['A_flag_probability'];p1[flag]=baseline['A_flag_probability']
    kb=V7.legacy_kernel(matrices,baseline['B_reference'])
    r=np.maximum(baseline['B_smear']*np.exp(baseline['B_mean']+kb@baseline['B_dual'])-1,0)
    def complete(p,cap=True):
        pred=10000*p+np.clip(1-p-baseline['zero_rate'],0,1)*r
        return np.clip(pred,0,10000) if cap else np.maximum(pred,0)
    base=.5*(complete(p0)+complete(p1))
    if name=='v7_without_upper_cap':pred=.5*(complete(p0,False)+complete(p1,False))
    else:
        feature_cache={}
        def features(key):
            if key not in feature_cache:
                if key=='raw_style':feature_cache[key]=raw_style_features(source)
                else:
                    from extract_chemberta import encode_smiles
                    if key=='both_canonical':feature_cache[key]=np.column_stack([encode_smiles(source,k,'canonical',cache,batch_size,threads) for k in ['mtr','mlm']])
                    else:
                        encoder,view=key.split('_',1)
                        feature_cache[key]=encode_smiles(source,encoder,view,cache,batch_size,threads)
            return feature_cache[key]
        def probability(head):
            spec=head['spec']
            if 'constant' in head:p=np.full(len(source),head['constant'])
            elif spec['family']=='paired':p=np.mean([probability(child) for child in head['components']],axis=0)
            elif spec['family']=='calibrated':p=head['estimator'].predict_proba(calibration_features(np.column_stack([p0,p1])))[:,1]
            else:
                z=features(spec['embedding'])
                if spec.get('include_source_flag'):z=np.column_stack([z,flag.astype(float)])
                z=head['scaler'].transform(z)
                if spec['family']=='logistic':p=head['estimator'].predict_proba(z)[:,1]
                else:
                    if spec.get('training_scope')=='all':raise ValueError('Unsupported full-row fusion reference')
                    ke=np.exp(-cdist(normalize(z),head['embedding_reference'],metric='sqeuclidean')/head['bandwidth'])
                    p=np.clip(head['mean']+.5*(ke+.5*(ka+kf))@head['dual'],0,1)
            return p if spec.get('training_scope')=='all' else np.where(flag,head['p_flag'],p)
        p=probability(bundle['head'])
        pred=complete(p)
        if name.endswith('_blend50'):pred=.5*(base+pred)
    ext=np.array([baseline['lookup'].get(s,np.nan) for s in keys]);covered=np.isfinite(ext)
    values=np.maximum(np.nan_to_num(ext),0) if name=='v7_without_upper_cap' else np.clip(np.nan_to_num(ext),0,10000)
    return np.where(covered,values,pred)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True)
    p.add_argument('--out',required=True);p.add_argument('--cache',default='work/chemberta_snapshots')
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();model=joblib.load(a.model)
    # All forbidden fields are excluded at file-read time.
    data=pd.read_csv(a.input,usecols=lambda col:col in ['SMILES','index'])
    if 'SMILES' not in data:raise ValueError('SMILES is required')
    if 'index' not in data:data['index']=np.arange(len(data))
    if data['index'].isna().any() or not data['index'].is_unique:raise ValueError('Invalid IDs')
    pred=infer(model,data.SMILES,a.cache,a.batch_size,a.threads)
    if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Invalid predictions')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'index':data['index'],'SI VACV':pred}).to_csv(out,index=False)


if __name__=='__main__':main()
