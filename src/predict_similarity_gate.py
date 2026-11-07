"""Predict with frozen PR3 experts and a validated saved similarity gate.

Only SMILES and optional index are loaded. The reference bank, expert weights,
source overrides and gate centers remain fixed; no inference-batch statistics
or fitting are performed.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import similarity_experts as E
import similarity_weight as G
from smiles_style import raw_style_features


def load_gate(path):
    config=json.loads(Path(path).read_text())
    params=config['recommended_gate']
    if float(config.get('alpha_source',0.))!=0. or float(params.get('alpha_source',np.nan))!=0.:
        raise ValueError('The PR8 result requires alpha_source=0')
    active=any(float(params.get(name,0.))!=0. for name in ('w0','beta_s','beta_d'))
    if active and not (config.get('validation_passed',False) and config.get('statistical_acceptance_passed',False)):
        raise ValueError('A nonzero recommended gate needs statistical acceptance and completed validation')
    numeric=[float(params[name]) for name in ('w0','s0','d0','beta_s','beta_d')]
    if not np.isfinite(numeric).all():raise ValueError('Nonfinite saved gate parameter')
    if not 0<=numeric[0]<=1:raise ValueError('Invalid saved constant weight')
    return params,config


def predict(model_path,gate_json,input_path,out_path,encoder_cache):
    params,config=load_gate(gate_json)
    source=pd.read_csv(input_path,usecols=lambda name:name in ('SMILES','index'))
    if 'SMILES' not in source:raise ValueError('SMILES column is required')
    if 'index' not in source:source['index']=np.arange(len(source))
    if source.empty:raise ValueError('No input molecules')
    if source['index'].isna().any() or not source['index'].is_unique:
        raise ValueError('index must be nonmissing and unique')
    original_sha=E.sha(model_path);bundle=joblib.load(model_path)
    if bundle.get('selected')!=E.SPEC['name']:
        raise ValueError('Expected the original PR3 paired-notation/MLM bundle')
    if bundle.get('encoding_batch_size')!=1 or bundle.get('encoding_threads')!=3:
        raise ValueError('Unexpected PR3 encoder settings')
    from extract_chemberta import encode_smiles
    smiles=source.SMILES.tolist()
    embeddings=dict(raw_style=raw_style_features(smiles),
                    mlm_raw=encode_smiles(smiles,'mlm','raw',encoder_cache,batch_size=1,threads=3))
    components=E.production(bundle,smiles,embeddings)
    if not np.all(components.alpha_source.to_numpy()==0.):
        raise AssertionError('Source override changed')
    prediction,weights=G.predict_gate(components,params)
    if not np.isfinite(prediction).all() or not np.isfinite(weights).all():
        raise ValueError('Nonfinite prediction or gate weight')
    if ((weights<0)|(weights>1)).any():raise ValueError('Invalid gate weights')
    if E.sha(model_path)!=original_sha:raise AssertionError('Production model changed')
    output=pd.DataFrame({'index':source['index'].to_numpy(),'SI VACV':prediction})
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True)
    output.to_csv(out_path,index=False)
    return dict(rows=len(output),model_sha256=original_sha,alpha_source=0.,gate_family=params.get('family'),
                weight_min=float(weights.min()),weight_max=float(weights.max()),
                statistical_acceptance_passed=bool(config.get('statistical_acceptance_passed',False)),
                validation_passed=bool(config.get('validation_passed',False)),
                encoder_batch_size=1,encoder_threads=3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--gate-json',required=True)
    parser.add_argument('--input',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--encoder-cache',default='work/chemberta_snapshots')
    args=parser.parse_args()
    print(json.dumps(predict(args.model,args.gate_json,args.input,args.out,args.encoder_cache),indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
