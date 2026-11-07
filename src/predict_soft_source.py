"""Portable prediction using the original PR3 weights and one source alpha.

Only SMILES and the optional index column are read. An explicit --alpha is a
diagnostic override; deployment should use the validated --alpha-json file.
No estimator, probability calibrator or feature transformer is refitted.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import soft_source as S
from smiles_style import raw_style_features


def selected_alpha(alpha=None,alpha_json=None):
    if (alpha is None)==(alpha_json is None):
        raise ValueError('Specify exactly one of alpha and alpha_json')
    if alpha_json is not None:
        config=json.loads(Path(alpha_json).read_text())
        alpha=float(config['recommended_alpha'])
        # A positive recommendation must be explicitly marked validated, even
        # if a future configuration omits validation_required_before_use.
        if alpha>0 and not config.get('validation_passed',False):
            raise ValueError('Positive recommended alpha has not passed validation')
    alpha=float(alpha)
    if not np.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('alpha must be finite and lie in [0,1]')
    return alpha


def predict(model_path,input_path,out_path,encoder_cache,alpha):
    from extract_chemberta import encode_smiles
    source=pd.read_csv(input_path,usecols=lambda name:name in ('SMILES','index'))
    if 'SMILES' not in source:raise ValueError('SMILES column is required')
    if 'index' not in source:source['index']=np.arange(len(source))
    if source['index'].isna().any() or not source['index'].is_unique:
        raise ValueError('index must be nonmissing and unique')
    if source.empty:raise ValueError('No input molecules')
    before=S.sha(model_path);bundle=joblib.load(model_path)
    if bundle.get('encoding_batch_size')!=1 or bundle.get('encoding_threads')!=3:
        raise ValueError('Unexpected PR3 encoder settings')
    smiles=source.SMILES.tolist()
    embeddings=dict(raw_style=raw_style_features(smiles),
                    mlm_raw=encode_smiles(smiles,'mlm','raw',encoder_cache,batch_size=1,threads=3))
    components=S.production_components(bundle,smiles,embeddings)
    prediction,p,leaves,_=S.apply_alpha(components,alpha,details=True)
    if not (np.isfinite(prediction).all() and np.isfinite(p).all() and np.isfinite(leaves).all()):
        raise ValueError('Nonfinite probability or prediction')
    if ((leaves<0)|(leaves>1)).any():raise ValueError('Invalid probability')
    if S.sha(model_path)!=before:raise AssertionError('Production model changed')
    output=pd.DataFrame({'index':source['index'].to_numpy(),'SI VACV':prediction})
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True)
    output.to_csv(out_path,index=False)
    return dict(rows=len(output),alpha=alpha,model_sha256=before,
                encoder_batch_size=1,encoder_threads=3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    choose=parser.add_mutually_exclusive_group(required=True)
    choose.add_argument('--alpha-json')
    choose.add_argument('--alpha',type=float,help='Explicit diagnostic override; does not establish a recommendation')
    parser.add_argument('--input',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--cache',default='work/chemberta_snapshots',help='Frozen encoder snapshot cache')
    args=parser.parse_args()
    alpha=selected_alpha(args.alpha,args.alpha_json)
    result=predict(args.model,args.input,args.out,args.cache,alpha)
    result['diagnostic_override']=args.alpha is not None
    result['alpha_source']='explicit_diagnostic' if args.alpha is not None else str(args.alpha_json)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
