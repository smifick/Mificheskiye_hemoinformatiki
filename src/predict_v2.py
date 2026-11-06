"""SMILES-only inference for the nested-CV v2 experiment."""
import argparse
from pathlib import Path
import joblib,numpy as np,pandas as pd
from train_model import standardize,TARGET
from train_v2 import extended_features,kernels,stage_predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    saved=joblib.load(a.model)
    data=pd.read_csv(a.input)
    F=extended_features([standardize(s)[0] for s in data.SMILES])
    K=kernels(F,saved['features']);pred=stage_predict(saved['bundle'],np.arange(len(data)),F,K)
    if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Invalid prediction')
    result=pd.DataFrame({'index':data['index'] if 'index' in data else np.arange(len(data)),TARGET:pred})
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.out,index=False)

if __name__=='__main__':main()
