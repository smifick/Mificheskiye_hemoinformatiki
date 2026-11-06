"""Inference from the locally trained v7 bundle, using SMILES only."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from train_v7 import infer


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--input',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    df=pd.read_csv(args.input,usecols=lambda name:name in ['index','SMILES'])
    if 'SMILES' not in df:raise ValueError('SMILES column is required')
    if 'index' not in df:df['index']=range(len(df))
    if not df['index'].is_unique:raise ValueError('Duplicate indices')
    model=joblib.load(args.model)
    if model.get('format_version')!=1:raise ValueError('Unsupported model format')
    pred=infer(model,df.SMILES)
    result=pd.DataFrame({'index':df['index'],'SI VACV':pred})
    target=Path(args.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(target,index=False)


if __name__=='__main__':main()
