"""Generate a Kaggle CSV while reading only SMILES and optional row IDs."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .modeling import predict_model


def predict_frame(bundle,data,smiles_col=None,id_col='index',sample=None):
    col=smiles_col or bundle['smiles_col']
    if col not in data:raise ValueError('Missing SMILES column: '+col)
    # Ignore all other input fields, including any accidentally provided labels,
    # concentration columns, missingness indicators, or assay metadata.
    clean=data[[col]+([id_col] if id_col in data else [])].copy()
    if id_col not in clean:
        if sample is not None:
            if len(sample)!=len(clean):raise ValueError('Sample/input row counts differ')
            clean[id_col]=sample[id_col].to_numpy()
        else:clean[id_col]=np.arange(len(clean))
    if clean[id_col].isna().any() or not clean[id_col].is_unique:raise ValueError('Invalid or duplicate test IDs')
    features=bundle['featurizer'].transform(clean[col])
    pred=predict_model(bundle['model'],features)
    if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Nonfinite or negative predictions')
    result=pd.DataFrame({id_col:clean[id_col],bundle['target_col']:pred})
    if sample is not None:
        if len(sample)!=len(result) or not sample[id_col].is_unique:raise ValueError('Invalid sample IDs')
        if set(sample[id_col])!=set(result[id_col]):raise ValueError('Sample IDs differ from test IDs')
        result=sample[[id_col]].merge(result,on=id_col,how='left',validate='one_to_one',sort=False)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--test',required=True)
    parser.add_argument('--out',default='submission.csv')
    parser.add_argument('--smiles-col')
    parser.add_argument('--id-col',default='index')
    parser.add_argument('--sample')
    args=parser.parse_args()
    bundle=joblib.load(args.model)
    if bundle.get('format_version')!=1:raise ValueError('Unsupported model format')
    col=args.smiles_col or bundle['smiles_col']
    data=pd.read_csv(args.test,usecols=lambda name:name in [col,args.id_col])
    sample=pd.read_csv(args.sample,usecols=[args.id_col]) if args.sample else None
    result=predict_frame(bundle,data,col,args.id_col,sample)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(out,index=False)
    print(f'Saved {len(result)} predictions to {out}')


if __name__=='__main__':main()
