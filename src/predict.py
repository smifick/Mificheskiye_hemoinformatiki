"""Predict SI from SMILES using the saved, trusted model. Ignores other assay columns."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import rdkit
from train_model import standardize, featurize, tanimoto, DESC, TARGET

def predict(model_path,table):
    bundle=joblib.load(model_path)  # Only load model files from a trusted source.
    config=bundle['config']
    if config['rdkit']!=rdkit.__version__:
        raise ValueError('RDKit version differs from training; use requirements.txt')
    if config['descriptor_names']!=[n for n,_ in DESC]:raise ValueError('Descriptor schema mismatch')
    smiles=[standardize(s)[0] for s in table.SMILES]
    X,D=featurize(smiles)
    kind=config['feature_kind']
    A=tanimoto(X,bundle['train_fingerprints']) if kind=='kernel' else {'bits':X,'desc':D,'both':np.column_stack([X,D])}[kind]
    y=bundle['model'].predict(A)
    if config['target_transform']=='log1p':y=np.expm1(np.clip(y,-20,20))
    y=np.maximum(y,0)
    if not np.isfinite(y).all():raise ValueError('Nonfinite predictions')
    return y

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    table=pd.read_csv(args.input)
    result=pd.DataFrame({'index':table['index'] if 'index' in table else np.arange(len(table)),TARGET:predict(args.model,table)})
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.out,index=False)

if __name__=='__main__':main()
