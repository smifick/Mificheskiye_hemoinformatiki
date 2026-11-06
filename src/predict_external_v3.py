"""Inference using only index and SMILES; all assay columns are excluded at load time."""
import argparse,joblib,numpy as np,pandas as pd
from train_external_v3 import canonical
from train_model import featurize,tanimoto,TARGET

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 b=joblib.load(a.model);d=pd.read_csv(a.input,usecols=['index','SMILES']);X=featurize([canonical(s,b['tautomer'])[0] for s in d.SMILES])[0]
 pred=np.maximum(b['model'].predict(tanimoto(X,b['fingerprints'])),0)
 if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction')
 pd.DataFrame({'index':d['index'],TARGET:pred}).to_csv(a.out,index=False)
if __name__=='__main__':main()
