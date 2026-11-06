"""Portable inference from transparent numeric JSON parameters; SMILES inputs only."""
import argparse,json
import numpy as np,pandas as pd
from scipy.special import expit
from train_model import featurize,tanimoto,TARGET
from train_external_v3 import canonical

def predict(parameters,smiles):
 X=featurize([canonical(s,True)[0] for s in smiles])[0]
 p=expit((X@np.asarray(parameters['classifier_coef'])+parameters['classifier_intercept'])*parameters['calibrator_coef']+parameters['calibrator_intercept'])
 ref=featurize(parameters['regression_smiles'])[0]
 r=tanimoto(X,ref)@np.asarray(parameters['regression_coefficients'])
 if parameters['log']:r=np.expm1(np.clip(r,-20,20))
 r=np.maximum(r,0);slope,intercept=parameters['correction'];r=np.maximum(slope*r+intercept,0)
 return 10000*p+(1-p)*r

def main():
 p=argparse.ArgumentParser();p.add_argument('--parameters',required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 parameters=json.load(open(a.parameters));data=pd.read_csv(a.input,usecols=['index','SMILES']);pred=predict(parameters,data.SMILES)
 assert np.isfinite(pred).all() and (pred>=0).all()
 pd.DataFrame({'index':data['index'],TARGET:pred}).to_csv(a.out,index=False)
if __name__=='__main__':main()
