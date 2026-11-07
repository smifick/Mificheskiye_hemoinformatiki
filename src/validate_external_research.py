"""Check the actual saved model, external exclusions and SMILES-only CLI."""
from pathlib import Path
import hashlib,json,subprocess,sys,tempfile
import joblib
import numpy as np
import pandas as pd
import train_v7 as V7
import external_si_model as M
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/external_20261107'

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--cache',default='work/chemberta_snapshots');a=ap.parse_args()
    saved=pd.read_csv(OUT/'submission.csv');test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['index','SMILES'])
    sample=pd.read_csv(ROOT/'data/raw/sample_submit.csv',usecols=['index'])
    bundle=joblib.load(OUT/'model.joblib');assert bundle['baseline_portable_model']['lookup']=={}
    bank=pd.read_csv(ROOT/'data/external_research/training_bank.csv')
    assert [V7.source_key(s) for s in bank.SMILES]==bank.tautomer_parent.tolist()
    testkeys={V7.source_key(s) for s in test.SMILES};assert not set(bank.tautomer_parent)&testkeys
    cfg=bundle['physical_config'];weights=bank.source.map(cfg['source_weights']).fillna(0).to_numpy(float)
    if not cfg['rho']:weights=np.where(bank.task.eq('SI'),weights,0)
    used=bank[weights>0].copy();used.to_csv(OUT/'final_used_external.csv',index=False)
    assert len(used)==bundle['physical_fit_counts']['physical_external_rows']+bundle['physical_fit_counts']['auxiliary_rows']
    assert not set(used.tautomer_parent)&testkeys
    errors=[]
    work=ROOT/'work';work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='inference_check_',dir=work) as d:
        d=Path(d);only=test[['SMILES']].copy();poison=test.iloc[::-1].copy()
        for column in ['CC50_uM','IC50_uM_VACV','SI VACV','Title','strain','Culture']:poison[column]=1e99
        for name,frame in [('only_smiles',only),('poisoned_reversed',poison)]:
            source=d/(name+'.csv');dest=d/(name+'_prediction.csv');frame.to_csv(source,index=False)
            subprocess.run([sys.executable,str(ROOT/'src/predict_research.py'),'--model',str(OUT/'model.joblib'),
              '--input',str(source),'--out',str(dest),'--cache',str(a.cache)],check=True,cwd=ROOT)
            pred=pd.read_csv(dest).sort_values('index').reset_index(drop=True)
            ref=saved.sort_values('index').reset_index(drop=True)
            assert list(pred.columns)==['index','SI VACV'] and pred['index'].equals(ref['index'])
            delta=float(np.max(abs(pred['SI VACV'].to_numpy()-ref['SI VACV'].to_numpy())))
            if delta>1e-6:raise AssertionError('CLI predictions drifted: '+str(delta))
            errors.append(dict(input=name,max_absolute_error=delta))
    result={'rows':len(saved),'columns':list(saved.columns),'ID_match_sample':saved['index'].equals(sample['index']),
      'finite':bool(np.isfinite(saved['SI VACV']).all()),'nonnegative':bool((saved['SI VACV']>=0).all()),
      'only_smiles_CLI_checks':errors,'external_test_labels_used':0,'exact_lookup_entries':0,
      'actual_external_training_rows':len(used),'external_sources':used.source.value_counts().to_dict(),
      'model_sha256':hashlib.sha256((OUT/'model.joblib').read_bytes()).hexdigest(),
      'submission_sha256':hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest()}
    assert result['ID_match_sample'] and result['finite'] and result['nonnegative']
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
