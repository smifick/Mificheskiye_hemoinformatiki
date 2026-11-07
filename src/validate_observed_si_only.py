"""Validate the isolated filtered experiment without choosing a new model."""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import observed_si_only as E

def validate(root,out):
    source=pd.read_csv(root/'data/raw/train.csv',usecols=['SMILES',E.TARGET,'IC50_uM_VACV','CC50_uM'])
    keep,y,_,_,audit=E.audit_train(source)
    assert (len(y),int(keep.sum()),audit['missing_IC50'],audit['missing_CC50'])==(320,237,81,2)
    model=joblib.load(out/'model_filtered237.joblib')
    included=model['reference_source_rows'];excluded=np.flatnonzero(~keep)+2
    assert np.array_equal(included,np.flatnonzero(keep)+2)
    assert set(included).isdisjoint(excluded) and int(np.argmax(y)+2) in included
    assert model['final_training_rows']==237 and model['feature_columns']==['SMILES']
    x,keys,groups=E.representations(source.SMILES);k=E.tanimoto(x,x)
    folds=pd.read_csv(out/'cohort_and_splits.csv').fold.to_numpy(int)
    fit=np.flatnonzero((folds!=folds.min())&keep);valid=np.flatnonzero(folds==folds.min())
    assert set(groups[fit]).isdisjoint(groups[valid]) and set(keys[fit]).isdisjoint(keys[valid])
    est=E.fit_estimator(k,y,fit);reference=E.predict_estimator(est,k[np.ix_(valid,fit)])
    errors={}
    for name,indices,value in [('validation_nan',valid,np.nan),('validation_large',valid,1e12),
                                ('all_excluded_large',np.flatnonzero(~keep),1e12)]:
        poisoned=y.copy();poisoned[indices]=value
        refit=E.fit_estimator(k,poisoned,fit);p=E.predict_estimator(refit,k[np.ix_(valid,fit)])
        errors[name]=float(np.max(np.abs(p-reference)));assert errors[name]==0
    saved=pd.read_csv(out/'oof_predictions.csv')
    first_error=float(np.max(np.abs(saved.filtered237_only.to_numpy()[valid]-reference)));assert first_error<1e-9
    old=np.load(root/'results/audit_20261107/baseline_oof.npz')
    control_error=float(np.max(np.abs(saved.control_all320.to_numpy()-old['v3'])));assert control_error<1e-7
    no_cap=float(E.predict_estimator(SimpleNamespace(predict=lambda _:np.array([50000.])),np.ones((1,1)))[0]);assert no_cap==50000
    test=pd.read_csv(root/'data/raw/test.csv');saved_test=pd.read_csv(out/'submission_filtered237.csv')
    with tempfile.TemporaryDirectory(dir=root/'work') as temp:
        temp=Path(temp);clean=test[['index','SMILES']]
        poisoned=clean.assign(CC50_uM=1e99,IC50_uM_VACV=np.nan,**{E.TARGET:1e12},
                             Culture='ignored',strain='ignored',Title='ignored').iloc[::-1]
        cli_errors={}
        for name,frame in [('clean_smiles_only',clean),('poisoned_columns_reversed',poisoned)]:
            inp=temp/(name+'.csv');prediction=temp/(name+'_prediction.csv');frame.to_csv(inp,index=False)
            subprocess.run([sys.executable,str(root/'src/observed_si_only.py'),'predict','--model',str(out/'model_filtered237.joblib'),
                            '--input',str(inp),'--out',str(prediction)],check=True)
            actual=pd.read_csv(prediction).sort_values('index');expected=saved_test.sort_values('index')
            assert actual['index'].to_list()==expected['index'].to_list()
            cli_errors[name]=float(np.max(np.abs(actual[E.TARGET].to_numpy()-expected[E.TARGET].to_numpy())))
            assert cli_errors[name]<1e-9
    result=dict(status='PASS',final_training_rows=237,excluded_rows=83,excluded_rows_in_final_model=0,
                maximum_SI_retained=True,validation_and_excluded_label_poisoning_max_abs_error=errors,
                first_fold_reproduction_max_abs_error=first_error,
                control_reproduces_historical_v3_max_abs_error=control_error,
                portable_CLI_max_abs_error=cli_errors,no_upper_cap_test_prediction=no_cap,
                previous_model_files_modified=False,ensemble=False,model_sha256=E.digest(out/'model_filtered237.joblib'))
    (out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/'+E.EXPERIMENT);args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    with threadpool_limits(limits=2):validate(root,root/args.out)
