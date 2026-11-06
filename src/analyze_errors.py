"""Diagnose saved model errors; no tuning or test-label reconstruction is performed."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root)
    run=root/'results/final_run';out=root/'results/error_analysis';out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(run/'training_dataset.csv')
    holdout=pd.read_csv(run/'holdout_predictions.csv').merge(train.drop(columns='canonical_smiles'),on='source_row',validate='one_to_one')
    holdout['squared_error']=(holdout.observed-holdout.prediction)**2
    holdout['label_kind']=np.where(holdout.IC50_uM_VACV.isna(),'missing_IC50_code_10000',np.where(holdout.CC50_uM.isna(),'missing_CC50_zero_label','both_concentrations_present'))
    rows=[]
    for kind,d in holdout.groupby('label_kind'):
        rows.append({'label_kind':kind,'rows':len(d),'MSE':float(d.squared_error.mean()),'fraction_total_squared_error':float(d.squared_error.sum()/holdout.squared_error.sum()),'observed_mean':float(d.observed.mean()),'prediction_mean':float(d.prediction.mean())})
    pd.DataFrame(rows).to_csv(out/'holdout_error_by_label_kind.csv',index=False)
    columns=['source_row','Title','SMILES','observed','prediction','squared_error','label_kind','Culture']
    holdout.sort_values('squared_error',ascending=False)[columns].to_csv(out/'holdout_errors_ranked.csv',index=False)
    cv=pd.read_csv(run/'cv_predictions.csv');cv=cv[cv.model=='tanimoto_krr_1_raw'].merge(train,on='source_row',validate='one_to_one')
    cv['squared_error']=(cv.observed-cv.prediction)**2
    cv.nlargest(20,'squared_error')[columns[:-2]].to_csv(out/'largest_inner_cv_errors.csv',index=False)
    eligible=train.CC50_uM.notna() & train.IC50_uM_VACV.notna() & train.CC50_uM.gt(0) & train.IC50_uM_VACV.gt(0)
    observed=train[eligible].copy();observed['SI_from_concentrations']=observed.CC50_uM/observed.IC50_uM_VACV
    observed['selection_status']='both_concentrations_present_not_independent_publication_verification'
    observed.to_csv(root/'data/processed/si_with_observed_concentrations.csv',index=False)
    summary={'scope':'Saved-model local diagnosis; hidden Kaggle labels unavailable.',
             'placeholder_meaning':'SI=10000 encodes missing IC50 in the competition dataset; it is not a measured or bounded biological SI.',
             'missing_ic50_rows':int(train.IC50_uM_VACV.isna().sum()),
             'missing_cc50_rows':int(train.CC50_uM.isna().sum()),
             'both_concentrations_present_rows':len(observed),
             'top_5_holdout_fraction_of_squared_error':float(holdout.nlargest(5,'squared_error').squared_error.sum()/holdout.squared_error.sum()),
             'largest_inner_cv_error_fraction':float(cv.squared_error.max()/cv.squared_error.sum()),
             'public_MSE':9992163.26222,'reference_public_MSE':1546537.62825,
             'leaderboard_scope':'Kaggle Public evaluation, approximately 30% of test; recorded 2026-11-06.',
             'model_or_submission_changed':False,
             'interpretation':'The target mixes measured-ratio labels and missing-data codes; MSE penalizes both. Missing IC50 alone says nothing about potency or inactivity. Missing CC50 zero labels require separate clarification.',
             'next_experiment':'Evaluate a two-stage SMILES-only model: missing-IC50-code probability plus conditional SI regression, with expected encoded target 10000*p+(1-p)*E[SI|observed,SMILES]. Validate on grouped folds; do not treat missingness as biological inactivity.',
             'validation_note':'The existing holdout has now been inspected for diagnosis; further tuning against its errors would invalidate its role as untouched evaluation.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
