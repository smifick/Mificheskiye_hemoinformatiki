"""Validate the final saved blend and estimate paired scaffold uncertainty."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import train_v7 as V


def main():
    root=V.T.ROOT
    r=root/'results/v7_final'
    model=joblib.load(r/'model.joblib')
    test=pd.read_csv(root/'data/raw/test.csv',usecols=['index','SMILES'])
    new=pd.read_csv(r/'submission_v7.csv')
    a=pd.read_csv(root/'submission_v6.csv')
    b=pd.read_csv(root/'results/v7/submission_v7.csv')
    err=float(np.max(np.abs(new['SI VACV']-.5*(a['SI VACV']+b['SI VACV']))))
    assert err<1e-7
    repeated=V.infer(model,test.SMILES)
    reloaderr=float(np.max(np.abs(repeated-new['SI VACV'])))
    assert reloaderr<1e-7
    p=pd.read_csv(root/'results/v7/confirmation_oof.csv').pivot(index=['seed','source_row','observed'],columns='model',values='prediction')
    y=p.index.get_level_values('observed').to_numpy()
    p['blend50']=(p.v6+p.feature)*.5
    p['delta']=(p.blend50-y)**2-(p.v6-y)**2
    rowdelta=p['delta'].groupby(level='source_row').mean().sort_index().to_numpy()
    w=joblib.load(root/'results/v7/world.joblib')[0]
    groups=w.groups
    ug=np.unique(groups)
    sums=np.array([rowdelta[groups==g].sum() for g in ug])
    sizes=np.array([(groups==g).sum() for g in ug])
    rng=np.random.default_rng(20261107)
    ix=rng.integers(0,len(ug),size=(20000,len(ug)))
    boot=sums[ix].sum(1)/sizes[ix].sum(1)
    ci=np.quantile(boot,[.025,.975]).tolist()
    byseed=[]
    for seed,d in p.groupby(level='seed'):
        yy=d.index.get_level_values('observed').to_numpy()
        byseed.append(dict(seed=int(seed),v6_MSE=float(np.mean((d.v6-yy)**2)),v7_MSE=float(np.mean((d.blend50-yy)**2))))
    cf=pd.DataFrame(byseed)
    cf.to_csv(r/'confirmation_blend.csv',index=False)
    # Save a compact, auditable error table without treating repeated folds as
    # independent observations.
    compact=p[['v6','blend50','delta']].groupby(level=['source_row','observed']).mean().reset_index()
    compact['scaffold_group']=w.groups
    compact.to_csv(r/'paired_oof_mean.csv',index=False)
    summary={'baseline_v6_mean_MSE':float(cf.v6_MSE.mean()),'candidate_v7_mean_MSE':float(cf.v7_MSE.mean()),
     'relative_change_pct':float(100*(cf.v7_MSE.mean()/cf.v6_MSE.mean()-1)),
     'seeds_better':int((cf.v7_MSE<cf.v6_MSE).sum()),'n_seeds':len(cf),
     'paired_MSE_change':float(rowdelta.mean()),'scaffold_cluster_bootstrap_95pct_CI':ci,
     'bootstrap_n':20000,'bootstrap_note':'Exploratory interval on reused development data; does not account for all model-search selection.',
     'independent_improvement_established':False,
     'nested_shortlist':json.loads((root/'results/v7/nested_summary.json').read_text()),
     'parent_repeated_cv':pd.read_csv(root/'results/v7/parent_confirmation.csv').groupby('model').MSE.mean().to_dict(),
     'v7_public_MSE':None,'v6_public_MSE_user_reported':9649437.79854}
    (r/'evaluation.json').write_text(json.dumps(summary,indent=2))
    report={'rows':len(new),'columns':list(new),
     'index_matches_sample':new['index'].equals(pd.read_csv(root/'data/raw/sample_submit.csv')['index']),
     'finite':bool(np.isfinite(new['SI VACV']).all()),'nonnegative':bool(new['SI VACV'].ge(0).all()),
     'max_abs_error_vs_expected_blend':err,'max_abs_error_reloaded_model':reloaderr,
     'changed_rows_vs_v6':int((np.abs(new['SI VACV']-a['SI VACV'])>1e-7).sum()),
     'prediction_difference_RMSE':float(np.sqrt(np.mean((new['SI VACV']-a['SI VACV'])**2)))}
    assert report['index_matches_sample'] and report['finite'] and report['nonnegative']
    (r/'validation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
