"""Paired uncertainty, error attribution, final conservative selection and submission."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from benchmark_si_tail_evidence import metrics, strata
from si_tail_evidence import NonnegativeProjection, candidate_specs, fit_spec, nested_selection, structures, subset
from train_model import tanimoto


def cluster_interval(delta, groups, seed=8911, n_bootstrap=2000):
    """Resample scaffold clusters; repeats of the same row are not new samples."""
    _, codes = np.unique(groups, return_inverse=True)
    sizes = np.bincount(codes)
    sums = np.bincount(codes, weights=delta)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        multiplicity = np.bincount(rng.integers(0,len(sizes),size=len(sizes)),minlength=len(sizes))
        estimates[i] = np.dot(multiplicity,sums)/np.dot(multiplicity,sizes)
    return {'mean_MSE_difference': float(np.mean(delta)),
            'cluster_bootstrap_95pct': np.quantile(estimates,[.025,.975]).tolist(),
            'bootstrap_replicates': n_bootstrap, 'seed': seed,
            'scope': 'Conditional diagnostic on saved OOF losses; no refitting during bootstrap'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',default='results/si_tail_evidence/benchmark')
    parser.add_argument('--out',default='results/si_tail_evidence/final')
    args=parser.parse_args()
    source,out=Path(args.benchmark),Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    oof=pd.read_csv(source/'oof_predictions.csv')
    train=pd.read_csv('data/raw/train.csv',usecols=['SMILES','SI VACV'])
    features,keys,groups=structures(train.SMILES)
    y=train['SI VACV'].to_numpy(float)
    scores,top_errors=[],[]
    losses={}
    for (seed,name),frame in oof.groupby(['seed','model'],sort=True):
        frame=frame.sort_values('row_index')
        p=frame.prediction.to_numpy(float)
        for projected in [False,True]:
            pred=np.maximum(p,0) if projected else p
            variant=name+'_nonnegative' if projected else name
            losses[seed,variant]=(pred-y)**2
            for stratum,mask in strata(y,frame.label_kind.to_numpy(),frame.tail_threshold_from_training.to_numpy()).items():
                scores.append({'seed':int(seed),'model':variant,'stratum':stratum,**metrics(y,pred,mask)})
        ordered=np.argsort((p-y)**2)[::-1]
        total=float(np.sum((p-y)**2))
        for rank,i in enumerate(ordered[:10],1):
            top_errors.append({'seed':int(seed),'model':name,'rank':rank,'row_index':int(i),
                               'observed':y[i],'prediction':p[i],'squared_error':float((p[i]-y[i])**2),
                               'fraction_of_total_SSE':float((p[i]-y[i])**2/total)})
    table=pd.DataFrame(scores)
    table.to_csv(out/'comparison_with_nonnegative_controls.csv',index=False)
    pd.DataFrame(top_errors).to_csv(out/'largest_error_contributions.csv',index=False)
    for mode in ['benchmark','extrapolation']:
        path=source if mode=='benchmark' else source.parent/'extrapolation'
        if not (path/'oof_predictions.csv').exists():
            continue
        predictions=pd.read_csv(path/'oof_predictions.csv')
        splits=pd.read_csv(path/'validation_splits.csv')
        range_rows=[]
        for (seed,fold),partition in splits.groupby(['seed','fold']):
            fit=partition[partition.role.eq('training')].row_index.to_numpy(int)
            valid=partition[partition.role.eq('validation')].row_index.to_numpy(int)
            measured=fit[(y[fit]>0)&(y[fit]!=10000)]
            max_physical=float(y[measured].max())
            for name,frame in predictions[predictions.seed.eq(seed)].groupby('model'):
                frame=frame.set_index('row_index').loc[valid]
                physical=frame.label_kind.eq('measured_ratio').to_numpy()
                p=frame.prediction.to_numpy()
                mask=physical & (y[valid]>max_physical)
                row={'seed':int(seed),'fold':int(fold),'model':name,'train_n':len(fit),
                     'train_max':float(y[fit].max()),'train_physical_max':max_physical,
                     'valid_max':float(y[valid].max()),'pred_max':float(p.max()),
                     **metrics(y[valid],p,mask)}
                range_rows.append(row)
        pd.DataFrame(range_rows).to_csv(out/f'{mode}_above_physical_training_range.csv',index=False)
    seeds=sorted(oof.seed.unique())
    baseline_names=['baseline_krr_nonnegative','baseline_lgbm_nonnegative']
    means=table[table.stratum.eq('all')].groupby('model').MSE.mean()
    strongest_baseline=min(baseline_names,key=lambda name:means[name])
    intervals=[]
    for name in ['mean_glm_nonnegative','smearing_nonnegative','nested_selector_nonnegative']:
        for reference in baseline_names:
            delta=np.mean([losses[seed,name]-losses[seed,reference] for seed in seeds],axis=0)
            intervals.append({'model':name,'reference':reference,**cluster_interval(delta,groups)})
    (out/'uncertainty.json').write_text(json.dumps(intervals,indent=2))
    # Eligibility guards: mean MSE, low/medium preservation and interval vs the
    # strongest existing eligible baseline. These are conservative final decisions;
    # all exploratory candidate scores remain available, including rejected ones.
    qualifying=[]
    for name in ['mean_glm_nonnegative','smearing_nonnegative','nested_selector_nonnegative']:
        preserves=True
        for stratum in ['all','physical_low','physical_medium']:
            values=table[table.stratum.eq(stratum)].groupby('model').MSE.mean()
            preserves &= bool(values[name] < values[strongest_baseline])
        interval=next(row for row in intervals if row['model']==name and row['reference']==strongest_baseline)
        if preserves and interval['cluster_bootstrap_95pct'][1] < 0:
            qualifying.append(name)
    if qualifying:
        selected=min(qualifying,key=lambda name:means[name])
        with threadpool_limits(limits=2):
            _,chosen,_,fitted=nested_selection(features,y,groups,subset(features,np.array([0])),seed=2719)
        spec=chosen[selected.removesuffix('_nonnegative')]['spec']
        model=NonnegativeProjection(fitted[spec['name']])
    else:
        selected=strongest_baseline
        spec=next(spec for spec in candidate_specs() if spec['name']==selected.removesuffix('_nonnegative'))
        with threadpool_limits(limits=2):
            model=NonnegativeProjection(fit_spec(spec,features,y,groups,seed=2719))
    # Keep the newly tested model as a separate research candidate even when the
    # conservative final decision retains a pre-existing baseline.
    candidate_source=source/'model.joblib'
    candidate_meta=json.loads((source/'final_selection.json').read_text())
    bundle={'format':'si_tail_evidence_v1','model':model,'spec':spec,
            'features':'SMILES only; canonical tautomer; no external labels or source lookup',
            'nonnegative_projection':True,'upper_prediction_cap':None}
    joblib.dump(bundle,out/'model.joblib',compress=3)
    pd.DataFrame({'source_row':np.arange(len(y))+2,'SMILES':train.SMILES,'SI':y,
                  'source':'organizer','features_used':'SMILES only'}).to_csv(out/'training_data.csv',index=False)
    test=pd.read_csv('data/raw/test.csv',usecols=['index','SMILES'])
    sample=pd.read_csv('data/raw/sample_submit.csv',usecols=['index'])
    test_features,test_keys,_=structures(test.SMILES)
    with threadpool_limits(limits=2):
        prediction=model.predict(test_features)
        repeated=joblib.load(out/'model.joblib')['model'].predict(test_features)
    np.testing.assert_array_equal(prediction,repeated)
    if len(test)!=len(sample) or not test['index'].is_unique or test['index'].tolist()!=sample['index'].tolist():
        raise ValueError('Submission IDs/order invalid')
    if not np.isfinite(prediction).all() or (prediction<0).any():
        raise ValueError('Submission predictions invalid')
    pd.DataFrame({'index':test['index'],'SI VACV':prediction}).to_csv(out/'submission.csv',index=False)
    similarities=tanimoto(test_features['fingerprints'],features['fingerprints'])
    pd.DataFrame({'index':test['index'],'prediction':prediction,'max_train_tanimoto':similarities.max(axis=1),
                  'nearest_training_row':similarities.argmax(axis=1),
                  'parent_seen':np.isin(test_keys,keys)}).to_csv(out/'prediction_audit.csv',index=False)
    report={'recommended_model':selected,'selected_spec':spec,'qualifying_new_models':qualifying,
            'selection_scope':'Conservative post-benchmark decision; baseline selection and CI guards are not a new untouched validation set',
            'reason':'Retain strongest existing eligible baseline if improvement or preservation of low/medium SI is not confirmed',
            'mean_seed_MSE':float(means[selected]),'strongest_baseline':strongest_baseline,
            'new_candidate_model':str(candidate_source),'new_candidate_policy':candidate_meta['selected_policy'],
            'new_candidate_spec':candidate_meta['selected_spec'],
            'submission_rows':len(test),'submission_min':float(prediction.min()),'submission_max':float(prediction.max()),
            'upper_prediction_cap':None,'nonnegative_projection':True,
            'nonnegative_projection_rationale':'For y>=0, replacing a negative prediction by zero cannot increase squared error; the threshold is fixed by support, not tuned',
            'external_training_rows':0,'public_score':None,'exact_model_reload':True}
    existing_path=Path('submission_candidate_v3.csv')
    if existing_path.exists():
        existing=pd.read_csv(existing_path,usecols=['index','SI VACV'])
        if existing['index'].tolist()==test['index'].tolist():
            report['max_abs_difference_vs_existing_v3_submission']=float(np.max(np.abs(prediction-existing['SI VACV'])))
    (out/'selection_and_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    print(json.dumps(intervals,indent=2),flush=True)


if __name__=='__main__':
    main()
