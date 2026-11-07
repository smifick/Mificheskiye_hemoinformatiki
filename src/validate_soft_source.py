"""Integration checks: outer-label poisoning, portable I/O, frozen artifacts."""
import argparse,copy,json,shutil,subprocess,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import soft_source as S
import run_soft_source as R

def validate(root,cache,out,encoder_cache):
    frozen=json.loads((out/'baseline_frozen.json').read_text())
    before={p:S.sha(root/p) for p in frozen['file_sha256']};assert before==frozen['file_sha256']
    state,emb,smiles=S.prepare(cache);ids=np.arange(len(state['y']))
    original=pd.read_csv(out/'primary_outer_components.csv.gz')
    positive=original.loc[original.selected_alpha>0,'outer_fold']
    checked_fold=int(positive.min()) if len(positive) else 0
    tr,va=R.folds(state,ids,'scaffold',2718,archived=True)[checked_fold]
    original=original[original.outer_fold==checked_fold].sort_values('source_row')
    original_alpha=float(original.selected_alpha.iloc[0]);errors={};alpha_checks={}
    with tempfile.TemporaryDirectory(dir=root/'work') as temp:
        temp=Path(temp)
        for name,value in [('NaN',np.nan),('large1e12',1e12)]:
            changed=state.copy();changed['world']=copy.copy(state['world'])
            changed['y']=state['y'].copy();changed['world'].y=state['world'].y.copy()
            changed['y'][va]=value;changed['world'].y[va]=value
            changed['train']=state['train'].copy();changed['world'].train=state['world'].train.copy()
            changed['train'].loc[va,'SI VACV']=value;changed['world'].train.loc[va,'SI VACV']=value
            # An empty cache per case forces four genuine model-copy refits.
            case=temp/name;case.mkdir();shutil.copy2(cache/'embedding_manifest.json',case/'embedding_manifest.json')
            shutil.copy2(cache/'mlm_raw.npy',case/'mlm_raw.npy')
            outer,inner,choices=R.nested(changed,emb,smiles,[(tr,va)],case,{},'validation_poison_'+name)
            outer=outer.sort_values('source_row');alpha_checks[name]=choices[0]['alpha']
            assert alpha_checks[name]==original_alpha
            errors[name]=float(np.max(np.abs(outer.prediction.to_numpy()-original.prediction.to_numpy())));assert errors[name]<1e-8
            assert set(inner.source_row).isdisjoint(va+2)
        cli=check_cli(root,out,temp,encoder_cache)
        positive_cli=check_positive_cli(root,out,temp,encoder_cache)
    assert {p:S.sha(root/p) for p in before}==before
    checks=dict(status='PASS',outer_label_poisoning_refit=True,outer_label_poisoning_max_prediction_error=errors,
                outer_label_poisoning_selected_alpha=alpha_checks,original_outer_alpha=original_alpha,checked_outer_fold=checked_fold,
                portable_CLI_max_error=cli,positive_alpha_CLI=positive_cli,IDs_preserved=True,forbidden_columns_ignored=True,
                production_and_baseline_file_hashes_unchanged=True,files_checked=before,
                frozen_feature_input_sha256=json.loads((cache/'embedding_manifest.json').read_text())['input_sha256'])
    (out/'validation.json').write_text(json.dumps(checks,indent=2))
    alpha=json.loads((out/'alpha.json').read_text());alpha['validation_passed']=True
    if alpha['statistical_acceptance_passed']:
        alpha['recommended_alpha']=alpha['calibration_selected_alpha']
        pending=out/'candidate_pending_validation.csv'
        if pending.exists():pending.replace(out/'submission_soft_source.csv')
    summary=json.loads((out/'summary.json').read_text());summary['recommended_alpha']=alpha['recommended_alpha']
    summary['validation_passed']=True;(out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'alpha.json').write_text(json.dumps(alpha,indent=2))
    concentration_diagnostic(out)
    print(json.dumps(checks,indent=2))

def check_cli(root,out,temp,encoder_cache):
    test=pd.read_csv(root/'data/raw/test.csv',usecols=['index','SMILES'])
    reference=pd.read_csv(root/'results/audit_20261107/submission.csv').sort_values('index')
    poison=test.assign(CC50_uM=1e99,IC50_uM_VACV=np.nan,**{'SI VACV':1e12},Culture='ignored',strain='ignored',Title='ignored').iloc[::-1]
    cli={}
    for name,frame in [('clean',test),('forbidden_poison_reversed',poison)]:
        inp=temp/(name+'.csv');target=temp/(name+'_prediction.csv');frame.to_csv(inp,index=False)
        subprocess.run([sys.executable,str(root/'src/predict_soft_source.py'),'--model',str(root/'results/audit_20261107/selected_model.joblib'),
                        '--alpha','0','--input',str(inp),'--out',str(target),'--cache',str(encoder_cache)],check=True)
        actual=pd.read_csv(target).sort_values('index')
        assert actual['index'].tolist()==reference['index'].tolist()
        cli[name]=float(np.max(np.abs(actual['SI VACV'].to_numpy()-reference['SI VACV'].to_numpy())));assert cli[name]<1e-7
    return cli

def concentration_diagnostic(out):
    primary=pd.read_csv(out/'primary_outer_components.csv.gz');y=primary.observed.to_numpy()
    baseline=S.apply_alpha(primary,0);fixed=S.apply_alpha(primary,.1)
    contribution=(y-baseline)**2-(y-fixed)**2;j=int(np.argmax(contribution))
    diagnostic=dict(alpha=.1,status='fixed-alpha diagnostic, not nested selection',total_MSE_reduction=float(contribution.sum()/len(y)),
                    largest_positive_source_row=int(primary.iloc[j].source_row),largest_positive_MSE_contribution=float(contribution[j]/len(y)),
                    share_of_net_gain_pct=float(100*contribution[j]/contribution.sum()),
                    MSE_reduction_without_largest=float((contribution.sum()-contribution[j])/len(y)))
    (out/'fixed_alpha_gain_concentration.json').write_text(json.dumps(diagnostic,indent=2))

def check_positive_cli(root,out,temp,encoder_cache):
    components=pd.read_csv(out/'production_components.csv.gz')
    source=components.source_present;lookup=components.lookup_applied
    idx=np.r_[np.flatnonzero(source&~lookup)[:3],np.flatnonzero(source&lookup)[:2],np.flatnonzero(~source)[:3]]
    subset=components.iloc[idx];clean=subset[['index','SMILES']]
    expected=pd.Series(S.apply_alpha(subset,.1),index=subset['index'])
    errors={}
    for name,frame in [('clean',clean),('poisoned',clean.assign(CC50_uM=1e99,IC50_uM_VACV=np.nan,Title='ignored',
                              Culture='ignored',strain='ignored',**{'SI VACV':1e12}).iloc[::-1])]:
        inp=temp/('positive_'+name+'.csv');target=temp/('positive_'+name+'_pred.csv');frame.to_csv(inp,index=False)
        subprocess.run([sys.executable,str(root/'src/predict_soft_source.py'),'--model',str(root/'results/audit_20261107/selected_model.joblib'),
                        '--alpha','.1','--input',str(inp),'--out',str(target),'--cache',str(encoder_cache)],check=True)
        actual=pd.read_csv(target).set_index('index')['SI VACV'];actual=actual.loc[expected.index]
        errors[name]=float(np.max(np.abs(actual.to_numpy()-expected.to_numpy())));assert errors[name]<1e-7
    return dict(alpha=.1,rows=len(subset),source_present_without_lookup=3,lookup=2,source_absent=3,max_abs_error=errors)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',default='work/soft_source_cache');p.add_argument('--out',default='results/soft_source_20261107')
    p.add_argument('--encoder-cache',default='work/chemberta_snapshots');args=p.parse_args();root=Path(__file__).resolve().parents[1]
    with threadpool_limits(limits=2):validate(root,root/args.cache,root/args.out,root/args.encoder_cache)
