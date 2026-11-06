"""Audit the actual SI project and reproduce comparable OOF predictions.

Concentrations/assay metadata are read only in the explicitly diagnostic audit.
All fitted estimators receive SMILES-derived arrays and training responses only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import solve
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold,StratifiedGroupKFold

import train_model as V1
import train_v2 as V2
import train_external_v3 as V3
import train_v5 as T
import train_v6 as V6
import train_v7 as V7


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def loss(y,p):return float(np.mean((y-p)**2))


def empirical_floor(keys,y):
    table=pd.DataFrame({'key':list(keys),'y':y,'source_row':np.arange(len(y))+2})
    table['group_mean']=table.groupby('key').y.transform('mean')
    table['SSE']=(table.y-table.group_mean)**2
    table['SSE_output_cap_10000']=(table.y-np.clip(table.group_mean,0,10000))**2
    groups=table.groupby('key',sort=False).agg(n=('y','size'),n_labels=('y','nunique'),SSE=('SSE','sum'))
    conflicts=groups[groups.n_labels>1]
    return {'n_unique':len(groups),'conflicting_groups':len(conflicts),
            'conflicting_rows':int(conflicts.n.sum()),'minimum_empirical_MSE':float(table.SSE.sum()/len(y)),
            'minimum_empirical_MSE_with_output_cap_10000':float(table.SSE_output_cap_10000.sum()/len(y))},table


def row_hashes(a):
    return [hashlib.sha256(np.ascontiguousarray(row,dtype=np.float32).tobytes()).hexdigest() for row in a]


def make_registry(root,out):
    records=[]
    def add(version,code,model_path,prediction,features,transform,validation,local_mse,public=None,evidence='unknown',ensemble='none'):
        p=root/prediction if prediction else None
        records.append(dict(version=version,code=code,model=model_path,prediction=prediction,
             prediction_sha256=sha(p) if p and p.exists() else '',features=features,transform=transform,
             validation=validation,local_MSE=local_mse,public_MSE=public,public_evidence=evidence,ensemble=ensemble))
    add('v1','src/train_model.py','results/final_run/model.joblib','submission.csv',
        'Morgan r2 2048 chirality; Tanimoto KRR alpha=1','raw',
        '52-row previously inspected scaffold holdout',12350204.87679665,9992163.26222,'historical repository record')
    add('v2','src/train_v2.py','results/v2/model_v2.joblib','submission_candidate_v2.csv',
        'SMILES fingerprints; auxiliary states derived from training y','raw + conditional log1p',
        '5 scaffold OOF seed2718; winner used for selection',19041223.058204416,9681096.22474,
        'historical repository record','50% KRR + 50% missing-code mixture')
    add('v3','src/train_external_v3.py','results/external_v3/model.joblib','submission_candidate_v3.csv',
        'tautomer Morgan r2 2048; no external SI selected','raw',
        'same 5 scaffold OOF seed2718',18888578.919285785,9929683.57063,'historical docs/two_component_v4.md record')
    add('v4','src/two_component_v4.py','results/two_component_v4/model_parameters.json','results/two_component_v4/submission_v4.csv',
        'SMILES two-component classifier/regressor + external ablations','conditional log1p calibrated',
        'same scaffold OOF; source-retrospective physical holdout separate',20038362.56)
    add('v5','src/train_v5.py','','','Morgan/torsion kernel + external conditional SI','conditional log1p Duan smearing',
        '8 balanced scaffold repetitions; recovered via supplied v6 metrics',18085334.17363744)
    add('v6','src/train_v6.py','','submission_v6.csv','v5 plus source-presence gate computed from SMILES','conditional log1p Duan smearing',
        '8 balanced scaffold repetitions',17491062.331982985,9649437.79854,'user-reported Kaggle result')
    add('v7','src/train_v7.py','results/v7_final/model.joblib','submission_candidate_v7.csv',
        'v6 and feature-Morgan inactive-code classifier; SMILES only','conditional log1p Duan smearing',
        '8 development-data scaffold repetitions seeds101..108',17121947.963889234,9421022.05828,
        'user screenshot, filename submission_candidate_v7.csv','fixed 50:50 complete-prediction ensemble')
    if (root/'results/v7/confirmation.csv').exists():
        alt=pd.read_csv(root/'results/v7/confirmation.csv')
        metric=float(alt.loc[alt.model.eq('feature'),'MSE'].mean())
        add('v7_full_feature_alternative','src/experiment_v7.py','recreated by source','results/v7/submission_v7.csv',
            'feature-Morgan classifier; not deployed v7 blend','conditional log1p smearing',
            '8 reused scaffold repetitions seeds101..108',metric)
    if (root/'results/kernel_v7/metrics.json').exists():
        m=json.loads((root/'results/kernel_v7/metrics.json').read_text())
        add('kernel_search_negative','src/train_kernel_v7.py','recreated by source','',
            'torsion/count kernel winner; 72-config search','raw',
            'scaffold OOF winner; nested search is worse',m['selected_oof_mse'])
    if (root/'results/smiles_only_modular/metrics.json').exists():
        m=json.loads((root/'results/smiles_only_modular/metrics.json').read_text())
        add('modular_smiles_only','si_pipeline/train.py','results/smiles_only_modular/model.joblib','results/smiles_only_modular/submission.csv',
            'RDKit descriptors + MACCS + Morgan counts; Extra Trees mixture','raw conditional SI',
            'nested scaffold CV; 118 tau-normalized groups and different seed',m['selected_OOF_MSE'])
    if (out/'targeted_results.json').exists():
        result=json.loads((out/'targeted_results.json').read_text())
        primary=next(r for r in result['nested'] if r['scheme']=='scaffold')
        add('audit_20261107_selected','src/run_targeted_research.py',str((out/'selected_model.joblib').relative_to(root)),
            str((out/'submission.csv').relative_to(root)),result['selected_for_submission']+'; SMILES-only technical-code head + legacy physical branch',
            'conditional log1p Duan smearing, final metric raw',
            'same published folds; full nested model/ensemble selection; repeated stability checks',primary['fixed_candidate_MSE'],
            ensemble='fixed 50:50 average of notation and frozen MLM code probabilities; legacy physical branch')
    pd.DataFrame(records).to_csv(out/'solution_registry.csv',index=False)
    (out/'public_score_registry.json').write_text(json.dumps(records,indent=2))
    return records


def make_state(root,out):
    allowed=pd.read_csv(root/'data/raw/train.csv',usecols=['SMILES','SI VACV'])
    test=pd.read_csv(root/'data/raw/test.csv',usecols=['index','SMILES'])
    # Construct a fresh legacy world to check exact ordering, not a stale cache.
    w=T.World();w.train=w.train[['SMILES','SI VACV']].copy();w.test=w.test[['index','SMILES']].copy()
    assert list(w.train.SMILES)==list(allowed.SMILES) and np.array_equal(w.y,allowed['SI VACV'])
    flag=V6.chembl_flags(w)
    smiles=[V1.standardize(s)[0] for s in list(allowed.SMILES)+list(test.SMILES)]
    f=V2.extended_features(smiles);k=V2.kernels(f)
    # Version-specific tau normalization is retained exactly for reproduction.
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    tau=np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(V3.canonical(s,True)[0])) for s in list(allowed.SMILES)+list(test.SMILES)],float)
    ktau=V1.tanimoto(tau,tau)
    fg=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=8192,includeChirality=True,
          atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    pharma=np.array([fg.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles],float)
    kfeature=V1.tanimoto(pharma,pharma)
    state=dict(world=w,flag=flag,y=w.y,train=allowed,test=test,canonical=np.array(smiles,object),
               F=f,K=k,K_tau=ktau,K_feature=kfeature,feature=pharma,
               groups=np.array([V1.standardize(s)[1] for s in allowed.SMILES]))
    joblib.dump(state,out/'state_cache.joblib',compress=3)
    return state


def pcode_feature(state,tr,va):
    w=state['world'];flag=state['flag'];c=(w.y==10000).astype(float)
    fit=tr[~flag[tr]];mu=c[fit].mean();k=state['K_feature']
    dual=solve(k[np.ix_(fit,fit)]+.5*np.eye(len(fit)),c[fit]-mu,assume_a='pos')
    p=np.clip(mu+k[np.ix_(va,fit)]@dual,0,1)
    ff=tr[flag[tr]];pf=(c[ff].sum()+.5)/(len(ff)+2.)
    return np.where(flag[va],pf,p)


def reproduce_fold(state,tr,va):
    y=state['y'];w=state['world'];p={}
    p['v1']=np.maximum(KernelRidge(alpha=1,kernel='precomputed').fit(state['K']['m2'][np.ix_(tr,tr)],y[tr]).predict(state['K']['m2'][np.ix_(va,tr)]),0)
    codes=np.where(y==10000,1,np.where(y==0,2,0))
    spec=dict(name='blend_m2_mixture_krr',family='blend',classifier='krr')
    b=V2.stage_fit(spec,tr,y,codes,state['F'],state['K'])
    p['v2']=V2.stage_predict(b,va,state['F'],state['K'])
    p['v3']=np.maximum(KernelRidge(alpha=1,kernel='precomputed').fit(state['K_tau'][np.ix_(tr,tr)],y[tr]).predict(state['K_tau'][np.ix_(va,tr)]),0)
    em=T.ext_allowed(w,va)
    # Validation labels are not used in the fitted branches or their calibration.
    _,_,r=T.predict(w,tr,va,V6.CFG,ext_mask=em)
    p0=float(np.mean(y[tr]==0))
    pc0=V6.code_prob(w,state['flag'],tr,va,.5)
    pc1=pcode_feature(state,tr,va)
    p['v6']=np.clip(10000*pc0+np.clip(1-pc0-p0,0,1)*r,0,10000)
    other=np.clip(10000*pc1+np.clip(1-pc1-p0,0,1)*r,0,10000)
    p['v7']=.5*(p['v6']+other)
    return p,dict(p_code_v6=pc0,p_code_feature=pc1,r=r,p_zero=np.full(len(va),p0))


def audit_representations(root,out,state):
    data=pd.read_csv(root/'data/raw/train.csv')  # Audit only; never fitted as X.
    y=state['y'];w=state['world'];n=len(y)
    rawcanonical=[Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=True) for s in data.SMILES]
    parent=[Chem.MolToSmiles(T.std(s),isomericSmiles=True) for s in data.SMILES]
    nostereo=[Chem.MolToSmiles(T.std(s),isomericSmiles=False) for s in data.SMILES]
    tt=np.array([T.tt.GetCountFingerprintAsNumPy(T.std(s)) for s in data.SMILES],float)
    representations={'raw_SMILES':data.SMILES,'full_canonical_graph':rawcanonical,'standardized_parent':parent,
        'tautomer_parent':w.tau[:n],'parent_without_stereo':nostereo,
        'Morgan2048_bits':row_hashes(state['F']['m2'][:n]),
        'feature_Morgan8192_bits':row_hashes(state['feature'][:n]),
        'v7_core_combined_fingerprints_and_source_flag':row_hashes(np.column_stack([state['F']['m2'][:n],tt,state['feature'][:n],state['flag'][:n]]))}
    floors=[];conflicts=[]
    for name,keys in representations.items():
        summary,table=empirical_floor(keys,y);floors.append(dict(representation=name,**summary))
        for key,d in table.groupby('key',sort=False):
            if d.y.nunique()<2:continue
            idx=d.source_row.to_numpy(int)-2
            conflicts.append(dict(representation=name,key=str(key),n=len(d),source_rows=json.dumps(d.source_row.tolist()),
                labels=json.dumps(d.y.tolist()),SSE=float(d.SSE.sum()),
                different_full_graphs=len(set(np.array(rawcanonical)[idx])),
                different_contexts=len(set(zip(data.iloc[idx].strain,data.iloc[idx].Culture))),
                original_SMILES=json.dumps(data.iloc[idx].SMILES.tolist()),
                context_for_diagnosis_only=json.dumps(list(zip(data.iloc[idx].strain,data.iloc[idx].Culture)))))
    pd.DataFrame(floors).to_csv(out/'representation_floors.csv',index=False)
    pd.DataFrame(conflicts).to_csv(out/'conflicting_inputs.csv',index=False)
    ratio=data.CC50_uM/data.IC50_uM_VACV;known=ratio.notna()
    summary={'train_rows':len(data),'test_rows':len(state['test']),'codes_10000':int((y==10000).sum()),
        'codes_10000_and_missing_IC50':int(((y==10000)&data.IC50_uM_VACV.isna()).sum()),
        'zero_labels':int((y==0).sum()),'zero_labels_and_missing_CC50':int(((y==0)&data.CC50_uM.isna()).sum()),
        'observed_ratio_rows':int(known.sum()),'ratio_agrees_rtol_2pct':int(np.isclose(y[known],ratio[known],rtol=.02).sum()),
        'ratio_max_absolute_difference':float(np.max(abs(y[known]-ratio[known]))),
        'concentrations_loaded_for_model_features':False,
        'note':'Empirical floors apply only to the known labels and the stated representation, not the hidden Kaggle test.'}
    (out/'target_audit.json').write_text(json.dumps(summary,indent=2))
    return data


def audit_submissions(root,out):
    sample=pd.read_csv(root/'data/raw/sample_submit.csv')
    paths=sorted(p for p in root.rglob('submission*.csv') if out not in p.parents)
    rows=[];preds={}
    for path in paths:
        d=pd.read_csv(path);name=str(path.relative_to(root))
        good=list(d)==['index','SI VACV'] and len(d)==len(sample) and d['index'].tolist()==sample['index'].tolist()
        finite=bool(np.isfinite(d['SI VACV']).all()) if 'SI VACV' in d else False
        rows.append(dict(path=name,sha256=sha(path),rows=len(d),schema_and_ID_order_match=good,finite=finite,
             min=float(d['SI VACV'].min()),mean=float(d['SI VACV'].mean()),median=float(d['SI VACV'].median()),max=float(d['SI VACV'].max())))
        if good and finite:preds[name]=d['SI VACV'].to_numpy()
    pd.DataFrame(rows).to_csv(out/'submission_audit.csv',index=False)
    frame=pd.DataFrame(preds)
    frame.corr().to_csv(out/'submission_correlations.csv')
    rmse=pd.DataFrame(index=frame.columns,columns=frame.columns,dtype=float)
    for a in frame:
        for b in frame:rmse.loc[a,b]=float(np.sqrt(np.mean((frame[a]-frame[b])**2)))
    rmse.to_csv(out/'submission_pairwise_prediction_RMSE.csv')
    # Disagreement is not test error: no hidden target is available here.
    if len(frame.columns)>1:
        spread=frame.std(axis=1)
        table=state_test=pd.read_csv(root/'data/raw/test.csv',usecols=['index','SMILES'])
        table=table.assign(prediction_std=spread.to_numpy(),prediction_min=frame.min(axis=1).to_numpy(),prediction_max=frame.max(axis=1).to_numpy())
        table.sort_values('prediction_std',ascending=False).to_csv(out/'submission_disagreements.csv',index=False)
    return rows


def technical_checks(root,out,state):
    n=len(state['y']);test=state['test'];ids=np.arange(n,n+len(test));allids=np.arange(n)
    checks=[]
    b=joblib.load(root/'results/final_run/model.joblib')
    order_error=float(np.max(abs(b['train_fingerprints']-state['F']['m2'][:n])))
    p=np.maximum(b['model'].predict(state['K']['m2'][np.ix_(ids,allids)]),0)
    expected=pd.read_csv(root/'submission.csv')['SI VACV'].to_numpy()
    checks.append(dict(check='v1_fingerprint_training_column_order',max_abs_error=order_error,passed=order_error<1e-12))
    err=float(np.max(abs(p-expected)))
    checks.append(dict(check='v1_saved_model_reproduction',max_abs_error=err,passed=err<1e-6))
    b2=joblib.load(root/'results/v2/model_v2.joblib')
    p2=V2.stage_predict(b2['bundle'],np.arange(len(test)),{key:value[n:] for key,value in state['F'].items()},V2.kernels({key:value[n:] for key,value in state['F'].items()},b2['features']))
    err=float(np.max(abs(p2-pd.read_csv(root/'submission_candidate_v2.csv')['SI VACV'].to_numpy())))
    checks.append(dict(check='v2_saved_model_reproduction',max_abs_error=err,passed=err<1e-6))
    b3=joblib.load(root/'results/external_v3/model.joblib')
    p3=np.maximum(b3['model'].predict(state['K_tau'][np.ix_(ids,allids)]),0)
    err=float(np.max(abs(p3-pd.read_csv(root/'submission_candidate_v3.csv')['SI VACV'].to_numpy())))
    checks.append(dict(check='v3_saved_model_reproduction',max_abs_error=err,passed=err<1e-6))
    portable=joblib.load(root/'results/v7_final/model.joblib')
    p7=V7.infer(portable,test.SMILES)
    err=float(np.max(abs(p7-pd.read_csv(root/'submission_candidate_v7.csv')['SI VACV'].to_numpy())))
    checks.append(dict(check='v7_saved_model_reproduction',max_abs_error=err,passed=err<1e-6))
    # Test graph-only feature invariance to a shuffled row order, then map back.
    perm=np.random.default_rng(20261107).permutation(len(test))[:12]
    psmall=V7.infer(portable,test.SMILES.iloc[perm])
    err=float(np.max(abs(psmall-p7[perm])))
    checks.append(dict(check='v7_shuffled_input_preserves_row_mapping',max_abs_error=err,passed=err<1e-6))
    table=pd.DataFrame(checks);table.to_csv(out/'technical_checks.csv',index=False)
    print('TECHNICAL',table.to_dict('records'),flush=True)
    if not table.passed.all():raise AssertionError('A saved-model audit failed; inspect technical_checks.csv before research')


def oof_audit(state,out):
    n=len(state['y']);y=state['y'];w=state['world'];ids=np.arange(n)
    # Seed alone is insufficient: relabelling groups can change their sorted
    # order and hence shuffled folds. Reuse the actual published assignments.
    root=Path(__file__).resolve().parents[1]
    published=pd.read_csv(root/'results/v2/validation_splits.csv').sort_values('source_row')
    assert published.source_row.tolist()==list(range(2,n+2))
    fold_labels=published.outer_fold.to_numpy(int)
    folds=[(ids[fold_labels!=f],ids[fold_labels==f]) for f in sorted(set(fold_labels))]
    predictions={name:np.zeros(n) for name in ['v1','v2','v3','v6','v7']}
    components={key:np.zeros(n) for key in ['p_code_v6','p_code_feature','r','p_zero']}
    neighbors={};fold_ids=np.zeros(n,int)
    for fold,(tr,va) in enumerate(folds):
        assert set(w.scf[tr]).isdisjoint(w.scf[va])
        assert set(w.tau[tr]).isdisjoint(w.tau[va])
        pp,cc=reproduce_fold(state,tr,va)
        for name,p in pp.items():predictions[name][va]=p
        for name,p in cc.items():components[name][va]=p
        fold_ids[va]=fold
        for i in va:
            near=tr[np.argsort(-state['K']['m2'][i,tr],kind='stable')[:3]]
            neighbors[i]={'max_train_tanimoto':float(state['K']['m2'][i,near[0]]),
                          'nearest_train_source_rows':json.dumps((near+2).tolist()),
                          'nearest_train_labels':json.dumps(y[near].tolist()),
                          'nearest_train_SMILES':json.dumps(state['train'].SMILES.iloc[near].tolist())}
        print('OOF_FOLD',fold,{name:loss(y[va],p[va]) for name,p in predictions.items()},flush=True)
    reference={'v1':19173935.93265446,'v2':19041223.058204416,'v3':18888578.919285785}
    summaries=[]
    for name,p in predictions.items():
        metric=loss(y,p);summaries.append(dict(model=name,MSE=metric,RMSE=float(np.sqrt(metric))))
        if name in reference and abs(metric-reference[name])>1e-4:raise AssertionError(f'Historical OOF differs for {name}: {metric}')
    pd.DataFrame(summaries).to_csv(out/'comparable_baseline_metrics.csv',index=False)
    kind=np.where(y==10000,'technical_10000',np.where(y==0,'technical_zero','observed_ratio'))
    rows=[]
    for name,p in predictions.items():
        for i in ids:rows.append(dict(model=name,source_row=int(i+2),SMILES=state['train'].SMILES.iloc[i],
            observed=float(y[i]),prediction=float(p[i]),squared_error=float((y[i]-p[i])**2),
            label_type=kind[i],fold=int(fold_ids[i]),scaffold=w.scf[i],**neighbors[i]))
    errors=pd.DataFrame(rows)
    errors.to_csv(out/'comparable_oof_errors.csv',index=False)
    errors[errors.model.eq('v7')].sort_values('squared_error',ascending=False).head(20).to_csv(out/'largest_errors_v7.csv',index=False)
    concentration=[]
    for name,d in errors.groupby('model'):
        se=np.sort(d.squared_error.to_numpy())[::-1]
        concentration.append(dict(model=name,**{f'top_{k}_SSE_fraction':float(se[:k].sum()/se.sum()) for k in [1,5,10,20]}))
    pd.DataFrame(concentration).to_csv(out/'error_concentration.csv',index=False)
    bytype=errors.groupby(['model','label_type']).agg(n=('squared_error','size'),SSE=('squared_error','sum'),MSE=('squared_error','mean')).reset_index()
    bytype['SSE_fraction']=bytype.SSE/bytype.groupby('model').SSE.transform('sum')
    bytype.to_csv(out/'errors_by_label_type.csv',index=False)
    errors['similarity_bin']=pd.cut(errors.max_train_tanimoto,[0,.3,.5,.7,.9,1.000001],include_lowest=True)
    errors.groupby(['model','similarity_bin'],observed=True).squared_error.agg(['size','mean','sum']).to_csv(out/'errors_by_similarity.csv')
    def band(v):
        if v==10000:return 'technical_10000'
        if v==0:return 'technical_zero'
        if v<=1:return '(0,1]'
        if v<=10:return '(1,10]'
        if v<=100:return '(10,100]'
        if v<=1000:return '(100,1000]'
        if v<10000:return '(1000,10000)'
        return '>10000_physical'
    errors['target_band']=errors.observed.map(band)
    errors.groupby(['model','target_band']).squared_error.agg(['size','mean','sum']).to_csv(out/'errors_by_target_range.csv')
    # Oracle diagnostics deliberately use validation truth; they are NOT models.
    oracle=10000*(y==10000)+(y!=10000)*(1-components['p_zero'])*components['r']
    oracle=np.clip(oracle,0,10000)
    pmean=.5*(components['p_code_v6']+components['p_code_feature'])
    physical_truth=np.where((y!=10000)&(y!=0),y,0.)
    oracle_b=np.clip(10000*pmean+np.clip(1-pmean-components['p_zero'],0,1)*physical_truth,0,10000)
    report={'baseline_v7_MSE':loss(y,predictions['v7']),
        'oracle_true_technical_code_MSE':loss(y,oracle),'oracle_true_physical_SI_MSE':loss(y,oracle_b),
        'oracle_true_physical_SI_without_upper_cap_MSE':loss(y,10000*pmean+np.clip(1-pmean-components['p_zero'],0,1)*physical_truth),
        'maximum_cross_validated_conditional_physical_prediction':float(components['r'].max()),
        'warning':'Oracles use validation labels solely to diagnose error sources; they cannot be deployed or used as candidate quality.'}
    (out/'oracle_diagnostics.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'baseline_oof.npz',y=y,fold=fold_ids,**predictions,**components)
    print('BASELINES',summaries,'ORACLES',report,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='results/audit_20261107')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];out=root/args.out;out.mkdir(parents=True,exist_ok=True)
    make_registry(root,out)
    audit_submissions(root,out)
    state=make_state(root,out)
    audit_representations(root,out,state)
    technical_checks(root,out,state)
    oof_audit(state,out)
    manifest={'source_commit':'adc5d9bc9ca4de1733d3604f48756c3b71ab5dfc',
        'v7_source_branch_commit':'368c69feb439542254bffa90e28d058aa5879790',
        'Kaggle_data_and_rules_access':'Unavailable via web connector; SMILES-only and concentration prohibition explicitly supplied by user.',
        'input_sha256':{str(p.relative_to(root)):sha(p) for p in [root/'data/raw/train.csv',root/'data/raw/test.csv',root/'data/raw/sample_submit.csv']}}
    (out/'audit_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
