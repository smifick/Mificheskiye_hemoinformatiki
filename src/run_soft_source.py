"""Nested validation of ONE alpha in fixed PR3 source routing; no other change."""
import argparse,hashlib,json,shutil
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import audit_regression as A
import research_heads as H
import soft_source as S
from run_targeted_research import folds

def mse(y,p):return float(np.mean((np.asarray(y)-p)**2))

def frame_save(table,path):
    table.to_csv(path,index=False,compression={'method':'gzip','mtime':0} if str(path).endswith('.gz') else None)

def make_fold(state,emb,smiles,tr,va,blocked,cache,registry,force=False):
    tr=np.asarray(tr,int);va=np.asarray(va,int);blocked=np.asarray(blocked,int)
    world=state['world']
    assert set(state['groups'][tr]).isdisjoint(state['groups'][va])
    assert set(world.tau[tr]).isdisjoint(world.tau[va])
    namespace=hashlib.sha256(world.K.tobytes()+world.ext_si.tobytes()+state['flag'].tobytes()).hexdigest()
    key=hashlib.sha256(json.dumps([tr.tolist(),va.tolist(),blocked.tolist(),namespace,S.SPEC,A.V6.CFG,
                                  S.sha(cache/'embedding_manifest.json'),S.sha(cache/'mlm_raw.npy')]).encode()+world.y[tr].tobytes()).hexdigest()
    filename=cache/('fold_'+key+'.csv.gz');model_path=cache/('fold_'+key+'.joblib')
    provenance_path=cache/('fold_'+key+'.json')
    if filename.exists() and not force:
        table=pd.read_csv(filename);registry[key]=json.loads(provenance_path.read_text());return table
    head=H.fit_head(S.SPEC,state,emb,tr)
    assert all(not state['flag'][child['fit']].any() for child in head['components'])
    r=H.physical(state,tr,va,blocked=blocked)
    table=S.components(head,emb,va,state['flag'][va],r,float(np.mean(state['y'][tr]==0)),
                       world.tau[va],world.scf[va],smiles[va])
    table.insert(0,'source_row',va+2);table['observed']=state['y'][va];table['model_id']=key
    # Published lookup membership is diagnostic only in CV, NEVER applied.
    lookup=A.T.lookup_table(world)
    candidate=np.array([lookup.get(k,np.nan) for k in world.tau[va]],float)
    table['frozen_lookup_candidate']=np.isfinite(candidate);table['frozen_lookup_value']=candidate
    old=H.old_probabilities(state,tr,va)
    table['fixed_unused_legacy_probability']=old[:,0];table['fixed_unused_feature_probability']=old[:,1]
    original=H.complete_predictions(state,tr,va,r,H.predict_head(head,state,emb,va,tr))
    np.testing.assert_allclose(S.apply_alpha(table,0),original,rtol=1e-14,atol=1e-10)
    ext=A.T.ext_allowed(world,va)
    if len(blocked):ext &= A.T.ext_allowed(world,blocked)
    used=world.ext_ids[ext]
    for denied in [va,blocked]:
        assert set(world.tau[used]).isdisjoint(world.tau[denied])
        assert set(world.scf[used][world.scf[used]!='']).isdisjoint(world.scf[denied][world.scf[denied]!=''])
    joblib.dump(head,model_path,compress=3)
    provenance=dict(training_source_rows=(tr+2).tolist(),validation_source_rows=(va+2).tolist(),
                    blocked_source_rows=(blocked+2).tolist(),head_weights_sha256=S.sha(model_path),
                    molecular_training_source_rows=[(child['fit']+2).tolist() for child in head['components']],
                    physical_config=A.V6.CFG,external_training_rows=int(ext.sum()),
                    external_allowed_mask_sha256=hashlib.sha256(ext.tobytes()).hexdigest(),
                    p_group_parent=head['p_flag'],p_group_leaves=[child['p_flag'] for child in head['components']],
                    zero_rate=float(np.mean(state['y'][tr]==0)),source_flagged_training=int(state['flag'][tr].sum()),
                    flagged_codes_training=int(((state['y'][tr]==10000)&state['flag'][tr]).sum()),
                    exact_lookup_policy='disabled for heldout CV, as in original PR3',
                    label_provenance='only training source rows; outer/inner validation labels excluded')
    registry[key]=provenance;provenance_path.write_text(json.dumps(provenance,indent=2))
    frame_save(table,filename);return table

def nested(state,emb,smiles,outer,cache,registry,scheme):
    inner_tables=[];outer_tables=[];choices=[]
    for fold,(tr,va) in enumerate(outer):
        tables=[]
        for j,(it,iv) in enumerate(GroupKFold(3,shuffle=True,random_state=314).split(tr,groups=state['groups'][tr])):
            t=make_fold(state,emb,smiles,tr[it],tr[iv],va,cache,registry)
            t['outer_fold']=fold;t['inner_fold']=j;t['scheme']=scheme;t['role']='inner'
            tables.append(t);inner_tables.append(t)
        inner=pd.concat(tables,ignore_index=True)
        alpha,selection=S.choose_alpha(inner)
        table=make_fold(state,emb,smiles,tr,va,[],cache,registry)
        table['outer_fold']=fold;table['inner_fold']=-1;table['scheme']=scheme;table['role']='outer'
        table['selected_alpha']=alpha;table['baseline']=S.apply_alpha(table,0);table['prediction']=S.apply_alpha(table,alpha)
        table['diagnostic_alpha1']=S.apply_alpha(table,1)
        for a in S.GRID:table['fixed_alpha_'+str(a)]=S.apply_alpha(table,a)
        outer_tables.append(table)
        choices.append(dict(scheme=scheme,fold=fold,alpha=alpha,inner_selection=selection,
                            outer_n=len(va),baseline_MSE=mse(table.observed,table.baseline.to_numpy()),
                            selected_MSE=mse(table.observed,table.prediction.to_numpy()),effects=S.effect_counts(table)))
        print('FOLD',scheme,fold,'alpha',alpha,'baseline',choices[-1]['baseline_MSE'],'selected',choices[-1]['selected_MSE'],flush=True)
    return pd.concat(outer_tables,ignore_index=True),pd.concat(inner_tables,ignore_index=True),choices

def freeze_baseline(root,out,bundle):
    base=bundle['baseline_portable_model'];head=bundle['head']
    files=['results/audit_20261107/selected_model.joblib','results/v7_final/model.joblib',
           'results/audit_20261107/submission.csv','data/raw/train.csv','data/raw/test.csv',
           'data/external/external_endpoints.csv','data/external/external_si_database.csv',
           'data/external_curated/vacv_si.csv','src/research_heads.py','src/predict_research.py',
           'src/train_v5.py','src/train_v6.py','src/train_v7.py','src/extract_chemberta.py']
    def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    weights=[]
    for child in head['components']:
        weights.append(dict(spec=child['spec'],training_source_rows=(child['fit']+2).tolist(),p_group=child['p_flag'],
                            coefficient_sha256=array_sha(child['estimator'].coef_),intercept_sha256=array_sha(child['estimator'].intercept_),
                            scaler_mean_sha256=array_sha(child['scaler'].mean_),scaler_scale_sha256=array_sha(child['scaler'].scale_)))
    record=dict(commit=S.COMMIT,selected=bundle['selected'],file_sha256={p:S.sha(root/p) for p in files},
                heads=weights,parent_p_group=head['p_flag'],group_formula='(flagged_training_codes + 0.5)/(flagged_training_count + 2)',
                zero_rate=base['zero_rate'],physical_config=A.V6.CFG,physical_mean=base['B_mean'],
                physical_smear=base['B_smear'],physical_dual_sha256=array_sha(base['B_dual']),
                ensemble=dict(notation=.5,mlm=.5,legacy_v7=0.),encoder=dict(model='DeepChem/ChemBERTa-77M-MLM',
                    revision='ed8a5374f2024ec8da53760af91a33fb8f6a15ff',batch=1,torch_threads=3,pooling='mean+CLS'),
                source_presence_sha256=hashlib.sha256('\n'.join(sorted(base['source_presence'])).encode()).hexdigest(),
                lookup_sha256=hashlib.sha256(json.dumps(base['lookup'],sort_keys=True).encode()).hexdigest(),
                numeric_order='leaf soft routing once; average50:50; F clipping0..10000; exact lookup clipped0..10000',
                source_present_excluded_from_molecular_head_training=True,
                intervention='consolidate identical leaf+parent idempotent overrides; no repeated alpha application',
                old_failed_experiment='gate_followup retrained MLM on all rows; not frozen alpha1 ablation')
    (out/'baseline_frozen.json').write_text(json.dumps(record,indent=2));return record

def report_metrics(table,name):
    y=table.observed.to_numpy();base=table.baseline.to_numpy();pred=table.prediction.to_numpy()
    masks={'all320':np.ones(len(y),bool),'technical10000':y==10000,'technical0':y==0,
           'physical237':(y>0)&(y!=10000),'small_SI_le100':y<=100,'physical_tail_gt1000':(y>1000)&(y!=10000),
           'source_present':table.source_present.to_numpy(bool),'actually_changed':np.abs(base-pred)>1e-9}
    rows=[]
    for label,mask in masks.items():
        if mask.sum():rows.append(dict(scheme=name,stratum=label,n=int(mask.sum()),baseline_MSE=mse(y[mask],base[mask]),
                                      selected_MSE=mse(y[mask],pred[mask])))
    return rows

def finalize(root,out,tables,production,choices,metrics,oofs,fixed,registry,final_alpha,calibration_details,reproduction,frozen):
    primary=tables['primary'];n=len(primary);y=primary.observed.to_numpy();base=primary.baseline.to_numpy();pred=primary.prediction.to_numpy()
    contribution=(y-base)**2-(y-pred)**2;order=np.argsort(-contribution)
    changes=primary[['source_row','observed','source_present','structure_group','scaffold_group','selected_alpha','baseline','prediction']].copy()
    changes['SSE_reduction']=contribution;changes['MSE_reduction_contribution']=contribution/n
    changes.sort_values('SSE_reduction',ascending=False).to_csv(out/'row_error_changes.csv',index=False)
    approved=mse(y,pred)<mse(y,base) and float(contribution.sum()-max(contribution.max(),0))>0 and final_alpha>0
    for name in ['confirmation5511','confirmation5512']:
        t=tables[name];approved &= mse(t.observed,t.prediction.to_numpy())<mse(t.observed,t.baseline.to_numpy())
    approved &= S.effect_counts(production)['effective_rows']>0
    # Positive candidates remain pending until mandatory validation passes.
    recommended_alpha=0.
    alpha_record=dict(calibration_selected_alpha=final_alpha,recommended_alpha=recommended_alpha,
                      calibration_selection=calibration_details,statistical_acceptance_passed=bool(approved),
                      validation_required_before_use=True,validation_passed=False,production_models_retrained=False)
    (out/'alpha.json').write_text(json.dumps(alpha_record,indent=2))
    if approved:
        pd.DataFrame({'index':production['index'],'SI VACV':S.apply_alpha(production,final_alpha)}).to_csv(out/'candidate_pending_validation.csv',index=False)
    else:shutil.copy2(root/'results/audit_20261107/submission.csv',out/'recommended_submission_unchanged_PR3.csv')
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(fixed).to_csv(out/'fixed_alpha_diagnostics.csv',index=False)
    pd.concat(oofs,ignore_index=True).to_csv(out/'nested_oof.csv',index=False)
    (out/'fold_choices.json').write_text(json.dumps(choices,indent=2));(out/'fold_model_provenance.json').write_text(json.dumps(registry,indent=2))
    cluster=pd.DataFrame({'group':primary.scaffold_group,'gain':contribution}).groupby('group').agg(gain=('gain','sum'),n=('gain','size'))
    rng=np.random.default_rng(7519);draw=rng.integers(0,len(cluster),(2000,len(cluster)))
    boot=cluster.gain.to_numpy()[draw].sum(1)/cluster.n.to_numpy()[draw].sum(1)
    summary=dict(baseline_MSE=mse(y,base),nested_MSE=mse(y,pred),improvement_pct=100*(1-mse(y,pred)/mse(y,base)),
                 production_alpha0_max_error=reproduction,production_effects=S.effect_counts(production),
                 primary_effects_alpha02=S.effect_counts(primary),primary_actually_changed=int((np.abs(base-pred)>1e-9).sum()),
                 primary_actually_changed_structures=int(primary.loc[np.abs(base-pred)>1e-9,'structure_group'].nunique()),
                 primary_actually_changed_scaffolds=int(primary.loc[np.abs(base-pred)>1e-9,'scaffold_group'].nunique()),
                 chosen_outer_alphas=[c['alpha'] for c in choices if c['scheme']=='primary'],
                 total_SSE_reduction=float(contribution.sum()),largest_positive_row_source_row=int(primary.iloc[order[0]].source_row),
                 largest_positive_row_SSE_reduction=float(contribution[order[0]]),
                 SSE_reduction_without_largest_positive=float(contribution.sum()-max(contribution.max(),0)),
                 scaffold_bootstrap_MSE_reduction_CI95=np.quantile(boot,[.025,.975]).tolist(),
                 statistical_acceptance_passed=bool(approved),recommended_alpha=recommended_alpha,
                 final_calibration_alpha=final_alpha,known_PR3_Public_MSE=6327610.34997,new_Public_MSE=None,
                 validation_mode='B nested procedure, fixed PR3 settings; production untouched',independent_data=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    assert all(S.sha(root/p)==digest for p,digest in frozen['file_sha256'].items())
    print('SUMMARY',json.dumps(summary),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',default='work/soft_source_cache')
    parser.add_argument('--out',default='results/soft_source_20261107');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];cache=root/args.cache;out=root/args.out
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(commit=S.COMMIT,small_alpha_grid=list(S.GRID),diagnostic_full_removal_alpha=1.,
                  outer_primary='archived5 scaffold folds',inner='3fold scaffold GroupKFold shuffle seed314',
                  confirmation_scaffold_seeds=[5511,5512],validation_mode='B nested procedure, fixed PR3 settings; production untouched',
                  no_independent_data=True,source_membership='frozen public database mapped from SMILES',
                  CV_lookup='disabled as in PR3; external validation molecules/scaffolds excluded from physical training',
                  production_lookup='original frozen exact lookup, applied last',
                  information_rule='at least2 affected distinct structures and scaffolds else alpha0',
                  acceptance='primary nested MSE improves; both confirmation MSE improve; primary gain survives removal of its largest positive-contribution row; final calibrated alpha>0; test has effective rows; all validations pass',
                  selection='unweighted MSE on ALL official inner OOF rows; ties lower alpha',
                  final_alpha_selection='fresh3fold scaffold OOF of all320 using same inner seed314, not leaderboard',
                  fixed='model weights per fitted copy, settings, features, physical model, smoothing, zero, lookup, ensemble and clipping',
                  intervention='one shared alpha; per-leaf preoverride probabilities; redundant identical parent override consolidated')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    state,emb,smiles=S.prepare(cache);n=len(state['y']);ids=np.arange(n);registry={}
    bundle=joblib.load(root/'results/audit_20261107/selected_model.joblib');frozen=freeze_baseline(root,out,bundle)
    production=S.production_components(bundle,smiles[n:],{k:v[n:] for k,v in emb.items()})
    production.insert(0,'index',state['test']['index'].to_numpy());production['model_id']='production_PR3_'+frozen['file_sha256']['results/audit_20261107/selected_model.joblib']
    for a in (*S.GRID,1.):production['prediction_alpha_'+str(a)]=S.apply_alpha(production,a)
    reference=pd.read_csv(root/'results/audit_20261107/submission.csv')
    reproduction=float(np.max(np.abs(reference['SI VACV']-S.apply_alpha(production,0))))
    if reproduction>1e-7:raise AssertionError('Production PR3 does not reproduce')
    frame_save(production,out/'production_components.csv.gz')
    print('PRODUCTION_REPRODUCED',reproduction,S.effect_counts(production),flush=True)
    all_choices=[];metrics=[];oofs=[];fixed=[];tables={}
    for scheme,seed in [('primary',None),('confirmation5511',5511),('confirmation5512',5512)]:
        outer=folds(state,ids,'scaffold',2718 if seed is None else seed,archived=seed is None)
        outer_table,inner_table,choices=nested(state,emb,smiles,outer,cache,registry,scheme)
        if scheme=='primary':
            saved=pd.read_csv(root/'results/audit_20261107/paired_new_model_errors.csv').sort_values('source_row')
            np.testing.assert_allclose(outer_table.sort_values('source_row').baseline,saved.new_prediction,rtol=1e-13,atol=1e-8)
        frame_save(outer_table,out/(scheme+'_outer_components.csv.gz'))
        frame_save(inner_table,out/(scheme+'_inner_components.csv.gz'))
        tables[scheme]=outer_table;all_choices.extend(choices);metrics+=report_metrics(outer_table,scheme)
        oofs.append(outer_table[['scheme','source_row','observed','outer_fold','selected_alpha','baseline','prediction','diagnostic_alpha1','structure_group','scaffold_group']])
        for a in (*S.GRID,1.):fixed.append(dict(scheme=scheme,alpha=a,diagnostic_only=True,MSE=mse(outer_table.observed,S.apply_alpha(outer_table,a))))
    calibration=[]
    for j,(tr,va) in enumerate(folds(state,ids,'scaffold',314,n=3)):
        t=make_fold(state,emb,smiles,tr,va,[],cache,registry);t['calibration_fold']=j;calibration.append(t)
    calibration=pd.concat(calibration,ignore_index=True);final_alpha,calibration_details=S.choose_alpha(calibration)
    frame_save(calibration,out/'final_calibration_components.csv.gz')
    finalize(root,out,tables,production,all_choices,metrics,oofs,fixed,registry,final_alpha,calibration_details,reproduction,frozen)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
