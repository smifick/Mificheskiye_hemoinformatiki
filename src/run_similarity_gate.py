"""Fully nested evaluation of a small S/M gate; PR8 source alpha remains zero.

Gate hyperparameters are scored on inner holdouts whose expert models AND
gate-calibration models exclude that holdout. No meta-CV of a global OOF array.
"""
import argparse,json,hashlib,shutil,gzip,importlib.metadata
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import similarity_experts as E
import similarity_weight as G
from run_targeted_research import folds

FAMILIES=('constant','s','d','sd')

def mse(y,p):return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def save(t,p):t.to_csv(p,index=False,compression={'method':'gzip','mtime':0} if str(p).endswith('.gz') else None)

def select_spec(specs,scores):
    best=min(scores[s['name']] for s in specs)
    tied=[s for s in specs if scores[s['name']]<=best+1e-6]
    rank={'constant':0,'d':1,'s':2,'sd':3}
    return min(tied,key=lambda s:(rank[s['family']],-s['penalty'],s['name']))

def split_rows(state,rows,seed=314,k=3):
    return [(rows[a],rows[b]) for a,b in GroupKFold(k,shuffle=True,random_state=seed).split(rows,groups=state['groups'][rows])]

def crossfit(state,emb,smiles,rows,blocked,cache,registry,seed=314):
    tables=[]
    for j,(tr,va) in enumerate(split_rows(state,rows,seed)):
        table=E.make_fold(state,emb,smiles,tr,va,blocked,cache,registry)
        table['crossfit_fold']=j;tables.append(table)
    return pd.concat(tables,ignore_index=True)

def tune(state,emb,smiles,rows,blocked,cache,registry,scheme,outer_fold):
    specs=G.specs();predictions={s['name']:[] for s in specs};fit_tables=[];evaluation_tables=[]
    for j,(tr,va) in enumerate(split_rows(state,rows)):
        # Extra nesting prevents the gate's calibration predictions from
        # carrying labels of this inner validation through other OOF models.
        calibration=crossfit(state,emb,smiles,tr,np.r_[blocked,va],cache,registry,seed=271)
        calibration['tuning_fold']=j;calibration['outer_fold']=outer_fold;calibration['scheme']=scheme
        evaluation=E.make_fold(state,emb,smiles,tr,va,blocked,cache,registry)
        evaluation['tuning_fold']=j;evaluation['outer_fold']=outer_fold;evaluation['scheme']=scheme
        for spec in specs:
            params=G.fit_gate(calibration,spec)
            p,weights=G.predict_gate(evaluation,params)
            predictions[spec['name']].append(pd.DataFrame(dict(source_row=evaluation.source_row,prediction=p,observed=evaluation.observed)))
        fit_tables.append(calibration);evaluation_tables.append(evaluation)
    scores={name:mse(pd.concat(t).observed,pd.concat(t).prediction) for name,t in predictions.items()}
    # Select regularization separately for each family, and a fully selected
    # procedure across all families, exclusively from the inner predictions.
    by_family={family:select_spec([s for s in specs if s['family']==family],scores) for family in FAMILIES}
    selected=select_spec(specs,scores)
    return pd.concat(evaluation_tables,ignore_index=True),pd.concat(fit_tables,ignore_index=True),by_family,selected,scores

def nested(state,emb,smiles,outer,cache,registry,scheme):
    outer_tables=[];inner_tables=[];calibration_tables=[];choices=[]
    for f,(tr,va) in enumerate(outer):
        gate_train,calibrations,family_specs,selected,scores=tune(state,emb,smiles,tr,va,cache,registry,scheme,f)
        gate_train['role']='inner_validation';calibrations['role']='inner_gate_calibration'
        table=E.make_fold(state,emb,smiles,tr,va,[],cache,registry)
        table['outer_fold']=f;table['scheme']=scheme;table['role']='outer_validation'
        table['production_M']=E.finish(table,table.pred_M.to_numpy())
        table['expert_S']=E.finish(table,table.pred_S.to_numpy())
        table['fixed50']=E.finish(table,.5*(table.pred_S+table.pred_M).to_numpy())
        parameters={}
        for family in FAMILIES:
            p=G.fit_gate(gate_train,family_specs[family]);parameters[family]=p
            prediction,weight=G.predict_gate(table,p)
            table['prediction_'+family]=prediction;table['weight_'+family]=weight
        chosen=G.fit_gate(gate_train,selected)
        table['prediction_selected'],table['weight_selected']=G.predict_gate(table,chosen)
        choices.append(dict(scheme=scheme,fold=f,selected_spec=selected,selected_gate=chosen,family_gates=parameters,
                            family_specs=family_specs,inner_MSE=scores,n=len(va),
                            production_M_MSE=mse(table.observed,table.production_M),
                            constant_MSE=mse(table.observed,table.prediction_constant),selected_MSE=mse(table.observed,table.prediction_selected)))
        outer_tables.append(table);inner_tables.append(gate_train);calibration_tables.append(calibrations)
        print('OUTER',scheme,f,'selected',selected['name'],'M',choices[-1]['production_M_MSE'],
              'constant',choices[-1]['constant_MSE'],'dynamic',choices[-1]['selected_MSE'],flush=True)
    return pd.concat(outer_tables,ignore_index=True),pd.concat(inner_tables,ignore_index=True),pd.concat(calibration_tables,ignore_index=True),choices

def freeze(root,out,bundle):
    files=['results/audit_20261107/selected_model.joblib','results/v7_final/model.joblib',
           'results/audit_20261107/submission.csv','submission_candidate_v7.csv','src/train_v5.py','src/train_v6.py',
           'src/train_v7.py','src/research_heads.py','src/predict_research.py','src/extract_chemberta.py',
           'data/raw/train.csv','data/raw/test.csv','data/external/external_endpoints.csv',
           'data/external/external_si_database.csv','data/external_curated/vacv_si.csv']
    v7=joblib.load(root/'results/v7_final/model.joblib');assert joblib.hash(v7)==joblib.hash(bundle['baseline_portable_model'])
    manifest=dict(commit=E.BASE_COMMIT,PR8_commit='a9b250dc5642bcfdec765e313e602b165e04fdd8',alpha_source=0.,
        file_sha256={p:E.sha(root/p) for p in files},expert_S='deployed v7 completeSI mixture legacy/feature50:50',
        expert_M='PR3 notationMLM probability mixture50:50 then completeSI',
        composition='mix SI AFTER expert clipping, BEFORE shared lookup; wS0 exactly PR3',
        source_smoothing='(flagged_training_codes+0.5)/(flagged_training_count+2)',
        source_absent_only_heads=True,production_reference_rows=len(v7['A_reference']['bits']),
        reference_branch='legacy code classifier A; organizer source-absent, no external or test',
        fingerprint='actual V7.matrices bits, Morgan radius2 bits2048 chirality; T.std preprocessing',
        physical_shared=True,physical_cfg=E.A.V6.CFG,zero_rate=v7['zero_rate'],source_p_group=v7['A_flag_probability'],
        encoder=dict(model='DeepChem/ChemBERTa-77M-MLM',revision='ed8a5374f2024ec8da53760af91a33fb8f6a15ff',batch=1,threads=3),
        lookup='unchanged frozen V/C exact lookup last; disabled for heldout CV exactly as PR3',
        physical_and_head_weights_in_original_bundles=True,prior_constant_control='PR3 notation_mlm_probability_average_blend50')
    manifest['fixed_internal_weights']=dict(S_legacy=.5,S_feature=.5,M_notation=.5,M_MLM=.5)
    manifest['unchanged_limits_and_order']='F uses clip(1-p-zero,0,1), then SI clip0..10000 per expert; common lookup clip0..10000 last'
    (out/'baseline_frozen.json').write_text(json.dumps(manifest,indent=2));return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',default='work/similarity_gate_cache')
    p.add_argument('--out',default='results/similarity_gate_20261107');args=p.parse_args()
    root=Path(__file__).resolve().parents[1];cache=root/args.cache;out=root/args.out;out.mkdir(parents=True,exist_ok=False)
    protocol=dict(baseline_commit=E.BASE_COMMIT,alpha_source=0.,specs=G.specs(),beta_s_bounds=[0,.5],beta_d_bounds=[-.5,.5],
        free_sign_s_control=False,level='FINAL SI after per-expert clipping, before common lookup',
        constant_w0='analytic optimal bounded constant from training OOF',centers='training calibration mean s,d only',
        objective='mean(((finalSI-y)/10000)^2)+lambda*mean((w-w0)^2)',disagreement_scale=10000,
        similarity='Morgan2048 Tanimoto maximum to actual source-absent organizer structural-code A training bank; no external/test',
        outer_primary='archived5 scaffold folds',confirmation_seeds=[6611,6612],inner_seed=314,subinner_seed=271,
        inner_folds=3,subinner_folds=3,
        selection='honest extra nested calibration per tuning fold; no metaCV of globally archived OOF',
        tie_rule='MSE tolerance1e-6; constant then d then s then sd; stronger regularization first',
        diagnostic_bins_s=[0,.3,.5,.7,.9,1.0000001],diagnostic_bins_d=[0,.1,.3,.6,1.0000001],
        acceptance='dynamic selected procedure beats productionM and tuned constant in primary and both repeats; gain vs constant remains after largest positive row; final chosen family dynamic and changes test; validation passes',
        no_new_independent_data=True,no_leaderboard_selection=True,
        fallback='retain production PR3 constant wS0 and alpha_source0')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    state,emb,smiles=E.prepare(cache);n=len(state['y']);ids=np.arange(n);registry={}
    packages=['numpy','pandas','scipy','scikit-learn','rdkit','joblib','threadpoolctl','catboost','lightgbm',
              'torch','transformers','tokenizers','huggingface-hub','safetensors','requests','pytest']
    (out/'environment.json').write_text(json.dumps({p:importlib.metadata.version(p) for p in packages},indent=2))
    shutil.copy2(cache/'embedding_manifest.json',out/'embedding_manifest.json')
    shutil.copy2(root/'results/audit_20261107/embeddings_mlm.json',out/'original_encoder_manifest.json')
    bundle=joblib.load(root/'results/audit_20261107/selected_model.joblib');frozen=freeze(root,out,bundle)
    reference=ids[~state['flag'][:n]]
    np.testing.assert_array_equal(state['structural_bits'][reference],bundle['baseline_portable_model']['A_reference']['bits'])
    bank=pd.DataFrame(dict(source_row=ids+2,SMILES=smiles[:n],structure_group=state['world'].tau[:n],scaffold_group=state['groups'],source_present=state['flag'][:n]))
    bank.to_csv(out/'organizer_reference_registry.csv',index=False)
    production=E.production(bundle,list(smiles[n:]),{k:v[n:] for k,v in emb.items()})
    production.insert(0,'index',state['test']['index'].to_numpy())
    assert production['index'].tolist()==pd.read_csv(root/'data/raw/sample_submit.csv',usecols=['index'])['index'].tolist()
    merror=float(np.max(np.abs(E.finish(production,production.pred_M.to_numpy())-pd.read_csv(root/'results/audit_20261107/submission.csv')['SI VACV'])))
    serror=float(np.max(np.abs(E.finish(production,production.pred_S.to_numpy())-pd.read_csv(root/'submission_candidate_v7.csv')['SI VACV'])))
    assert max(merror,serror)<1e-7
    save(production,out/'production_components.csv.gz');print('PRODUCTION_REPLAY',merror,serror,flush=True)
    tables={};all_choices=[]
    for scheme,seed in [('primary',None),('confirmation6611',6611),('confirmation6612',6612)]:
        outer=folds(state,ids,'scaffold',2718 if seed is None else seed,archived=seed is None)
        outer_t,inner_t,calibration_t,choices=nested(state,emb,smiles,outer,cache,registry,scheme)
        if scheme=='primary':
            original=pd.read_csv(root/'results/audit_20261107/paired_new_model_errors.csv').sort_values('source_row')
            np.testing.assert_allclose(outer_t.sort_values('source_row').production_M,original.new_prediction,atol=1e-7,rtol=0)
        for role,t in [('outer',outer_t),('inner',inner_t),('calibration',calibration_t)]:save(t,out/(scheme+'_'+role+'_components.csv.gz'))
        tables[scheme]=outer_t;all_choices+=choices
    full_oof,full_calib,family_specs,chosen_spec,scores=tune(state,emb,smiles,ids,np.array([],int),cache,registry,'final_training',-1)
    params=G.fit_gate(full_oof,chosen_spec);constant_params=G.fit_gate(full_oof,family_specs['constant'])
    save(full_oof,out/'final_training_components.csv.gz');save(full_calib,out/'final_subinner_components.csv.gz')
    (out/'final_training_selection.json').write_text(json.dumps(dict(chosen_spec=chosen_spec,gate=params,constant_gate=constant_params,inner_MSE=scores),indent=2))
    (out/'fold_choices.json').write_text(json.dumps(all_choices,indent=2))
    (out/'model_provenance.json.gz').write_bytes(gzip.compress(json.dumps(registry).encode(),mtime=0))
    finalize(root,out,tables,production,params,constant_params,merror,serror)
    assert all(E.sha(root/p)==h for p,h in frozen['file_sha256'].items())

def finalize(root,out,tables,production,params,constant_params,merror,serror):
    rows=[];oofs=[];change_tables=[]
    methods=['production_M','expert_S','fixed50']+['prediction_'+f for f in FAMILIES]+['prediction_selected']
    for scheme,t in tables.items():
        y=t.observed.to_numpy()
        for method in methods:
            rows.append(dict(scheme=scheme,method=method,n=len(t),MSE=mse(y,t[method]),
                relative_to_production_pct=100*(mse(y,t[method])/mse(y,t.production_M)-1),
                relative_to_constant_pct=100*(mse(y,t[method])/mse(y,t.prediction_constant)-1)))
        oofs.append(t)
        if scheme=='primary':
            for control in ['production_M','prediction_constant']:
                delta=(y-t[control].to_numpy())**2-(y-t.prediction_selected.to_numpy())**2
                changes=t[['source_row','observed','structure_group','scaffold_group','source_present','s','d',control,'prediction_selected']].copy()
                changes['control']=control;changes['SSE_reduction']=delta;changes['MSE_contribution']=delta/len(t)
                change_tables.append(changes)
    comparison=pd.DataFrame(rows);comparison.to_csv(out/'comparison.csv',index=False)
    save(pd.concat(oofs,ignore_index=True),out/'nested_oof.csv.gz')
    pd.concat(change_tables,ignore_index=True).sort_values('SSE_reduction',ascending=False).to_csv(out/'row_error_changes.csv',index=False)
    primary=tables['primary'];y=primary.observed.to_numpy()
    approved=params['family']!='constant'
    for t in tables.values():
        approved &= mse(t.observed,t.prediction_selected)<min(mse(t.observed,t.production_M),mse(t.observed,t.prediction_constant))
    gain=(y-primary.prediction_constant.to_numpy())**2-(y-primary.prediction_selected.to_numpy())**2
    approved &= float(gain.sum()-max(gain.max(),0))>0
    proposed,weights=G.predict_gate(production,params);baseline=E.finish(production,production.pred_M.to_numpy())
    affected=np.abs(proposed-baseline)>1e-9;approved &= bool(affected.any())
    fallback=dict(format_version=1,spec=dict(name='production_M',family='constant',penalty=0.),family='constant',penalty=0.,
        w0=0.,s0=params['s0'],d0=params['d0'],beta_s=0.,beta_d=0.,disagreement_scale=10000.,alpha_source=0.)
    config=dict(candidate_gate=params,trained_constant_gate=constant_params,recommended_gate=fallback,
                statistical_acceptance_passed=bool(approved),validation_passed=False,alpha_source=0.,
                recommendation='retain production PR3 wS0 until statistical and validation criteria pass')
    (out/'gate.json').write_text(json.dumps(config,indent=2))
    if approved:pd.DataFrame({'index':production['index'],'SI VACV':proposed}).to_csv(out/'candidate_pending_validation.csv',index=False)
    else:shutil.copy2(root/'results/audit_20261107/submission.csv',out/'recommended_submission_unchanged_PR3.csv')
    production['candidate_weight']=weights;production['candidate_prediction']=proposed
    production['recommended_prediction']=baseline
    save(production,out/'production_with_gate.csv.gz')
    cluster=pd.DataFrame(dict(group=primary.scaffold_group,gain=gain)).groupby('group').agg(gain=('gain','sum'),n=('gain','size'))
    draw=np.random.default_rng(8137).integers(0,len(cluster),(2000,len(cluster)))
    boot=cluster.gain.to_numpy()[draw].sum(1)/cluster.n.to_numpy()[draw].sum(1)
    pair_diff=np.abs(production.pred_S-production.pred_M)>1e-9
    primary_change=np.abs(primary.prediction_selected-primary.production_M)>1e-9
    summary=dict(experts_level='finalSI after original per-expert clipping, before common lookup',alpha_source=0.,
        primary_MSE={m:mse(y,primary[m]) for m in methods},
        primary_gate_vs_constant_improvement_pct=100*(1-mse(y,primary.prediction_selected)/mse(y,primary.prediction_constant)),
        production_M_max_reproduction_error=merror,production_S_max_reproduction_error=serror,
        production_pair_different_rows=int(pair_diff.sum()),source_present_pair_different=int((pair_diff&production.source_present).sum()),
        production_candidate_affected=int(affected.sum()),production_candidate_structures=int(production.loc[affected,'structure_group'].nunique()),
        production_candidate_scaffolds=int(production.loc[affected,'scaffold_group'].nunique()),
        primary_selected_affected=int(primary_change.sum()),primary_selected_structures=int(primary.loc[primary_change,'structure_group'].nunique()),
        primary_selected_scaffolds=int(primary.loc[primary_change,'scaffold_group'].nunique()),
        gain_vs_constant_SSE=float(gain.sum()),largest_positive_gain_SSE=float(gain.max()),
        gain_without_largest_positive=float(gain.sum()-max(gain.max(),0)),
        scaffold_bootstrap_gain_vs_constant_MSE_CI95=np.quantile(boot,[.025,.975]).tolist(),
        statistical_acceptance_passed=bool(approved),recommended_wS_constant=0.,new_Kaggle_MSE=None,
        known_PR3_Public_MSE=6327610.34997,independent_new_data=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    # Descriptive fit on the global outer OOF is deliberately not used above
    # for selection, acceptance or deployment; it is an optimistic diagnostic.
    diagnostic=G.fit_gate(primary,G.specs()[0])
    (out/'diagnostic_global_constant.json').write_text(json.dumps(dict(diagnostic_only=True,
        warning='Fitted to outer evaluation OOF labels; not independent quality',gate=diagnostic),indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
