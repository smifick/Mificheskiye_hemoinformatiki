"""Paired error attribution, external-bank ablations, uncertainty and figures."""
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss,roc_auc_score

import audit_regression as A
import research_heads as H


def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/audit_20261107'
    state=joblib.load(out/'state_cache.joblib');w=state['world'];y=state['y'];n=len(y)
    base=np.load(out/'baseline_oof.npz');emb=H.load_embeddings(out)
    result=json.loads((out/'targeted_results.json').read_text())
    chosen=result['selected_for_submission']
    spec=next(s for s in H.candidate_specs() if s['name']==chosen.removesuffix('_blend50')) if chosen not in ['v7','v7_without_upper_cap'] else None
    oof=pd.read_csv(out/'targeted_nested_oof.csv');primary=oof[oof.scheme.eq('scaffold')].sort_values('source_row')
    pred=primary.fixed_candidate.to_numpy()
    old=base['v7']
    errors=pd.read_csv(out/'comparable_oof_errors.csv')
    errors=errors[errors.model.eq('v7')].sort_values('source_row').reset_index(drop=True)
    errors['new_prediction']=pred;errors['new_squared_error']=(pred-y)**2
    errors['SSE_reduction']=errors.squared_error-errors.new_squared_error
    diagnostic_raw=pd.read_csv(root/'data/raw/train.csv')
    conflicting=pd.read_csv(out/'conflicting_inputs.csv')
    badrows=set()
    for text in conflicting.loc[conflicting.representation.eq('full_canonical_graph'),'source_rows']:
        badrows.update(json.loads(text))
    causes=[]
    for _,row in errors.iterrows():
        reasons=[]
        if row.source_row in badrows:reasons.append('same full chemical graph has conflicting known labels/context')
        if row.observed>10000:reasons.append('rare high-SI tail; neighbors have much lower labels; no supported extrapolation')
        if row.label_type=='technical_10000' and row.new_prediction<5000:reasons.append('underestimated probability of technical code')
        if row.label_type=='observed_ratio' and row.observed<100 and row.new_prediction>5000:reasons.append('overestimated probability of technical code')
        if row.max_train_tanimoto<.3:reasons.append('low structural similarity to the training fold')
        if not reasons:reasons.append('residual regression/code uncertainty; cause not independently established')
        causes.append('; '.join(reasons))
    errors['diagnostic_reason_not_a_feature']=causes
    errors.to_csv(out/'paired_new_model_errors.csv',index=False)
    errors.sort_values('new_squared_error',ascending=False).head(20).to_csv(out/'largest_new_model_errors.csv',index=False)
    errors.sort_values('SSE_reduction',ascending=False).head(20).to_csv(out/'observations_driving_improvement.csv',index=False)
    errors['scaffold']=errors.scaffold.fillna('ACYCLIC')
    groups=errors.groupby('scaffold').agg(n=('source_row','size'),baseline_SSE=('squared_error','sum'),
        candidate_SSE=('new_squared_error','sum'),SSE_reduction=('SSE_reduction','sum')).reset_index().sort_values('SSE_reduction',ascending=False)
    groups.to_csv(out/'groups_driving_improvement.csv',index=False)
    summaries=[]
    for label,p in [('v7',old),(chosen,pred)]:
        se=(p-y)**2;order=np.sort(se)[::-1]
        summaries.append(dict(model=label,MSE=float(se.mean()),**{f'top_{k}_SSE_fraction':float(order[:k].sum()/se.sum()) for k in [1,5,10,20]}))
    pd.DataFrame(summaries).to_csv(out/'new_error_concentration.csv',index=False)
    rows=[]
    for model,col in [('v7','squared_error'),(chosen,'new_squared_error')]:
        for label,d in errors.groupby('label_type'):
            rows.append(dict(model=model,label_type=label,n=len(d),MSE=float(d[col].mean()),SSE_fraction=float(d[col].sum()/errors[col].sum())))
    pd.DataFrame(rows).to_csv(out/'new_errors_by_type.csv',index=False)
    errors['similarity_bin']=pd.cut(errors.max_train_tanimoto,[0,.3,.5,.7,.9,1.000001],include_lowest=True)
    errors.groupby('similarity_bin',observed=True)[['squared_error','new_squared_error']].agg(['size','mean','sum']).to_csv(out/'new_errors_by_similarity.csv')
    # Cluster uncertainty: average repeated error differences per original row
    # before resampling whole scaffold groups. No repeated rows treated as i.i.d.
    conf=pd.read_csv(out/'targeted_confirmation_oof.csv')
    intervals=[];rng=np.random.default_rng(20261107)
    for scheme,d in conf.groupby('scheme'):
        wide=d.pivot(index=['seed','source_row','observed'],columns='model',values='prediction')
        yy=wide.index.get_level_values('observed').to_numpy()
        delta=((wide[chosen]-yy)**2-(wide['v7']-yy)**2).groupby(level='source_row').mean().sort_index().to_numpy()
        groups=state['groups'];unique=np.unique(groups)
        sums=np.array([delta[groups==g].sum() for g in unique]);sizes=np.array([(groups==g).sum() for g in unique])
        ix=rng.integers(0,len(unique),size=(20000,len(unique)))
        boot=sums[ix].sum(1)/sizes[ix].sum(1)
        low,high=np.quantile(boot,[.025,.975])
        intervals.append(dict(scheme=scheme,mean_MSE_change=float(delta.mean()),cluster_bootstrap_95pct_low=float(low),cluster_bootstrap_95pct_high=float(high)))
    pd.DataFrame(intervals).to_csv(out/'paired_cluster_uncertainty.csv',index=False)
    # Obtain actual code probabilities using only training targets per fold.
    pc=np.zeros(n)
    for fold in sorted(set(base['fold'])):
        va=np.flatnonzero(base['fold']==fold);tr=np.flatnonzero(base['fold']!=fold)
        if spec:
            fitted=H.fit_head(spec,state,emb,tr);pc[va]=H.predict_head(fitted,state,emb,va,tr)
        else:pc[va]=.5*(base['p_code_v6'][va]+base['p_code_feature'][va])
    old_pc=.5*(base['p_code_v6']+base['p_code_feature'])
    quality=[]
    for model,p in [('v7_probability_mean',old_pc),(chosen,pc)]:
        quality.append(dict(model=model,Brier=float(brier_score_loss(y==10000,p)),ROC_AUC=float(roc_auc_score(y==10000,p))))
    pd.DataFrame(quality).to_csv(out/'code_probability_metrics.csv',index=False)
    # Existing external data: distinguish new-to-organizer from new-to-project.
    allcells=pd.read_csv(root/'data/chembl_package/vacv_si_all_cells.csv')
    vero=pd.read_csv(root/'data/chembl_package/vacv_si_vero.csv')
    official_parents=set(w.tau[:w.n_tr+w.n_te]);official_scaffolds=set(w.scf[:w.n_tr+w.n_te])
    external_summary={'all_cells_rows':len(allcells),'vero_rows':len(vero),
        'physical_bank_counts':w.ext.tier.value_counts().to_dict(),'new_numeric_SI_labels_downloaded_in_this_study':0}
    for label,d in [('all_cells',allcells),('vero',vero)]:
        keys=[A.T.taut_key(A.T.std(s)) for s in d.SMILES]
        scaffolds=[A.T.gscaf(A.T.std(s)) for s in d.SMILES]
        novel=np.array([k not in official_parents for k in keys]);newsc=np.array([g not in official_scaffolds for g in scaffolds])
        external_summary[label]={'structures':len(set(keys)),'scaffolds':len(set(scaffolds)),
            'new_structures_vs_train':len(set(keys)-set(w.tau[:w.n_tr])),
            'test_overlap_structures':len(set(keys)&set(w.tau[w.n_tr:w.n_tr+w.n_te])),
            'after_removing_train_test_structures':{'rows':int(novel.sum()),'structures':len(set(np.array(keys)[novel])),'scaffolds':len(set(np.array(scaffolds)[novel]))},
            'after_also_removing_train_test_scaffolds':{'rows':int((novel&newsc).sum()),'structures':len(set(np.array(keys)[novel&newsc])),'scaffolds':len(set(np.array(scaffolds)[novel&newsc]))}}
    archive=root/'data/chembl_package/vacv_chembl_data.zip'
    with zipfile.ZipFile(archive) as z:
        summaries=[name for name in z.namelist() if name.endswith('chembl_export/collection_summary.json')]
        if summaries:external_summary['updated_parser_export_summary']=json.loads(z.read(summaries[0]))
    (out/'external_novelty_audit.json').write_text(json.dumps(external_summary,indent=2))
    # Physical-bank ablation only: fixed source-presence features and fixed code
    # heads. No new inferred SI pairs are constructed from raw endpoints.
    bank_pred={name:np.zeros(n) for name in ['legacy_V_plus_C','reviewed_V_only','organizer_physical_only']}
    fold_eligible=[]
    for fold in sorted(set(base['fold'])):
        va=np.flatnonzero(base['fold']==fold);tr=np.flatnonzero(base['fold']!=fold)
        mask=A.T.ext_allowed(w,va)
        for name in bank_pred:
            cfg=dict(A.V6.CFG)
            cfg['w']=({'V':1.,'C':1.} if name=='legacy_V_plus_C' else {'V':1.} if name=='reviewed_V_only' else {})
            ew=np.array([cfg['w'].get(t,0.) for t in w.ext_tier])
            r=A.T.fit_B(w,tr,mask,ew,va,cfg)
            bank_pred[name][va]=H.complete_predictions(state,tr,va,r,pc[va])
            fold_eligible.append(dict(fold=int(fold),bank=name,eligible_external_rows=int((mask&(ew>0)).sum())))
    pd.DataFrame([dict(bank=name,MSE=A.loss(y,p)) for name,p in bank_pred.items()]).to_csv(out/'physical_bank_ablation.csv',index=False)
    pd.DataFrame(fold_eligible).to_csv(out/'physical_bank_fold_counts.csv',index=False)
    # Add empirically observed embedding identities to the representation audit.
    floors=pd.read_csv(out/'representation_floors.csv')
    floors=floors[~floors.representation.str.startswith('ChemBERTa_')]
    extra=[]
    for encoder in ['mtr','mlm']:
        for view in ['raw','canonical']:
            summary,_=A.empirical_floor(A.row_hashes(emb[f'{encoder}_{view}'][:n]),y)
            extra.append(dict(representation=f'ChemBERTa_{encoder}_{view}',**summary))
    pd.concat([floors,pd.DataFrame(extra)],ignore_index=True).to_csv(out/'representation_floors.csv',index=False)
    A.make_registry(root,out);A.audit_submissions(root,out)
    # Export scientific plots as standalone artifacts.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for name,p in [('v7',old),('Notation + frozen MLM',pred)]:
        se=np.sort((p-y)**2)[::-1];ax.plot(np.arange(1,n+1),np.cumsum(se)/se.sum(),label=name)
    ax.set(xlabel='Most erroneous observations included',ylabel='Fraction of total squared error',xlim=(1,80),ylim=(0,1))
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'error_concentration.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for label,d in errors.groupby('label_type'):
        ax.scatter(d.max_train_tanimoto,d.new_squared_error,s=22,alpha=.65,label=label)
    ax.set_yscale('log');ax.set(xlabel='Maximum training-fold Morgan similarity',ylabel='Squared OOF error (new candidate)')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'error_vs_similarity.png',dpi=200);plt.close(fig)
    dev=pd.read_csv(out/'targeted_development.csv')
    names=['v7','raw_smiles_notation_logistic','notation_mlm_probability_average','mlm_raw_logistic_0.1','mlm_canonical_logistic_0.1']
    fig,ax=plt.subplots(figsize=(10,5));plot=dev[dev.model.isin(names)].pivot(index='model',columns='scheme',values='MSE')/1e6
    plot=plot.rename(index={'raw_smiles_notation_logistic':'Notation',
                            'notation_mlm_probability_average':'Notation + MLM',
                            'mlm_raw_logistic_0.1':'MLM raw',
                            'mlm_canonical_logistic_0.1':'MLM canonical'})
    plot.plot.bar(ax=ax);ax.set(ylabel='OOF MSE (millions)',xlabel='');ax.tick_params(axis='x',rotation=15)
    fig.tight_layout();fig.savefig(out/'raw_vs_canonical.png',dpi=200);plt.close(fig)
    summary={'selected':chosen,'primary_v7_MSE':A.loss(y,old),'primary_candidate_MSE':A.loss(y,pred),
        'primary_change_pct':100*(A.loss(y,pred)/A.loss(y,old)-1),
        'error_concentration':pd.read_csv(out/'new_error_concentration.csv').to_dict('records'),
        'code_probability_metrics':quality,'uncertainty':intervals,
        'physical_bank_ablation':pd.read_csv(out/'physical_bank_ablation.csv').to_dict('records'),
        'note':'The bank ablation is diagnostic; it does not retrospectively alter the frozen classifier-study selection.'}
    (out/'diagnosis_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
