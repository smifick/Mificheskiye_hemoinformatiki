"""Derived diagnostics for the frozen S/M gate study; never fit or select.

Inner relative-advantage tables describe the training evidence. Outer tables
are separately labelled diagnostic evaluations. Repeated appearances of the
same source row and repeated splits are never counted as new observations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import similarity_weight as G

SCHEMES=('primary','confirmation6611','confirmation6612')
S_BINS=(0.,.3,.5,.7,.9,1.0000001)
D_BINS=(0.,.1,.3,.6,1.0000001)
METHODS=('production_M','expert_S','fixed50','prediction_constant',
         'prediction_s','prediction_d','prediction_sd','prediction_selected')
EPS=1e-9


def counts(table):
    return dict(n=len(table),unique_source_rows=int(table.source_row.nunique()) if 'source_row' in table else len(table),
                unique_structures=int(table.structure_group.nunique()),unique_scaffolds=int(table.scaffold_group.nunique()))


def bins(table):
    return {'s':pd.cut(table.s,S_BINS,include_lowest=True),
            'd':pd.cut(table.d,D_BINS,include_lowest=True)}


def relative_advantage(table,scheme,stage):
    table=table.copy()
    y=table.observed.to_numpy();table['delta_SE']=(table.pred_S-y)**2-(table.pred_M-y)**2
    table['s_bin']=bins(table)['s'];table['d_bin']=bins(table)['d']
    result=[]
    # Aggregate evidence plus per-outer-fold evidence retains the multiplicity
    # of calibration rows without implying independent repeated observations.
    partitions=[('pooled',table)]
    if 'outer_fold' in table:
        partitions += [(str(f),d) for f,d in table.groupby('outer_fold',sort=True)]
    for fold,part in partitions:
        for population,subset in [('all',part),('experts_differ',part[np.abs(part.pred_S-part.pred_M)>EPS])]:
            for axis,columns in [('s',['s_bin']),('d',['d_bin']),('s_x_d',['s_bin','d_bin'])]:
                grouper=columns[0] if len(columns)==1 else columns
                for key,group in subset.groupby(grouper,observed=True,sort=True):
                    delta=group.delta_SE.to_numpy();keys=(key,) if len(columns)==1 else key
                    entry=dict(scheme=scheme,stage=stage,outer_fold=fold,population=population,axis=axis,
                               **counts(group),delta_SE_mean=float(np.mean(delta)),delta_SE_median=float(np.median(delta)),
                               delta_SE_q25=float(np.quantile(delta,.25)),delta_SE_q75=float(np.quantile(delta,.75)),
                               delta_SE_sum=float(delta.sum()),share_S_better=float(np.mean(delta < -1e-7)),
                               share_M_better=float(np.mean(delta > 1e-7)),
                               interpretation='negative delta_SE favors S; counts include repeated training appearances')
                    entry.update({column:str(value) for column,value in zip(columns,keys)})
                    result.append(entry)
    return result


def strata(table):
    y=table.observed.to_numpy();source=table.source_present.to_numpy(bool)
    result=[('all',np.ones(len(table),bool)),('technical10000',y==10000),('technical0',y==0),
            ('physical237',(y>0)&(y!=10000)),('small_SI_le100',y<=100),
            ('physical_tail_gt1000',(y>1000)&(y!=10000)),('source_present',source),('source_absent',~source),
            ('experts_differ',np.abs(table.pred_S-table.pred_M).to_numpy()>EPS)]
    for axis,values in bins(table).items():
        result += [(axis+'_'+str(key),(values==key).to_numpy()) for key in values.cat.categories]
    return result


def stratum_metrics(table,scheme):
    result=[]
    for label,mask in strata(table):
        if not mask.any():continue
        part=table.loc[mask];y=part.observed.to_numpy()
        for method in METHODS:
            error=part[method].to_numpy()-y
            result.append(dict(scheme=scheme,stratum=label,method=method,**counts(part),
                               MSE=float(np.mean(error**2)),MAE=float(np.mean(np.abs(error))),
                               SSE=float(np.sum(error**2)),mean_prediction=float(part[method].mean())))
    return result


def distribution(values,prefix):
    a=np.asarray(values,float)
    return {prefix+'_'+name:float(value) for name,value in (
        ('min',np.min(a)),('q05',np.quantile(a,.05)),('q25',np.quantile(a,.25)),('median',np.median(a)),
        ('mean',np.mean(a)),('q75',np.quantile(a,.75)),('q95',np.quantile(a,.95)),('max',np.max(a)))}


def weight_distributions(table,scheme):
    result=[]
    weights={'production_M':np.zeros(len(table)),'expert_S':np.ones(len(table)),'fixed50':np.full(len(table),.5)}
    weights.update({column.removeprefix('weight_'):table[column].to_numpy() for column in table if column.startswith('weight_')})
    for label,mask in strata(table):
        if not mask.any():continue
        for method,values in weights.items():
            a=values[mask]
            result.append(dict(scheme=scheme,stratum=label,method=method,**counts(table.loc[mask]),
                               **distribution(a,'weight'),share_zero=float(np.mean(a<=1e-12)),share_one=float(np.mean(a>=1-1e-12))))
    return result


def similarity_distributions(table,scheme,stage):
    source=table.source_present.to_numpy(bool);result=[]
    for label,mask in [('all',np.ones(len(table),bool)),('source_present',source),('source_absent',~source)]:
        if not mask.any():continue
        part=table.loc[mask]
        result.append(dict(scheme=scheme,stage=stage,stratum=label,**counts(part),**distribution(part.s,'s'),
                           **distribution(part.d,'d'),**distribution(part.reference_count,'reference_count')))
    return result


def effect(table,prediction,label):
    base=finish(table,table.pred_M.to_numpy());changed=np.abs(prediction-base)>EPS
    source=table.source_present.to_numpy(bool);lookup=table.lookup_applied.to_numpy(bool)
    return dict(method=label,rows=len(table),changed_rows=int(changed.sum()),
                changed_structures=int(table.loc[changed,'structure_group'].nunique()),
                changed_scaffolds=int(table.loc[changed,'scaffold_group'].nunique()),
                changed_source_present=int((changed&source).sum()),changed_source_absent=int((changed&~source).sum()),
                changed_lookup_rows=int((changed&lookup).sum()),max_abs_SI_change=float(np.abs(prediction-base).max()))


def finish(table,prediction):
    return np.where(table.lookup_applied.to_numpy(bool),np.clip(table.lookup_value.to_numpy(),0,10000),prediction)


def production_effects(table,config):
    raw_diff=np.abs(table.pred_S-table.pred_M).to_numpy()>EPS
    final_diff=np.abs(finish(table,table.pred_S.to_numpy())-finish(table,table.pred_M.to_numpy()))>EPS
    source=table.source_present.to_numpy(bool);lookup=table.lookup_applied.to_numpy(bool)
    info=dict(rows=len(table),source_present=int(source.sum()),source_absent=int((~source).sum()),lookup_rows=int(lookup.sum()),
              expert_pair_diff_before_lookup=int(raw_diff.sum()),expert_pair_diff_after_lookup=int(final_diff.sum()),
              pair_diff_hidden_by_lookup=int((raw_diff&~final_diff).sum()),
              different_source_present=int((final_diff&source).sum()),different_source_absent=int((final_diff&~source).sum()),
              different_structures=int(table.loc[final_diff,'structure_group'].nunique()),
              different_scaffolds=int(table.loc[final_diff,'scaffold_group'].nunique()),
              alpha_source=0.,old_PR8_count_not_reused=True)
    result=[]
    for label,key in [('candidate','candidate_gate'),('trained_constant','trained_constant_gate'),('recommended','recommended_gate')]:
        prediction,weights=G.predict_gate(table,config[key]);result.append(effect(table,prediction,label))
    result.append(effect(table,finish(table,.5*(table.pred_S+table.pred_M).to_numpy()),'fixed50'))
    result.append(effect(table,finish(table,table.pred_S.to_numpy()),'expert_S'))
    return info,result


def parameter_centers(out):
    rows=[]
    for choice in json.loads((out/'fold_choices.json').read_text()):
        gates=dict(choice['family_gates']);gates['selected']=choice['selected_gate']
        for role,params in gates.items():
            rows.append(dict(scheme=choice['scheme'],outer_fold=choice['fold'],role=role,
                             **{k:params[k] for k in ('family','w0','s0','d0','beta_s','beta_d','penalty','training_rows')}))
    selection=json.loads((out/'final_training_selection.json').read_text())
    for role,key in [('selected','gate'),('constant','constant_gate')]:
        params=selection[key]
        rows.append(dict(scheme='final_training',outer_fold=-1,role=role,
                         **{k:params[k] for k in ('family','w0','s0','d0','beta_s','beta_d','penalty','training_rows')}))
    return rows


def report(root,out):
    if not (out/'summary.json').exists():raise FileNotFoundError('Finish run_similarity_gate.py before derived reporting')
    relative=[];metrics=[];weights=[];similarities=[];added=[]
    for scheme in SCHEMES:
        inner=pd.read_csv(out/(scheme+'_inner_components.csv.gz'))
        outer=pd.read_csv(out/(scheme+'_outer_components.csv.gz'))
        calibration=pd.read_csv(out/(scheme+'_calibration_components.csv.gz'))
        relative+=relative_advantage(inner,scheme,'inner_training_diagnostic')
        relative+=relative_advantage(outer,scheme,'outer_evaluation_diagnostic')
        metrics+=stratum_metrics(outer,scheme);weights+=weight_distributions(outer,scheme)
        for stage,table in [('subinner_gate_calibration',calibration),('inner_validation',inner),('outer_validation',outer)]:
            similarities+=similarity_distributions(table,scheme,stage)
        y=outer.observed.to_numpy();mse_d=float(np.mean((outer.prediction_d-y)**2));mse_sd=float(np.mean((outer.prediction_sd-y)**2))
        added.append(dict(scheme=scheme,n=len(outer),d_only_MSE=mse_d,sd_MSE=mse_sd,
                          addition_of_s_MSE_reduction=mse_d-mse_sd,addition_of_s_improvement_pct=100*(1-mse_sd/mse_d),
                          interpretation='family-specific regularization selected only inside train; repeated schemes use same molecules'))
    for stage,file in [('final_training','final_training_components.csv.gz'),('final_subinner','final_subinner_components.csv.gz'),
                       ('production','production_components.csv.gz')]:
        table=pd.read_csv(out/file);similarities+=similarity_distributions(table,'final',stage)
    production=pd.read_csv(out/'production_components.csv.gz');config=json.loads((out/'gate.json').read_text())
    effects,effect_rows=production_effects(production,config)
    outputs={'relative_advantage_bins.csv':relative,'stratum_metrics.csv':metrics,'weight_distributions.csv':weights,
             'similarity_distributions.csv':similarities,'addition_of_similarity.csv':added,
             'production_gate_effects.csv':effect_rows,'gate_parameter_centers.csv':parameter_centers(out)}
    for filename,rows in outputs.items():pd.DataFrame(rows).to_csv(out/filename,index=False)
    (out/'production_expert_effects.json').write_text(json.dumps(effects,indent=2))
    limitations=dict(no_fitting_or_selection=True,relative_advantage='negative delta_SE favors S, positive favors M',
        bin_edges=dict(s=S_BINS,d=D_BINS),
        inner_evidence='descriptive training evidence, repeated source-row appearances counted separately and unique counts shown',
        outer_evidence='diagnostic comparison after predictions frozen; not used to redesign gate',
        groups='unique structure/scaffold counts do not establish statistical independence',
        repeats='same historical molecules; no new independent observations',
        similarity_shift='reference bank grows across nesting levels and in final inference; gate centers remain fixed from training')
    (out/'diagnostics_scope.json').write_text(json.dumps(limitations,indent=2))
    print(json.dumps(dict(production=effects,addition_of_similarity=added),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default='results/similarity_gate_20261107')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];report(root,root/args.out)
