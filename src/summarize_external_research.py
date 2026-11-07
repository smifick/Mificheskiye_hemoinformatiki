"""Paired error attribution, source novelty, actual final training composition."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import train_v5 as T
import train_v7 as V7
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/external_20261107'

def main():
    data=pd.read_csv(OUT/'source_development_oof.csv');data['squared_error']=(data.prediction-data.observed)**2
    base=data[data.model.eq('clean_legacy')][['scheme','source_row','squared_error']].rename(columns={'squared_error':'baseline_squared_error'})
    data=data.merge(base,on=['scheme','source_row'],validate='many_to_one')
    data['SSE_reduction']=data.baseline_squared_error-data.squared_error
    data['label_type']=np.select([data.observed.eq(10000),data.observed.eq(0)],['technical_10000','technical_zero'],default='measured_SI')
    data.to_csv(OUT/'source_paired_errors.csv',index=False)
    data.groupby(['scheme','model','label_type']).agg(n=('source_row','size'),MSE=('squared_error','mean'),SSE_reduction=('SSE_reduction','sum')).to_csv(OUT/'source_errors_by_type.csv')
    raw=pd.read_csv(ROOT/'data/raw/train.csv',usecols=['SMILES','SI VACV'])
    scaf=[T.gscaf(T.std(s)) for s in raw.SMILES]
    data['scaffold']=data.source_row.map({i+2:s for i,s in enumerate(scaf)})
    data.groupby(['scheme','model','scaffold'],dropna=False).agg(n=('source_row','size'),SSE_reduction=('SSE_reduction','sum')).to_csv(OUT/'source_group_changes.csv')
    official=pd.concat([pd.read_csv(ROOT/'data/raw/train.csv',usecols=['SMILES']),pd.read_csv(ROOT/'data/raw/test.csv',usecols=['SMILES'])])
    officialkeys={V7.source_key(s) for s in official.SMILES};officialscaf={T.gscaf(T.std(s)) for s in official.SMILES}
    old=pd.read_csv(ROOT/'data/external/external_endpoints.csv',usecols=['SMILES']).dropna()
    previouskeys=officialkeys|{V7.source_key(s) for s in old.SMILES.unique()}
    previousscaf=officialscaf|{T.gscaf(T.std(s)) for s in old.SMILES.unique()}
    bank=pd.read_csv(ROOT/'data/external_research/training_bank.csv');novelty=[]
    for source,d in bank.groupby('source'):
        keys=set(d.tautomer_parent);sc=set(d.scaffold.dropna())
        novelty.append(dict(source=source,rows=len(d),structures=len(keys),scaffolds=len(sc),
           structures_new_vs_official=len(keys-officialkeys),scaffolds_new_vs_official=len(sc-officialscaf),
           structures_new_vs_previous_project=len(keys-previouskeys),scaffolds_new_vs_previous_project=len(sc-previousscaf)))
    pd.DataFrame(novelty).to_csv(OUT/'used_pilot_bank_novelty.csv',index=False)
    used=pd.read_csv(OUT/'final_used_external.csv')
    final={'external_rows':len(used),'structures':used.tautomer_parent.nunique(),
      'new_structures_vs_previous_project':len(set(used.tautomer_parent)-previouskeys),
      'new_scaffolds_vs_previous_project':len(set(used.scaffold.dropna())-previousscaf),
      'new_collection_rows_in_final':int((~used.source.str.startswith('legacy_')).sum()),
      'note':'New-source rows were evaluated but rejected for final fit by the frozen acceptance criterion.'}
    (OUT/'final_data_composition.json').write_text(json.dumps(final,indent=2)+'\n')
    pred=pd.read_csv(OUT/'submission.csv');oldpred=pd.read_csv(ROOT/'results/audit_20261107/submission.csv')
    diff=pred['SI VACV']-oldpred['SI VACV']
    comp={'rows':len(pred),'mean':float(pred['SI VACV'].mean()),'std':float(pred['SI VACV'].std()),
      'max':float(pred['SI VACV'].max()),'min':float(pred['SI VACV'].min()),
      'correlation_with_successful':float(pred['SI VACV'].corr(oldpred['SI VACV'])),
      'prediction_RMSE_vs_successful':float(np.sqrt(np.mean(diff**2))),
      'changed_rows_at_1e_6':int((abs(diff)>1e-6).sum()),
      'warning':'Prediction disagreement only; new test MSE cannot be computed without labels.'}
    (OUT/'submission_comparison.json').write_text(json.dumps(comp,indent=2)+'\n')
    for source,dest in [('/workspace/work/external_all_tests.log','unit_tests.txt'),('/workspace/work/external_research_run.log','model_run.log')]:
        if Path(source).exists():shutil.copy(source,OUT/dest)
    print(json.dumps({'final':final,'comparison':comp},indent=2))

if __name__=='__main__':main()
