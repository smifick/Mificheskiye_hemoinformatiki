"""Retrospective physical-output check; never used for model selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import external_si_model as M
from run_external_research import configurations
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/external_20261107'

def main():
    ctx=M.load_context(ROOT,ROOT/'data/external_research/training_bank.csv')
    bank=ctx['bank'];query=bank[bank.source.eq('PMC10995631')];keys=set(query.tautomer_parent);scaffolds=set(query.scaffold)
    assert not keys&set(ctx['org_keys']) and not scaffolds&set(ctx['org_scaffolds'])
    protected=bank.tautomer_parent.isin(keys)|bank.scaffold.isin(scaffolds)
    ctx['bank']=bank.copy();ctx['bank'].loc[protected,'source']='excluded_publication_scaffolds'
    rows=ctx['n']+query.index.to_numpy();truth=query.value.to_numpy(float);records=[];summary=[]
    for cfg in configurations()[:3]:
        core=M.fit_physical(ctx,np.arange(ctx['n']),cfg,[])
        pred=np.maximum(core['smear']*np.exp(M.predict_log(ctx,core,rows))-1,0)
        summary.append(dict(model=cfg['name'],n=len(truth),MSE_physical=float(np.mean((pred-truth)**2)),
            RMSE_log1p=float(np.sqrt(np.mean((np.log1p(pred)-np.log1p(truth))**2))),physical_external_rows=core['physical_external_rows']))
        records.extend(dict(model=cfg['name'],measurement_id=r.measurement_id,SMILES=r.SMILES,doi=r.doi,
            SI=float(r.value),prediction_physical=float(p)) for (_,r),p in zip(query.iterrows(),pred))
    pd.DataFrame(records).to_csv(OUT/'publication_transfer_predictions.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'publication_transfer_metrics.csv',index=False)
    note={'source':'PMC10995631','mode':'physical conditional output only, no technical-code labels',
      'excluded_source_rows':int(protected.sum()),'new_independent_validation_claim':False,
      'not_used_for_selection':True,'warning':'Retrospective diagnostic: source and labels were inspected and used in earlier source-ablation candidates. Not a new blind test.'}
    (OUT/'publication_transfer_protocol.json').write_text(json.dumps(note,indent=2)+'\n');print(summary)

if __name__=='__main__':main()
