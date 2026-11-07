"""Assemble traceable physical labels and separate auxiliary tasks."""
from pathlib import Path
import hashlib,json,re
import joblib
import numpy as np
import pandas as pd
import train_v5 as T
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/external_20261107'

def select_auxiliary_rows(data,budget=128):
    data=data.copy()
    data['selection_hash']=data.activity_id.map(lambda s:hashlib.sha256(('ChEMBL:'+str(s)).encode()).hexdigest())
    return data.sort_values('selection_hash').groupby('endpoint',sort=True).head(budget)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    w=joblib.load(ROOT/'results/audit_20261107/state_cache.joblib')['world']
    test=set(w.tau[w.n_tr:w.n_tr+w.n_te]);old=set(w.tau)
    records=[]
    def add(smiles,value,source,task,sign,measurement_id,doi='',url='',license='',cell=''):
        mol=T.std(smiles)
        if mol is None:raise ValueError('Invalid source structure')
        parent=T.taut_key(mol)
        records.append(dict(SMILES=smiles,value=float(value),source=source,task=task,sign=sign,
            measurement_id=str(measurement_id),doi=doi,source_url=url,license=license,cell=cell,
            tautomer_parent=parent,scaffold=T.gscaf(mol),test_structure_overlap=parent in test,
            new_structure_vs_official_and_legacy=parent not in old))
    for i,row in w.ext.iterrows():
        add(row.SMILES,row.SI,'legacy_'+row.tier,'SI',1,'legacy:'+str(i),row.doi,cell=row.cell)
    p=pd.read_csv(ROOT/'data/external_research/pmc/training_si.csv').fillna('')
    for _,r in p.iterrows():
        if r.SI_relation!='=' or r.IC50_relation!='=' or r.CC50_relation!='=':continue
        add(r.SMILES,r.SI,'PMC10995631','SI',1,str(r.pmcid)+':'+r.table+':'+r.compound_id,
            r.doi,r.source_url,r.license,r.cell_line)
    c=pd.read_csv(ROOT/'data/external_research/chembl/endpoint_bank.csv').fillna('')
    c=c[c.endpoint.isin(['IC50','EC50','CC50'])&c.relation.eq('=')&(pd.to_numeric(c.value_uM,errors='coerce')>0)].copy()
    c=select_auxiliary_rows(c)
    for _,r in c.iterrows():
        cell=re.sub('[^a-z0-9]','',str(r.cell).lower()) or 'unspecified'
        task='ChEMBL:'+r.task+':'+cell
        add(r.SMILES,r.value_uM,'aux_chembl',task,1 if r.endpoint=='CC50' else -1,
            'chembl:'+str(r.activity_id),r.doi,r.source_url,r.license_url,cell)
    p=pd.read_csv(ROOT/'data/external_research/pubchem/auxiliary_endpoints.csv').fillna('')
    p=p[p.relation.eq('=')&(pd.to_numeric(p.value_uM,errors='coerce')>0)]
    for _,r in p.iterrows():
        task='PubChem:'+str(r.pubchem_aid)+':'+r.endpoint+':'+str(r.phenotype)+':'+str(r.channel)
        add(r.SMILES,r.value_uM,'aux_pubchem',task,1 if r.phenotype=='cytotoxicity' else -1,
            'pubchem:'+str(r.pubchem_aid)+':'+str(r.pubchem_sid)+':'+str(r.result_tag)+':'+r.endpoint+':'+str(r.channel),
            r.doi,r.source_url,r.license_url,str(r.cell_line))
    data=pd.DataFrame(records)
    if data.measurement_id.duplicated().any():
        data[data.measurement_id.duplicated(False)].to_csv(OUT/'duplicate_measurement_review.csv',index=False)
        raise ValueError('Repeated measurement IDs require explicit review')
    data[data.test_structure_overlap].to_csv(OUT/'training_bank_test_excluded.csv',index=False)
    eligible=data[~data.test_structure_overlap].drop(columns=['test_structure_overlap']).reset_index(drop=True)
    dest=ROOT/'data/external_research/training_bank.csv';eligible.to_csv(dest,index=False)
    bysource=eligible.groupby('source').agg(rows=('SMILES','size'),structures=('tautomer_parent','nunique'),scaffolds=('scaffold','nunique')).reset_index()
    bysource.to_csv(OUT/'training_bank_counts.csv',index=False)
    manifest={'rows':len(eligible),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
       'test_label_rows_excluded':int(data.test_structure_overlap.sum()),
       'auxiliary_sampling':'128 ChEMBL measurements per endpoint selected by SHA256 of activity ID only; all exact PubChem numeric rows. No selection by values or organizer responses.',
       'physical_SI_definition':'PMC author-matched CC50/IC50; old V/C quality remains explicitly flagged by source',
       'no_external_test_labels':True,'new_code10000_labels':0,'source_counts':bysource.to_dict('records')}
    (OUT/'training_bank_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
