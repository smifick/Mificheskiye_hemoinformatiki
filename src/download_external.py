"""Retrieve ChEMBL VACV assays, endpoints, publication DOI and conservative SI records.
Source API: https://www.ebi.ac.uk/chembl/api/data/ ; cache + URL/SHA256 manifest.
No EC50 is silently renamed IC50. Missing concentrations stay missing.
"""
import argparse, concurrent.futures, hashlib, json, re, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode, urljoin
import pandas as pd
from rdkit import Chem
BASE='https://www.ebi.ac.uk/chembl/api/data/'

class Client:
    def __init__(self, cache):
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True);self.manifest=[]
    def get(self,url):
        path=self.cache/(hashlib.sha256(url.encode()).hexdigest()+'.json')
        if path.exists(): raw=path.read_bytes()
        else:
            for attempt in range(3):
                try:
                    with urlopen(Request(url,headers={'User-Agent':'VACV-academic-data-audit/1.0'}),timeout=45) as f:raw=f.read()
                    path.write_bytes(raw);break
                except Exception:
                    if attempt==2:raise
                    time.sleep(attempt+1)
        self.manifest.append({'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'cache_file':path.name})
        return json.loads(raw)
    def pages(self,resource,**filters):
        url=BASE+resource+'.json?'+urlencode(dict(filters,limit=1000));rows=[]
        while url:
            data=self.get(url);rows+=data['activities' if resource=='activity' else resource+'s'];nxt=data.get('page_meta',{}).get('next');url=urljoin(BASE,nxt) if nxt else None
        return rows
    def batches(self,resource,field,ids,extra=None):
        chunks=[ids[i:i+30] for i in range(0,len(ids),30)]
        def run(chunk):return self.pages(resource,**{field+'__in':','.join(chunk),**(extra or {})})
        rows=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(run,chunks):rows+=result
        return rows

def norm_cell(a):
    v=(a or {}).get('assay_cell_type') or ''
    return re.sub(r'[^a-z0-9]','',v.lower())

def duration(desc):
    x=re.findall(r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b',(desc or '').lower())
    return x[-1] if len(x)==1 else ''

def finite_number(v):
    try:
        z=float(v);return z if z>0 and z<float('inf') else None
    except (TypeError,ValueError):return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--cache',default='work/chembl_cache');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);c=Client(a.cache)
    assays=c.pages('assay',assay_organism__icontains='Vaccinia');amap={r['assay_chembl_id']:r for r in assays}
    print('VACV assays:',len(assays),flush=True)
    # Restrict to functional cell/organism assays; molecular binding is a different endpoint.
    usable=[r['assay_chembl_id'] for r in assays if r.get('assay_type')=='F']
    acts=c.batches('activity','assay_chembl_id',usable,{'standard_type__in':'IC50,EC50,CC50,SI,Selectivity index'})
    print('VACV endpoint records:',len(acts),flush=True)
    docs=sorted({r['document_chembl_id'] for r in acts if r.get('document_chembl_id')})
    publications=c.batches('document','document_chembl_id',docs)
    dmap={r['document_chembl_id']:r for r in publications}
    cyto=c.batches('activity','document_chembl_id',docs,{'standard_type__in':'CC50'})
    more=sorted({r['assay_chembl_id'] for r in cyto if r['assay_chembl_id'] not in amap})
    amap.update({r['assay_chembl_id']:r for r in c.batches('assay','assay_chembl_id',more)})
    records={str(r['activity_id']):r for r in acts+cyto};normalized=[]
    vacv_ids={str(r['activity_id']) for r in acts}
    unit_scale={'nM':.001,'uM':1.,'µM':1.,'μM':1.,'mM':1000.,'M':1e6,'pM':1e-6}
    for aid,r in records.items():
        assay=amap.get(r['assay_chembl_id'],{});doc=dmap.get(r.get('document_chembl_id'),{})
        endpoint=r.get('standard_type') or '';value=finite_number(r.get('standard_value'));units=r.get('standard_units') or ''
        if endpoint.lower() in ['si','selectivity index'] and str(r.get('standard_value')) in ['0','0.0']:value=0.0
        relation=r.get('standard_relation') or r.get('relation') or ''
        smi=r.get('canonical_smiles') or '';mol=Chem.MolFromSmiles(smi) if smi else None
        conc=value*unit_scale[units] if value and units in unit_scale else None
        direct_si=value if endpoint.lower() in ['si','selectivity index'] and units in ['','dimensionless','ratio'] else None
        relevant=aid in vacv_ids
        bad=[]
        if not mol:bad.append('invalid_or_missing_smiles')
        if relation!='=':bad.append('censored_or_unknown_relation')
        if r.get('data_validity_comment'):bad.append('chembl_validity_flag')
        if r.get('potential_duplicate'):bad.append('potential_duplicate')
        if value is None:bad.append('nonpositive_or_missing_value')
        if endpoint in ['IC50','EC50','CC50'] and conc is None:bad.append('unsupported_concentration_units')
        normalized.append(dict(SMILES=Chem.MolToSmiles(mol) if mol else smi,IC50_uM=conc if endpoint=='IC50' else None,
            EC50_uM=conc if endpoint=='EC50' else None,CC50_uM=conc if endpoint=='CC50' else None,SI=direct_si,
            doi=doc.get('doi') or doc.get('doi_chembl') or '',endpoint=endpoint,value=r.get('standard_value'),units=units,relation=relation,
            activity_id=aid,molecule_chembl_id=r.get('molecule_chembl_id'),assay_chembl_id=r['assay_chembl_id'],
            document_chembl_id=r.get('document_chembl_id'),assay_organism=assay.get('assay_organism'),
            cell_line=assay.get('assay_cell_type') or '',cell_key=norm_cell(assay),duration_hours=duration(assay.get('description')),
            assay_description=assay.get('description') or r.get('assay_description') or '',is_vacv_assay=relevant,
            validation_status='valid_endpoint' if not bad else ';'.join(bad),
            source='ChEMBL',source_url=BASE+'activity/'+aid+'.json',publication_url=('https://doi.org/'+doc['doi']) if doc.get('doi') else ''))
    df=pd.DataFrame(normalized);df.to_csv(out/'external_endpoints.csv',index=False)
    pd.DataFrame(assays).to_json(out/'vacv_assays.json',orient='records',indent=2)
    pairs=[];direct=[]
    valid=df[df.validation_status=='valid_endpoint']
    cc=valid[valid.endpoint=='CC50']
    for _,r in valid[(valid.is_vacv_assay)&(valid.endpoint.isin(['IC50','EC50','SI','Selectivity index']))].iterrows():
        if pd.notna(r.SI):
            direct.append(dict(SMILES=r.SMILES,IC50_uM=None,CC50_uM=None,SI=r.SI,doi=r.doi,
                               source='ChEMBL_direct_SI',source_url=r.source_url,assay_chembl_id=r.assay_chembl_id,
                               document_chembl_id=r.document_chembl_id,validation_status='direct_reported_SI_needs_assay_review'))
            continue
        for _,s in cc[(cc.molecule_chembl_id==r.molecule_chembl_id)&(cc.document_chembl_id==r.document_chembl_id)].iterrows():
            matched=bool(r.cell_key and r.cell_key==s.cell_key and r.duration_hours and r.duration_hours==s.duration_hours)
            denom=r.IC50_uM if r.endpoint=='IC50' else r.EC50_uM
            if not pd.notna(denom) or denom<=0:continue
            pairs.append(dict(SMILES=r.SMILES,IC50_uM=r.IC50_uM,EC50_uM=r.EC50_uM,CC50_uM=s.CC50_uM,
                SI=s.CC50_uM/denom,doi=r.doi,denominator_endpoint=r.endpoint,
                antiviral_activity_id=r.activity_id,cytotoxicity_activity_id=s.activity_id,
                cell_line_antiviral=r.cell_line,cell_line_cyto=s.cell_line,
                duration_antiviral=r.duration_hours,duration_cyto=s.duration_hours,
                source_url=r.source_url,cyto_source_url=s.source_url,
                validation_status='metadata_matched_manual_review_required' if matched else 'blocked_context_not_verified'))
    pair_cols=['SMILES','IC50_uM','EC50_uM','CC50_uM','SI','doi','denominator_endpoint','antiviral_activity_id','cytotoxicity_activity_id','cell_line_antiviral','cell_line_cyto','duration_antiviral','duration_cyto','source_url','cyto_source_url','validation_status']
    pd.DataFrame(pairs,columns=pair_cols).to_csv(out/'candidate_si_pairs.csv',index=False)
    pd.DataFrame(direct,columns=['SMILES','IC50_uM','CC50_uM','SI','doi','source','source_url','assay_chembl_id','document_chembl_id','validation_status']).to_csv(out/'direct_si.csv',index=False)
    ic50_pairs=pd.DataFrame(pairs,columns=pair_cols)
    ic50_pairs=ic50_pairs[ic50_pairs.denominator_endpoint=='IC50'].copy()
    ic50_pairs.to_csv(out/'external_si_database.csv',index=False)
    df[['SMILES','IC50_uM','CC50_uM','SI','doi','EC50_uM','endpoint','relation','validation_status','source_url']].to_csv(out/'collected_database.csv',index=False)
    summary={'source':'ChEMBL','retrieved_utc':datetime.now(timezone.utc).isoformat(),'vacv_assays':len(assays),'functional_assays':len(usable),
             'vacv_endpoint_records':len(acts),'additional_cc50_records':len(cyto),'all_unique_endpoint_records':len(df),
             'endpoint_counts':df.endpoint.value_counts().to_dict(),'valid_endpoint_rows':int((df.validation_status=='valid_endpoint').sum()),
             'rows_with_doi':int(df.doi.ne('').sum()),'direct_si_records':len(direct),'candidate_pairs':len(pairs),
             'automatically_approved_training_rows':0,
             'policy':'No automatic ratio across unmatched cell/time contexts; EC50 separate from IC50; external labels not merged into organizer training.',
             'license_url':'https://chembl.gitbook.io/chembl-interface-documentation/about/terms-of-use',
             'queries':len(c.manifest)}
    (out/'collection_summary.json').write_text(json.dumps(summary,indent=2))
    (out/'request_manifest.json').write_text(json.dumps(c.manifest,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
