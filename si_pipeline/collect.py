"""Collect ChEMBL IC50/CC50 pairs using an explicitly reviewed assay manifest.

Discover candidate assays first; matching a molecule alone does not establish
that antiviral and cytotoxicity measurements can be divided into a valid SI.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .chemistry import canonicalize

BASE = 'https://www.ebi.ac.uk/chembl/api/data/'
REQUIRED = ['context_id','target_chembl_id','ic50_assay_id','cc50_assay_id',
            'cell_line','ic50_hours','cc50_hours','evidence_url']
FACTORS = {'M':1e6,'mM':1e3,'uM':1.,'nM':1e-3,'pM':1e-6,'fM':1e-9}
PROVENANCE_COLUMNS=['output_row','context_id','smiles','target_chembl_id','cell_line',
    'ic50_hours','cc50_hours','pairing_policy','ic50_assay_id','cc50_assay_id',
    'ic50_activity_ids','cc50_activity_ids','document_chembl_id','evidence_url','replicate_aggregation']
REJECTION_COLUMNS=['context_id','activity_id','reason','level']


class ChEMBL:
    def __init__(self, cache):
        self.cache=Path(cache)
        self.cache.mkdir(parents=True,exist_ok=True)
        self.log=[]
        self.session=requests.Session()
        retry=Retry(total=5,backoff_factor=.5,status_forcelist=[429,500,502,503,504],allowed_methods=['GET'],respect_retry_after_header=True)
        self.session.mount('https://',HTTPAdapter(max_retries=retry))
        self.session.headers.update({'Accept':'application/json','User-Agent':'si-pipeline/1.0'})

    def get(self, resource, params=None):
        url=urljoin(BASE,resource)
        if urlparse(url).netloc!='www.ebi.ac.uk' or not urlparse(url).path.startswith('/chembl/api/data/'):
            raise ValueError('Unexpected API pagination URL')
        request_url=requests.Request('GET',url,params=params).prepare().url
        key=hashlib.sha256(request_url.encode()).hexdigest()
        path=self.cache/(key+'.json')
        hit=path.exists()
        if hit:
            raw=path.read_bytes()
        else:
            response=self.session.get(url,params=params,timeout=(10,60))
            response.raise_for_status()
            response.json()  # Fail before caching an HTML or malformed response.
            raw=response.content
            path.write_bytes(raw)
        self.log.append({'url':request_url,'sha256':hashlib.sha256(raw).hexdigest(),
                         'cache_hit':hit,'accessed_at_utc':datetime.now(timezone.utc).isoformat()})
        return json.loads(raw)

    def pages(self, resource, key, **params):
        url=resource
        params={'limit':1000,**params}
        while url:
            data=self.get(url,params)
            yield from data[key]
            url=data.get('page_meta',{}).get('next')
            params=None


def to_uM(value, units):
    unit=str(units).strip().replace('µ','u').replace('μ','u')
    if unit not in FACTORS:
        raise ValueError('unsupported_units')
    try:
        concentration=float(value)*FACTORS[unit]
    except (ValueError,TypeError,OverflowError):
        raise ValueError('invalid_value') from None
    if not math.isfinite(concentration) or concentration<=0:
        raise ValueError('nonpositive_or_nonfinite_value')
    return concentration


def clean_activity(record, endpoint):
    if record.get('standard_type')!=endpoint:
        raise ValueError('wrong_endpoint')
    if record.get('standard_relation')!='=':
        raise ValueError('censored_or_unknown_relation')
    if record.get('data_validity_comment') not in (None,'','Manually validated'):
        raise ValueError('chembl_validity_flag')
    if record.get('potential_duplicate') in (1,True,'1'):
        raise ValueError('chembl_potential_duplicate')
    smiles=canonicalize(record.get('canonical_smiles'))
    value=to_uM(record.get('standard_value'),record.get('standard_units'))
    return {'smiles':smiles,'value_uM':value,'activity_id':record.get('activity_id'),
            'document_chembl_id':record.get('document_chembl_id')}


def cell_key(value):
    return re.sub(r'[^a-z0-9]','',str(value).lower().replace('cells','').replace('cell',''))


def validate_pair(row, ic_assay, cc_assay):
    for field in [f for f in REQUIRED if f not in ['ic50_hours','cc50_hours']]:
        if pd.isna(row[field]) or not str(row[field]).strip():
            raise ValueError(f'missing_manifest_field:{field}')
    if not str(row['evidence_url']).startswith(('https://','http://','doi:')):
        raise ValueError('evidence_url_required')
    policy=row.get('pairing_policy','equal_duration')
    policy='equal_duration' if pd.isna(policy) else policy
    if policy=='equal_duration':
        hours=[float(row['ic50_hours']),float(row['cc50_hours'])]
        if not all(math.isfinite(h) and h>0 for h in hours) or not math.isclose(*hours):
            raise ValueError('incompatible_or_unknown_exposure_duration')
    elif policy=='author_defined_si':
        verified=str(row.get('reported_si_verified','')).strip().lower() in ['true','1','yes']
        if not verified:raise ValueError('author_defined_pair_requires_table_verification')
    else:raise ValueError('unknown_pairing_policy')
    if ic_assay.get('target_chembl_id')!=row['target_chembl_id']:
        raise ValueError('wrong_inhibition_target')
    doc=ic_assay.get('document_chembl_id')
    if not doc or doc!=cc_assay.get('document_chembl_id'):
        raise ValueError('different_or_unknown_source_documents')
    for assay in [ic_assay,cc_assay]:
        declared=assay.get('assay_cell_type')
        if declared and cell_key(declared)!=cell_key(row['cell_line']):
            raise ValueError('cell_line_metadata_conflict')
    cells=[a.get('cell_chembl_id') for a in [ic_assay,cc_assay]]
    if all(cells) and cells[0]!=cells[1]:
        raise ValueError('different_cell_identifiers')
    return doc


def collect_pairs(client, manifest):
    if not set(REQUIRED).issubset(manifest.columns):
        raise ValueError('Assay manifest must contain '+','.join(REQUIRED))
    if manifest.empty or manifest.context_id.duplicated().any():
        raise ValueError('Provide nonempty, uniquely identified reviewed assay pairs')
    rows=[];provenance=[];rejected=[]
    for _,row in manifest.iterrows():
        ia=client.get(f'assay/{row.ic50_assay_id}.json')
        ca=client.get(f'assay/{row.cc50_assay_id}.json')
        try:
            doc=validate_pair(row,ia,ca)
        except (ValueError,TypeError) as error:
            rejected.append({'context_id':row.context_id,'reason':str(error),'level':'pair'})
            continue
        document=client.get(f'document/{doc}.json')
        doi=(document.get('doi') or '').strip()
        doi=re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)','',doi,flags=re.I)
        if doi and not re.fullmatch(r'10\.\d{4,9}/\S+',doi,re.I):
            rejected.append({'context_id':row.context_id,'reason':'invalid_doi_left_blank','level':'document'})
            doi=''
        tables={}
        for endpoint,assay_id in [('IC50',row.ic50_assay_id),('CC50',row.cc50_assay_id)]:
            cleaned=[];seen=set()
            for record in client.pages('activity.json','activities',assay_chembl_id=assay_id,standard_type=endpoint):
                aid=record.get('activity_id')
                if aid is not None and aid in seen:
                    continue
                if aid is not None:seen.add(aid)
                try:
                    value=clean_activity(record,endpoint)
                    if value['document_chembl_id']!=doc:
                        raise ValueError('activity_document_conflict')
                    cleaned.append(value)
                except ValueError as error:
                    rejected.append({'context_id':row.context_id,'activity_id':aid,'reason':str(error),'level':'activity'})
            tables[endpoint]=cleaned
        grouped={}
        for endpoint,records in tables.items():
            df=pd.DataFrame(records,columns=['smiles','value_uM','activity_id','document_chembl_id'])
            grouped[endpoint]={s:(float(np.exp(np.log(d.value_uM).mean())),d.activity_id.tolist()) for s,d in df.groupby('smiles')}
        for smiles in sorted(set(grouped['IC50'])&set(grouped['CC50'])):
            ic,ii=grouped['IC50'][smiles]
            cc,ci=grouped['CC50'][smiles]
            si=cc/ic
            if not math.isfinite(si):
                rejected.append({'context_id':row.context_id,'reason':'nonfinite_ratio','level':'pair'})
                continue
            rows.append({'smiles':smiles,'IC50_uM':ic,'CC50_uM':cc,'SI':si,'doi':doi})
            provenance.append({'output_row':len(rows),'context_id':row.context_id,'smiles':smiles,
                'target_chembl_id':row.target_chembl_id,'cell_line':row.cell_line,
                'ic50_hours':row.ic50_hours,'cc50_hours':row.cc50_hours,
                'pairing_policy':row.get('pairing_policy','equal_duration'),
                'ic50_assay_id':row.ic50_assay_id,'cc50_assay_id':row.cc50_assay_id,
                'ic50_activity_ids':json.dumps(ii),'cc50_activity_ids':json.dumps(ci),
                'document_chembl_id':doc,'evidence_url':row.evidence_url,
                'replicate_aggregation':'geometric mean within the same approved assay only'})
    return (pd.DataFrame(rows,columns=['smiles','IC50_uM','CC50_uM','SI','doi']),
            pd.DataFrame(provenance,columns=PROVENANCE_COLUMNS),pd.DataFrame(rejected,columns=REJECTION_COLUMNS))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    discover=sub.add_parser('discover')
    discover.add_argument('--target-id',required=True)
    discover.add_argument('--out',required=True)
    collect=sub.add_parser('collect')
    collect.add_argument('--pairs',required=True)
    collect.add_argument('--out',required=True)
    parser.add_argument('--cache',default='work/chembl_cache_v2')
    args=parser.parse_args()
    client=ChEMBL(args.cache)
    output=Path(args.out)
    output.parent.mkdir(parents=True,exist_ok=True)
    if args.command=='discover':
        if not re.fullmatch(r'CHEMBL\d+',args.target_id):
            raise ValueError('Use an actual ChEMBL target ID')
        records=list(client.pages('assay.json','assays',target_chembl_id=args.target_id))
        pd.DataFrame(records).to_csv(output,index=False)
    else:
        dataset,provenance,rejected=collect_pairs(client,pd.read_csv(args.pairs))
        dataset.to_csv(output,index=False)
        provenance.to_csv(output.with_suffix('.provenance.csv'),index=False)
        rejected.to_csv(output.with_suffix('.rejected.csv'),index=False)
        print(json.dumps({'accepted_pairs':len(dataset),'rejected_records_or_contexts':len(rejected)}))
    output.with_suffix('.requests.json').write_text(json.dumps(client.log,indent=2))


if __name__=='__main__':main()
