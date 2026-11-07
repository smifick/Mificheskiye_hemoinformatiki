"""Auditable extraction of published numerical SI tables, not assay protocols.

Only reviewed table schemas and exact name -> unique CID -> exact synonym ->
SMILES/InChIKey correspondence are accepted. Raw full text stays in the cache.
The public evidence export contains attribution and required table columns.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import quote, urlencode, urlparse
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from rdkit import Chem
import train_v5 as T

API = 'https://www.ebi.ac.uk/europepmc/webservices/rest/'
PUG = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/'
SOURCES = {
 'PMC10995631': {'doi': '10.1099/jgv.0.001971', 'table_id': 'T2', 'table': 'Table 2',
                 'cell_line': 'HeLa', 'strain': 'Western Reserve',
                 'license': 'CC-BY-4.0', 'units': 'nM', 'schema': 'cardiac'},
}
SI_COLUMNS = ['SMILES','IC50_uM','CC50_uM','SI','SI_reported','doi','cell_line',
              'strain','compound_id','pubchem_cid','InChIKey','endpoint','si_definition',
              'table','source_row','source_url','structure_url','license',
              'validation_status','parent_key','scaffold','test_structure_overlap',
              'train_structure_overlap','existing_external_structure','new_to_project_structure']


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


class CachedHTTP:
    """Persistent SHA-checked cache, <=2 req/s, retry temporary failures."""
    def __init__(self, directory, refresh=False, interval=.55, timeout=45):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.refresh=refresh;self.interval=interval;self.timeout=timeout
        self.session=requests.Session()
        self.session.headers['User-Agent']='Published-SI-table-audit/1.0'
        self.last=0.;self.manifest=[]

    def get(self,url):
        p=urlparse(url)
        if p.scheme!='https' or p.netloc not in {'www.ebi.ac.uk','pubchem.ncbi.nlm.nih.gov'}:
            raise ValueError('Unsupported official source URL')
        key=hashlib.sha256(url.encode()).hexdigest()
        file=self.directory/(key+'.raw');meta_file=self.directory/(key+'.json')
        cached=file.exists() and meta_file.exists() and not self.refresh
        if cached:
            raw=file.read_bytes();meta=json.loads(meta_file.read_text())
            if hashlib.sha256(raw).hexdigest()!=meta['sha256']:
                raise ValueError('Cached response SHA256 mismatch')
        else:
            for attempt in range(5):
                delay=self.interval-(time.monotonic()-self.last)
                if delay>0:time.sleep(delay)
                self.last=time.monotonic()
                try:
                    response=self.session.get(url,timeout=(15,self.timeout))
                    if response.status_code in {408,429,500,502,503,504}:
                        if attempt==4:response.raise_for_status()
                        retry=response.headers.get('Retry-After','')
                        time.sleep(min(float(retry) if retry.isdigit() else 2**attempt,16))
                        continue
                    response.raise_for_status();raw=response.content;break
                except (requests.Timeout,requests.ConnectionError):
                    if attempt==4:raise
                    time.sleep(2**attempt)
            meta={'url':url,'retrieved_utc':now(),'status':response.status_code,
                  'content_type':response.headers.get('Content-Type',''),
                  'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
            temp=file.with_suffix('.part');temp.write_bytes(raw);temp.replace(file)
            write_json(meta_file,meta)
        self.manifest.append({**meta,'cache_file':str(file),'cache_hit':cached,'accessed_utc':now()})
        return raw

    def json(self,url):
        return json.loads(self.get(url))


def text(e):
    return ' '.join(''.join(e.itertext()).split()) if e is not None else ''


def search(client,query,page_size=100):
    cursor='*';rows=[];seen=set();expected=None
    while True:
        data=client.json(API+'search?'+urlencode(dict(query=query,format='json',
                         pageSize=page_size,cursorMark=cursor,resultType='core')))
        if expected is None:expected=int(data['hitCount'])
        if int(data['hitCount'])!=expected:raise RuntimeError('Search corpus changed during pagination')
        batch=data.get('resultList',{}).get('result',[])
        if not batch:break
        for r in batch:
            key=(r.get('source'),r.get('id'))
            if key in seen:raise RuntimeError('Duplicate search result')
            seen.add(key)
            rows.append({k:r.get(k) for k in ['source','id','pmcid','doi','title','pubYear',
                         'isOpenAccess','inEPMC','hasSuppl','firstPublicationDate']})
        nxt=data.get('nextCursorMark')
        if not nxt or nxt==cursor:break
        cursor=nxt
    if len(rows)!=expected:raise RuntimeError(f'Incomplete pagination: {len(rows)} != {expected}')
    return rows


def number(value,unit='uM'):
    s=' '.join(value.replace('µ','u').replace('μ','u').split()).strip();relation='='
    for symbol in ['>=','<=','>','<','≥','≤']:
        if s.startswith(symbol):relation=symbol;s=s[len(symbol):].strip();break
    m=re.match(r'^(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b',s)
    if not m:return None,'missing',None
    value=float(m.group(1))
    if not math.isfinite(value) or value<=0:return None,'invalid',None
    units=re.search(r'\b(nM|uM|mM)\b',s);unit=units.group(1) if units else unit
    if unit not in {'nM','uM','mM'}:raise ValueError('Unsupported concentration units')
    return value,relation,value*{'nM':.001,'uM':1.,'mM':1000.}[unit]


def extract_rows(raw,pmcid,source):
    root=ET.fromstring(raw)
    doi=next((e.text for e in root.findall('.//article-meta/article-id')
              if e.attrib.get('pub-id-type')=='doi'),'')
    if doi.lower()!=source['doi']:raise ValueError('Pinned PMCID/DOI mismatch')
    table=root.find(f".//table-wrap[@id='{source['table_id']}']")
    if table is None:raise ValueError('Reviewed table is missing')
    headers=[text(c) for c in table.findall('.//tr')[0]]
    if headers[:4]!=['Compound','IC50','CC50','SI']:raise ValueError('Reviewed table schema changed')
    permission=text(root.find('.//article-meta/permissions'))
    if 'Creative Commons Attribution' not in permission:raise ValueError('Expected attribution license missing')
    rows=[]
    for i,row in enumerate(table.findall('.//tbody/tr'),1):
        cells=[text(c) for c in row]
        if len(cells)!=6:raise ValueError('Unexpected concentration table row')
        name,ic,cc,si=cells[:4]
        _,ir,iv=number(ic,source['units']);_,cr,cv=number(cc,source['units'])
        sv,sr,_=number(si)
        ratio=cv/iv if iv and cv else None
        decision='exact_author_paired_IC50_CC50' if ir==cr==sr=='=' and ratio is not None else 'excluded_bound_or_missing'
        if decision.startswith('exact') and abs(ratio-sv)/max(ratio,sv)>.03:
            decision='review_reported_SI_disagrees_over_3pct'
        rows.append(dict(pmcid=pmcid,doi=doi,table=source['table'],source_row=i,
                         compound_id=name,IC50_raw=ic,CC50_raw=cc,SI_raw=si,
                         IC50_relation=ir,CC50_relation=cr,SI_relation=sr,
                         IC50_uM=iv,CC50_uM=cv,SI_reported=sv,SI=ratio,
                         cell_line=source['cell_line'],strain=source['strain'],
                         decision=decision,license=source['license'],
                         source_url=f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/#{source["table_id"]}'))
    evidence=ET.Element('published-table-evidence',pmcid=pmcid,doi=doi)
    ET.SubElement(evidence,'title').text=text(root.find('.//article-title'))
    ET.SubElement(evidence,'license_url').text='https://creativecommons.org/licenses/by/4.0/'
    ET.SubElement(evidence,'fulltext_source_url').text=API+pmcid+'/fullTextXML'
    for row in rows:
        item=ET.SubElement(evidence,'row',number=str(row['source_row']))
        for k in ['compound_id','IC50_raw','CC50_raw','SI_raw','cell_line','strain']:
            ET.SubElement(item,k).text=row[k]
    return rows,evidence,permission


def exact_structure(client,name):
    ids=client.json(PUG+'compound/name/'+quote(name,safe='')+'/cids/JSON').get('IdentifierList',{}).get('CID',[])
    if len(ids)!=1:raise ValueError(f'Nonunique exact-name identity: {len(ids)} CIDs')
    cid=ids[0]
    synonyms=client.json(PUG+f'compound/cid/{cid}/synonyms/JSON')['InformationList']['Information'][0]['Synonym']
    if name.casefold() not in {s.casefold() for s in synonyms}:
        raise ValueError('Publication name absent from exact CID synonyms')
    url=PUG+f'compound/cid/{cid}/property/IsomericSMILES,InChIKey/JSON'
    properties=client.json(url)['PropertyTable']['Properties']
    if len(properties)!=1 or properties[0]['CID']!=cid:raise ValueError('CID property response mismatch')
    prop=properties[0];smiles=prop.get('SMILES') or prop.get('IsomericSMILES')
    mol=Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:raise ValueError('Invalid source SMILES')
    if Chem.MolToInchiKey(mol)!=prop['InChIKey']:raise ValueError('SMILES/InChIKey mismatch')
    return dict(pubchem_cid=cid,SMILES=Chem.MolToSmiles(mol),InChIKey=prop['InChIKey'],
                structure_url=url,identity_validation='unique_exact_name_CID_exact_synonym_and_InChIKey',
                structure_license='https://pubchem.ncbi.nlm.nih.gov/docs/copyright')


def structure_keys(smiles):
    mol=T.std(smiles)
    return T.taut_key(mol),T.gscaf(mol)


def review_conditional_source(client,out):
    """A negative pilot: inspect only the numeric table and license metadata.

    Published facts may be scientifically relevant, but do not silently label
    a conditional pandemic-reuse permission as permanent CC-BY redistribution.
    This source is not admitted into the public training dataset in this run.
    """
    pmcid='PMC9533862';url=API+pmcid+'/fullTextXML'
    raw=client.get(url);tree=ET.fromstring(raw)
    permission=text(tree.find('.//article-meta/permissions'))
    table=tree.find(".//table-wrap[@id='tbl0010']")
    if table is None:raise ValueError('Negative-pilot concentration table missing')
    exact=0;rows=[];previous_name='';previous_cc=''
    for i,row in enumerate(table.findall('.//tbody/tr'),1):
        cells=[text(c) for c in row]
        if len(cells)!=4:raise ValueError('Negative-pilot table schema changed')
        name,ic,cc,si=cells
        if not name and previous_name=='Nigericin':name=previous_name;cc=previous_cc
        previous_name,previous_cc=name,cc
        _,ir,iv=number(ic);_,cr,cv=number(cc);sv,sr,_=number(si)
        complete=ir==cr==sr=='=' and iv is not None and cv is not None
        exact+=int(complete)
        rows.append(dict(pmcid=pmcid,doi='10.1016/j.antiviral.2010.10.001',table='Table 1',
            source_row=i,compound_id=name,numeric_exact_pair=complete,
            disposition='not_admitted_conditional_license_not_revalidated',
            source_url=f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/#tbl0010'))
    conditional=('for as long as' in permission and 'research re-use' in permission)
    record=dict(source='Europe PMC / negative pilot',pmcid=pmcid,
        doi='10.1016/j.antiviral.2010.10.001',url=url,
        license='Conditional Elsevier research-reuse permission; not permanent CC-BY',
        license_url=f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/',
        table='Table 1',table_rows=len(rows),exact_numeric_pairs=exact,
        verified_SI_rows=0,training_eligible_rows=0,new_to_project_structures=0,
        new_to_organizer_scaffolds=0,conditional_permission_detected=conditional,
        disposition='Not admitted: current continuing permission and precise chemical forms not established; no model labels generated')
    pd.DataFrame(rows).to_csv(out/'negative_pilot_rows.csv',index=False)
    write_json(out/'negative_pilot_summary.json',record)
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.')
    p.add_argument('--out',default='data/external_research/pmc')
    p.add_argument('--cache',default='work/external_source_cache/pmc')
    p.add_argument('--refresh',action='store_true')
    args=p.parse_args();root=Path(args.root).resolve();out=root/args.out
    out.mkdir(parents=True,exist_ok=True)
    client=CachedHTTP(root/args.cache,args.refresh)
    query='(TITLE_ABS:"vaccinia" AND (TITLE_ABS:"selectivity" OR TITLE_ABS:"cytotoxicity")) AND OPEN_ACCESS:Y'
    discovery=search(client,query)
    pd.DataFrame(discovery).to_csv(out/'literature_discovery.csv',index=False)
    rows=[];registry=[];identity_log=[];review=[]
    for pmcid,source in SOURCES.items():
        url=API+pmcid+'/fullTextXML'
        raw=client.get(url);table,evidence,permission=extract_rows(raw,pmcid,source)
        ET.ElementTree(evidence).write(out/(pmcid+'_table_evidence.xml'),encoding='utf-8',xml_declaration=True)
        rows.extend(table)
        registry.append(dict(source='Europe PMC / primary publisher table',pmcid=pmcid,
            doi=source['doi'],url=url,license=source['license'],
            license_url='https://creativecommons.org/licenses/by/4.0/',
            table=source['table'],table_rows=len(table),
            exact_numeric_pairs=sum(r['decision'].startswith('exact') for r in table),
            endpoint='cell-based VACV IC50 and author-paired CC50',
            cell_line=source['cell_line'],strain=source['strain'],
            cell_evidence='Table2 references Figure1; numerical response curves are HeLa, not later A-RPE-19 assays',
            structure_access='PubChem exact name -> unique CID -> exact synonym -> SMILES/InChIKey',
            limitation='Named identity validated; supplier-lot stereochemical purity not independently established'))
    pd.DataFrame(rows).to_csv(out/'table_rows.csv',index=False)
    # Concentrations from organizer train/test are never loaded.
    train=pd.read_csv(root/'data/raw/train.csv',usecols=['SMILES'])
    test=pd.read_csv(root/'data/raw/test.csv',usecols=['SMILES'])
    tr_keys={structure_keys(s)[0] for s in train.SMILES}
    te_keys={structure_keys(s)[0] for s in test.SMILES}
    org_scaffolds={structure_keys(s)[1] for s in list(train.SMILES)+list(test.SMILES)}
    known=pd.read_csv(root/'data/external/external_endpoints.csv',usecols=['SMILES','doi'])
    known_chemical_keys=[structure_keys(s) for s in known.SMILES.dropna().unique()
                         if Chem.MolFromSmiles(s) is not None]
    known_structures={k for k,g in known_chemical_keys}
    all_known_scaffolds={g for k,g in known_chemical_keys}|org_scaffolds
    existing=pd.read_csv(root/'data/external/external_si_database.csv')
    old_pairs={(structure_keys(r.SMILES)[0],str(r.doi).lower())
               for r in existing.itertuples() if isinstance(r.SMILES,str)}
    exact_rows=[r for r in rows if r['decision'].startswith('exact')]
    identities={}
    for name in sorted({r['compound_id'] for r in exact_rows}):
        try:
            identities[name]=exact_structure(client,name)
            identity_log.append({'compound_id':name,**identities[name]})
        except (ValueError,KeyError,requests.HTTPError) as exc:
            identities[name]=None
            review.append(dict(compound_id=name,decision='review_identity',reason=str(exc)))
    accepted=[];excluded=[]
    for row in exact_rows:
        ident=identities[row['compound_id']]
        if ident is None:continue
        key,scaffold=structure_keys(ident['SMILES'])
        record={**row,**ident,'endpoint':'IC50',
                'si_definition':'CC50/IC50, author-paired cell-based VACV endpoints',
                'validation_status':'reviewed_table_exact_pair_unique_structure',
                'parent_key':key,'scaffold':scaffold,'test_structure_overlap':key in te_keys,
                'train_structure_overlap':key in tr_keys,
                'existing_external_structure':key in known_structures,
                'new_to_project_structure':key not in tr_keys|te_keys|known_structures}
        if key in te_keys:excluded.append({**record,'exclusion':'test_structure'})
        if (key,row['doi'].lower()) in old_pairs:excluded.append({**record,'exclusion':'duplicate_source_measurement'})
        accepted.append(record)
    verified=pd.DataFrame(accepted) if accepted else pd.DataFrame(columns=SI_COLUMNS)
    if len(verified):
        training=verified[~verified.test_structure_overlap.astype(bool)].copy()
        training=training[~training.apply(lambda r:(r.parent_key,r.doi.lower()) in old_pairs,axis=1)]
    else:training=verified.copy()
    verified.to_csv(out/'verified_si.csv',index=False)
    training.to_csv(out/'training_si.csv',index=False)
    pd.DataFrame(identity_log).to_csv(out/'structure_map.csv',index=False)
    pd.DataFrame(excluded,columns=list(verified.columns)+['exclusion']).to_csv(out/'excluded_si.csv',index=False)
    pd.DataFrame(review,columns=['compound_id','decision','reason']).to_csv(out/'review.csv',index=False)
    measurements=[]
    for row in rows:
        for endpoint in ['IC50','CC50']:
            measurements.append({k:row[k] for k in ['pmcid','doi','table','source_row','compound_id',
                                'cell_line','strain','source_url','license']}|
                                {'endpoint':endpoint,'value_uM':row[endpoint+'_uM'],
                                 'relation':row[endpoint+'_relation'],'raw_value':row[endpoint+'_raw']})
    pd.DataFrame(measurements).to_csv(out/'measurements.csv',index=False)
    for r in registry:
        d=training[training.doi.eq(r['doi'])]
        r.update(training_eligible_rows=len(d),training_structures=d.parent_key.nunique(),
                 new_to_project_structures=d[d.new_to_project_structure.astype(bool)].parent_key.nunique(),
                 new_to_organizer_scaffolds=len(set(d.scaffold)-org_scaffolds),
                 new_to_project_scaffolds=len(set(d.scaffold)-all_known_scaffolds))
    negative=review_conditional_source(client,out)
    pd.DataFrame(registry+[negative]).to_csv(out/'source_registry.csv',index=False)
    summary=dict(retrieved_utc=now(),search_query=query,bibliographic_hits=len(discovery),
        reviewed_fulltext_sources=len(SOURCES),negative_pilot_sources=1,table_rows=len(rows),
        individual_endpoint_rows=len(measurements),exact_numeric_pairs_before_identity=len(exact_rows),
        verified_SI_rows=len(verified),unique_resolved_structures=int(verified.parent_key.nunique()),
        training_eligible_rows=len(training),training_structures=int(training.parent_key.nunique()),
        training_scaffolds=int(training.scaffold.nunique()),
        new_to_project_structures_training=int(training[training.new_to_project_structure.astype(bool)].parent_key.nunique()),
        new_to_organizer_scaffolds_training=len(set(training.scaffold)-org_scaffolds),
        new_to_project_scaffolds_training=len(set(training.scaffold)-all_known_scaffolds),
        test_overlapping_rows_excluded=int(verified.test_structure_overlap.sum()),
        ambiguous_identities=len(review),
        policy='Exact author-defined IC50/CC50 pairs; bounds and missing results excluded; exact-name unique CID plus exact CID synonym and SMILES/InChIKey agreement; no external test-structure labels in training; caller must additionally remove validation structures/scaffolds per fold.',
        source_version='Europe PMC live corpus; PMCID, DOI and response SHA256 pinned',
        licenses={'PMC10995631':'CC-BY-4.0','PubChem_structure':'https://pubchem.ncbi.nlm.nih.gov/docs/copyright'},
        raw_response_policy='Full XML stays in cache; exported evidence contains selected numeric columns, attribution and license URL only.',
        input_SHA256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in [root/'data/raw/train.csv',root/'data/raw/test.csv']})
    write_json(out/'collection_summary.json',summary)
    write_json(out/'request_manifest.json',client.manifest)
    result=root/'results/external_20261107';result.mkdir(parents=True,exist_ok=True)
    write_json(result/'pmc_collection_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
