"""Publication-verified, cell-matched VACV SI extraction from ChEMBL endpoints."""
import argparse,json,hashlib,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--cache',default='work/literature_cache');a=p.parse_args();root=Path(a.root);cache=Path(a.cache);cache.mkdir(parents=True,exist_ok=True)
 url='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10290961/fullTextXML';path=cache/'PMC10290961.xml'
 if not path.exists():
  with urllib.request.urlopen(url,timeout=60) as f:path.write_bytes(f.read())
 xml=ET.fromstring(path.read_bytes());table=xml.find(".//table-wrap[@id='tbl1']")
 expected=[]
 for row in table.findall('.//tr')[2:]:
  vals=[' '.join(''.join(e.itertext()).split()) for e in row]
  if len(vals)!=10:continue
  for cell,offset in [('Vero',4),('A549',7)]:
   try:ic,cc,si=map(float,vals[offset:offset+3])
   except ValueError:continue
   expected.append(dict(doi='10.1016/j.ejmech.2023.115601',compound=vals[0],cell=cell,ic=ic,cc=cc,reported_si=si,table='Table 1',publication_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC10290961/'))
 # Reviewed numeric means from Table 1, PMID 33479623. Bounds and NA excluded.
 rows=[('40',1908.6,1.7,1123),('41',750.2,1.8,417),('42',319,12.3,26),('44',18.9,5.9,3),('46',18.1,3.6,5),('47',316.7,10.6,30),('48',316.9,14.4,22),('49',316.9,2.5,126),('53',1225.9,4.6,267),('54',372.5,4.4,85),('55',13.7,2.8,5),('56',316.7,2.6,122),('58a',37,3.1,12),('58b',329.5,14.8,22),('61a',7.9,.7,11),('61b',811.5,2.5,325),('63a',164.2,2.3,71),('63b',95.8,.6,160),('Cidofovir',475.3,40,12),('ST-246',1276,.01,127600)]
 for name,cc,ic,si in rows:expected.append(dict(doi='10.1039/d0md00108b',compound=name,cell='Vero',ic=ic,cc=cc,reported_si=si,table='Table 1',publication_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC7651861/'))
 ep=pd.read_csv(root/'data/external/external_endpoints.csv');ep['doi_key']=ep.doi.str.lower();accepted=[];review=[]
 for e in expected:
  d=ep[(ep.doi_key==e['doi']) & (ep.cell_line.str.lower()==e['cell'].lower()) & ep.validation_status.eq('valid_endpoint')]
  ic=d[d.is_vacv_assay & d.endpoint.eq('IC50') & np.isclose(d.IC50_uM,e['ic'],rtol=.015,atol=1e-8)]
  cc=d[d.endpoint.eq('CC50') & np.isclose(d.CC50_uM,e['cc'],rtol=.015,atol=1e-8)]
  pairs=ic.merge(cc,on='molecule_chembl_id',suffixes=('_ic','_cc'))
  mids=pairs.molecule_chembl_id.unique()
  review.append(dict(**e,matched_molecules=len(mids)))
  if len(mids)!=1:continue
  r=pairs.iloc[0]
  accepted.append(dict(SMILES=r.SMILES_ic,IC50_uM=e['ic'],CC50_uM=e['cc'],SI=e['cc']/e['ic'],SI_reported=e['reported_si'],doi=e['doi'],cell_line=e['cell'],compound_id=e['compound'],endpoint='cell_based_VACV_IC50',table=e['table'],molecule_chembl_id=mids[0],antiviral_activity_id=int(r.activity_id_ic),cytotoxicity_activity_id=int(r.activity_id_cc),source_url=r.source_url_ic,cyto_source_url=r.source_url_cc,publication_url=e['publication_url'],validation_status='publication_table_and_cell_matched',duration_policy='Author-defined pairing; equal assay duration not required'))
 out=root/'data/external_curated';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(accepted).to_csv(out/'vacv_si.csv',index=False);pd.DataFrame(review).to_csv(out/'table_match_audit.csv',index=False)
 summary={'reviewed_sources':2,'reviewed_exact_table_rows':len(expected),'matched_rows':len(accepted),'unique_molecules':len({x['molecule_chembl_id'] for x in accepted}),'policy':'Same publication, molecule and cell; cell-based VACV IC50; publication-defined SI pair; bounds excluded. Structures attributed through ChEMBL and numeric table matching, not independent structure-image verification.','xml_url':url,'xml_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 (out/'curation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
