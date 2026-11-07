"""Build a source-data snapshot with explicit provenance and component licenses."""
from pathlib import Path
import argparse,hashlib,json,zipfile
import pandas as pd
import train_v7 as V7
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/external_research';OUT=ROOT/'results/external_20261107'

def sha(raw):return hashlib.sha256(raw).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache-base',default='work/external_source_cache');a=ap.parse_args()
    cache=ROOT/a.cache_base
    records=[
      dict(source='ChEMBL37',url='https://www.ebi.ac.uk/chembl/api/data/',access='official paginated API + previous snapshot',license='CC-BY-SA-3.0',license_url='https://chembl.gitbook.io/chembl-interface-documentation/about/terms-of-use',downloaded_rows=9606,eligible_endpoint_rows=2160,verified_new_SI=0,new_structures_vs_previous_project=0,new_scaffolds_vs_previous_project=0,evaluated_rows=384,final_used_new_rows=0,status='numeric auxiliary labels previously archived; transfer not accepted'),
      dict(source='PubChem',url='https://pubchem.ncbi.nlm.nih.gov/rest/pug/',access='ESearch + PUG REST AID CSV/SID lists',license='PubChem/depositor terms; no universal license asserted',license_url='https://pubchem.ncbi.nlm.nih.gov/docs/downloads',downloaded_rows=390,eligible_endpoint_rows=208,verified_new_SI=0,new_structures_vs_previous_project=98,new_scaffolds_vs_previous_project=31,evaluated_rows=208,final_used_new_rows=0,status='98 structures in full pilot;63 exact-numeric structures evaluated; transfer not accepted'),
      dict(source='PMC10995631',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC10995631/#T2',access='Europe PMC official XML + exact PubChem identity mapping',license='CC-BY-4.0',license_url='https://creativecommons.org/licenses/by/4.0/',downloaded_rows=24,eligible_endpoint_rows=24,verified_new_SI=12,new_structures_vs_previous_project=8,new_scaffolds_vs_previous_project=3,evaluated_rows=12,final_used_new_rows=0,status='author-paired physical SI, HeLa; addition not accepted'),
      dict(source='PMC9533862',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC9533862/',access='open pilot metadata/table review',license='conditional publisher research reuse; not cleared',license_url='',downloaded_rows=8,eligible_endpoint_rows=0,verified_new_SI=0,new_structures_vs_previous_project=None,new_scaffolds_vs_previous_project=None,evaluated_rows=0,final_used_new_rows=0,status='structure forms/reuse eligibility unresolved; no labels accepted'),
      dict(source='TDC_HTS',url='https://tdcommons.ai/single_pred_tasks/hts/',access='official dataset documentation only',license='dataset specific',license_url='',downloaded_rows=0,eligible_endpoint_rows=0,verified_new_SI=0,new_structures_vs_previous_project=0,new_scaffolds_vs_previous_project=0,evaluated_rows=0,final_used_new_rows=0,status='documented HIV/SARS-CoV2 and other-target classification tasks; no SI VACV labels'),
      dict(source='DeepChem_MoleculeNet',url='https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html',access='official dataset documentation only',license='dataset specific',license_url='',downloaded_rows=0,eligible_endpoint_rows=0,verified_new_SI=0,new_structures_vs_previous_project=0,new_scaffolds_vs_previous_project=0,evaluated_rows=0,final_used_new_rows=0,status='tool/dataset catalog; no compatible new SI labels downloaded')]
    pd.DataFrame(records).to_csv(OUT/'source_registry.csv',index=False)
    bank=pd.read_csv(DATA/'training_bank.csv')
    physical=bank[bank.task.eq('SI')].copy()
    # Same numeric results across different DOI are candidates, not automatically
    # proven duplicate measurements. They are never advertised as new data.
    suspicious=physical[physical.duplicated(['tautomer_parent','value'],keep=False)].sort_values(['tautomer_parent','value'])
    suspicious.to_csv(OUT/'numeric_collisions_review.csv',index=False)
    same_measurement=physical.duplicated(['tautomer_parent','value','doi','cell'],keep=False)
    pmc=set(bank.loc[bank.source.eq('PMC10995631'),'tautomer_parent'])
    pub=pd.read_csv(DATA/'pubchem/auxiliary_endpoints.csv')
    pkeys={V7.source_key(s) for s in pub.SMILES}
    chem=pd.read_csv(DATA/'chembl/endpoint_bank.csv');ckeys={V7.source_key(s) for s in chem.SMILES}
    dedup={'same_context_exact_physical_collision_rows':int(same_measurement.sum()),
      'same_structure_value_review_rows':len(suspicious),
      'PMC_PubChem_shared_structures':len(pmc&pkeys),'ChEMBL_PubChem_shared_structures':len(ckeys&pkeys),
      'PMC_new_DOI_already_in_legacy':bool(set(physical.loc[physical.source.str.startswith('legacy_'),'doi'].dropna())&set(physical.loc[physical.source.eq('PMC10995631'),'doi'])),
      'note':'Different DOI can still hide repeated reporting. Unresolved numeric coincidences stay on review; no universal cross-source independence is claimed.'}
    (OUT/'cross_source_dedup_audit.json').write_text(json.dumps(dedup,indent=2)+'\n')
    (DATA/'README.md').write_text('''# Open-data snapshot for SI regression\n\nThe collection contains separately identified physical SI and auxiliary endpoints.\n\n- pmc/training_si.csv:12 author-paired CC50/IC50 SI,12 structures, HeLa; CC-BY4.0.\n- chembl/endpoint_bank.csv:2160 exact numerical auxiliary endpoint rows; ChEMBL37, CC-BY-SA3.0. These were already in the earlier source archive.\n- pubchem/auxiliary_endpoints.csv:390 rows;208 exact positive numerical endpoints, the remainder includes censored records.98 structures total. PubChem/depositor terms apply.\n- training_bank.csv:657 evaluated rows =53 retained legacy physical+12 PMC physical+384 ChEMBL auxiliary+208 PubChem auxiliary. Units:SI is dimensionless, other outputs use micromolar. All numeric training rows have exact relation.\n\nNo external test-structure numerical labels enter this bank. Task names preserve endpoint/source context; auxiliary measurements do not become SI or code10000. Missing values are not zero labels. The final model rejected new-source additions and uses only the53 retained legacy physical records.\n\nThe ZIP additionally contains inclusion/exclusion audits, selected numerical table evidence, allowed raw API/numeric-table responses, and SHA256SUMS. Full article XML and PubChem description/protocol text remain in the private work cache; their URLs and hashes remain in manifests. Source licenses apply to their separate components; no universal license is asserted for this collection. Pretrained encoder weights are absent.\n\nSee docs/external_research_20261107.md and results/external_20261107/source_registry.csv for counts, restrictions, source effects and reproduction commands.\n''')
    (DATA/'pubchem/README.md').write_text('''# PubChem pilot\n\nOfficial ESearch found53 non-ChEMBL AIDs.13 dose-response AIDs and explicit VACV rows of mixed panel602175 were downloaded; SID/CSV completeness was checked.390 endpoint rows,98 structures and31 scaffolds are new to the prior project. There are208 exact positive numeric values, including199 AC50 and9 TC50, and no approved SI.9 antiviral IC50 values are censored.\n\nReporter AC50, cytotoxicity AC50 and TC50 stay separate. Active/Inactive do not become organizer code10000. No test-structure labels remain. DOI is blank unless linked to a primary measurement publication unambiguously.\n\nAccess:https://pubchem.ncbi.nlm.nih.gov/rest/pug/ ; https://eutils.ncbi.nlm.nih.gov/entrez/eutils/ . Terms:https://pubchem.ncbi.nlm.nih.gov/docs/downloads and depositor-specific restrictions; no universal CC license inferred.\n\nRun:python src/collect_pubchem_research.py --cache work/external_source_cache/pubchem . Rate2.5 requests/sec, retry, cached response SHA/date, search pagination, and per-assay SID counts are implemented. Large primary screen not downloaded. Full API description/protocol responses stay in work cache; only numeric tables and non-operational metadata are exported.\n''')
    entries={}
    for p in sorted(DATA.rglob('*')):
        if p.is_file() and p.suffix!='.zip' and p.name!='archive_manifest.json':entries[str(p.relative_to(ROOT))]=p.read_bytes()
    # New ChEMBL raw status/target/activity responses carry only database fields;
    # no full articles or experimental protocols are fetched into this archive.
    for r in json.loads((OUT/'chembl_raw_response_index.json').read_text()):
        p=Path(r['cache_file'])
        if p.exists():
            raw=p.read_bytes();assert sha(raw)==r['sha256'];entries['raw/'+r['archive_path']]=raw
    for r in json.loads((DATA/'pubchem/request_manifest.json').read_text()):
        if '/description/' in r['url']:continue
        p=cache/'pubchem'/r['cache_file']
        if p.exists():
            raw=p.read_bytes();assert sha(raw)==r['sha256'];entries['raw/pubchem/'+p.name]=raw
    entries['README.txt']=b'Source-separated numeric snapshot. See data/external_research/README.md and SHA256SUMS. No article Methods, PubChem protocols or encoder checkpoints included.\n'
    checksum=''.join(sha(raw)+'  '+name+'\n' for name,raw in sorted(entries.items()))
    entries['SHA256SUMS']=checksum.encode()
    dest=DATA/'open_data_bundle.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,raw in sorted(entries.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,11,7,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,raw,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    manifest={'archive':str(dest.relative_to(ROOT)),'sha256':sha(dest.read_bytes()),'bytes':dest.stat().st_size,
       'members':len(entries),'raw_api_members':sum(n.startswith('raw/') for n in entries),'uncompressed_bytes':sum(len(v) for v in entries.values()),
       'member_sha256':{n:sha(v) for n,v in entries.items()}}
    (DATA/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print({k:v for k,v in manifest.items() if k!='member_sha256'})

if __name__=='__main__':main()
