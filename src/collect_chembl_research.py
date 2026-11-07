"""Audit the frozen ChEMBL export and extend organism-level endpoint coverage.

The numeric auxiliary endpoints remain separate tasks. This collector does not
approve arbitrary concentration pairs as SI and does not assign Kaggle's code
10000 to any external record. No external label of an official test structure
is eligible for subsequent training.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile

import pandas as pd
from rdkit import Chem, RDLogger

import download_external as D
import train_v5 as T

RDLogger.DisableLog('rdApp.*')
LICENSE_URL = 'https://chembl.gitbook.io/chembl-interface-documentation/about#data-licensing'
LICENSE_CODE = 'CC-BY-SA-3.0'


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


class PacedClient(D.Client):
    """Single-worker requests, persistent raw-response cache, bounded retries."""
    def __init__(self, *args, minimum_interval=.5, **kwargs):
        super().__init__(*args, workers=1, **kwargs)
        self.minimum_interval = minimum_interval
        self.last = 0.

    def get(self, url, use_cache=True):
        path = self.cache / (sha_bytes(url.encode()) + '.json')
        if not (use_cache and not self.refresh and path.exists()):
            delay = self.minimum_interval - (time.monotonic() - self.last)
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()
        return super().get(url, use_cache)


def graph_keys(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    parent = T.std(smiles)
    if parent is None:
        return None
    return dict(canonical_smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
                InChIKey=Chem.MolToInchiKey(mol),
                parent_smiles=Chem.MolToSmiles(parent, isomericSmiles=True),
                tautomer_parent=T.taut_key(parent), scaffold=T.gscaf(parent))


def si_definition(description):
    """Recognize explicit definitions only; ambiguous prose stays in review."""
    text = re.sub(r'\s+', '', str(description).lower())
    if re.search(r'cc50.*[/].*ic50', text):
        return 'CC50/IC50'
    if re.search(r'cc50.*[/].*ec50', text):
        return 'CC50/EC50'
    return 'unverified_definition'


def reference_sets(root):
    def keys(path):
        return {T.taut_key(T.std(s)) for s in pd.read_csv(path, usecols=['SMILES']).SMILES}
    train = keys(root / 'data/raw/train.csv')
    test = keys(root / 'data/raw/test.csv')
    curated = pd.read_csv(root / 'data/external_curated/vacv_si.csv')
    return train, test, {T.taut_key(T.std(s)) for s in curated.SMILES}


def collect(args):
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = root / 'data/chembl_package/vacv_chembl_data.zip'
    with zipfile.ZipFile(archive) as z:
        raw = json.loads(z.read('chembl_export/raw_activities.json'))
        assays = {a['assay_chembl_id']: a for a in json.loads(z.read('chembl_export/all_assays.json'))}
        targets = {t['target_chembl_id']: t for t in json.loads(z.read('chembl_export/targets.json'))}
        documents = {d['document_chembl_id']: d for d in json.loads(z.read('chembl_export/documents.json'))}
        frozen = pd.read_csv(io.BytesIO(z.read('chembl_export/external_endpoints.csv')))
        original_summary = json.loads(z.read('chembl_export/collection_summary.json'))
        archive_manifest = {'archive': str(archive.relative_to(root)),
                            'archive_sha256': sha_bytes(archive.read_bytes()),
                            'members': {name: sha_bytes(z.read(name)) for name in z.namelist()},
                            'original_collection_summary': original_summary}
    client = PacedClient(args.cache, timeout=args.timeout)
    records = {str(r['activity_id']): r for r in raw}
    original_ids = set(records)
    organism_ids = [k for k, t in targets.items()
                    if t['target_type'] == 'ORGANISM' and 'vaccinia' in (t.get('organism') or '').lower()]
    discovery = {'status': 'offline_archive_only', 'target_ids': organism_ids}
    try:
        if not args.offline:
            before = client.get(D.BASE + 'status.json')
            if before['chembl_db_version'] != original_summary['chembl_db_version']:
                raise RuntimeError('Archive/API release differs; refresh the complete export first')
            fresh_targets = client.pages('target', organism__icontains='vaccinia')
            targets.update({t['target_chembl_id']: t for t in fresh_targets})
            frozen_organism_ids = set(organism_ids)
            organism_ids = [t['target_chembl_id'] for t in fresh_targets
                            if t['target_type'] == 'ORGANISM' and 'vaccinia' in (t.get('organism') or '').lower()]
            if not organism_ids:
                raise RuntimeError('No Vaccinia organism target in the fresh target census')
            # No endpoint filter: this audits the completeness of the old five-type whitelist.
            all_target = client.pages('activity', target_chembl_id__in=','.join(sorted(organism_ids)))
            records.update({str(r['activity_id']): r for r in all_target})
            extra_assays = sorted({r['assay_chembl_id'] for r in all_target} - assays.keys())
            if extra_assays:
                assays.update({a['assay_chembl_id']: a for a in client.batches('assay', 'assay_chembl_id', extra_assays)})
            extra_docs = sorted({r['document_chembl_id'] for r in all_target if r.get('document_chembl_id')} - documents.keys())
            if extra_docs:
                documents.update({d['document_chembl_id']: d for d in client.batches('document', 'document_chembl_id', extra_docs)})
            after = client.get(D.BASE + 'status.json', use_cache=False)
            if before['chembl_db_version'] != after['chembl_db_version']:
                raise RuntimeError('ChEMBL release changed during retrieval')
            discovery = dict(status='complete_target_endpoint_census', target_ids=organism_ids,
                             target_discovery_rows=len(fresh_targets),
                             new_organism_target_ids_vs_archive=sorted(set(organism_ids)-frozen_organism_ids),
                             organism_target_activity_rows=len(all_target),
                             endpoint_counts=dict(Counter(r.get('standard_type') or '' for r in all_target)),
                             new_activity_ids=len(set(records) - original_ids), before=before, after=after)
    finally:
        write_json(out / 'request_manifest.json', client.manifest)
        write_json(out / 'archive_manifest.json', archive_manifest)

    confirmed_assays = {aid for aid, a in assays.items() if D.assay_decision(a, targets).startswith('included_')}
    confirmed_docs = {r.get('document_chembl_id') for r in records.values() if r['assay_chembl_id'] in confirmed_assays}
    confirmed_activity_ids = {str(r['activity_id']) for r in records.values() if r['assay_chembl_id'] in confirmed_assays}
    normalized = pd.DataFrame([D.normalize_record(r, assays, documents, confirmed_activity_ids)
                               for r in records.values()])
    train, test, curated = reference_sets(root)
    unique_keys = {s: graph_keys(s) for s in normalized.SMILES.dropna().unique()}
    metadata = []
    for row in normalized.to_dict('records'):
        key = unique_keys.get(row['SMILES'])
        if key is None:
            key = dict(canonical_smiles=row.get('SMILES') or '', InChIKey='',
                       parent_smiles='', tautomer_parent='', scaffold='')
        endpoint = row['endpoint']
        status = row['validation_status']
        value = (row.get(endpoint + '_uM') if endpoint in ['IC50', 'EC50', 'CC50'] else row.get('SI'))
        definition = si_definition(row.get('assay_description')) if endpoint.lower() in D.SI_TYPES else ''
        task = ''
        eligible = status == 'valid_endpoint'
        reason = []
        if endpoint in ['IC50', 'EC50']:
            task = 'VACV_' + endpoint
            if row['assay_chembl_id'] not in confirmed_assays:
                eligible = False; reason.append('virus_target_not_confirmed')
        elif endpoint == 'CC50':
            task = 'cytotoxicity_CC50'
            if row.get('document_chembl_id') not in confirmed_docs:
                eligible = False; reason.append('not_from_confirmed_VACV_publication')
        elif endpoint.lower() in D.SI_TYPES:
            task = 'reported_' + definition
            if definition == 'unverified_definition':
                eligible = False; reason.append('SI_definition_unverified')
            # An explicit VACV ratio may be linked to a host-cell target. Keep it
            # as a separate auxiliary endpoint with its definition visible.
            if not re.search(r'vaccinia|\bvacv\b', str(row.get('assay_description') or ''), re.I):
                eligible = False; reason.append('VACV_context_not_confirmed')
        else:
            eligible = False; reason.append('endpoint_not_a_declared_training_task')
        if status != 'valid_endpoint':
            reason.append(status)
        if key['tautomer_parent'] in test:
            eligible = False; reason.append('official_test_structure_excluded')
        # No external technical-code labels are constructed.
        entry = dict(SMILES=key['canonical_smiles'], **{k: v for k, v in key.items() if k != 'canonical_smiles'},
                     source='ChEMBL', source_molecule_id=row['molecule_chembl_id'],
                     endpoint=endpoint, value_uM=value if endpoint in ['IC50','EC50','CC50'] else None,
                     value=row['value'], relation=row['relation'], units=row['units'],
                     task=task, reported_SI=value if endpoint.lower() in D.SI_TYPES else None,
                     si_definition=definition, cell=row['cell_line'], strain=row['assay_strain'],
                     doi=row['doi'], assay_id=row['assay_chembl_id'], activity_id=row['activity_id'],
                     document_id=row['document_chembl_id'], source_url=row['source_url'],
                     license_url=LICENSE_URL, license=LICENSE_CODE,
                     source_release=original_summary['chembl_db_version'],
                     endpoint_quality_status=status,
                     validation_status='eligible_separate_endpoint' if eligible else ';'.join(reason),
                     training_eligible=eligible, is_vacv_confirmed=row['assay_chembl_id'] in confirmed_assays,
                     is_organizer_train_structure=key['tautomer_parent'] in train,
                     is_organizer_test_structure=key['tautomer_parent'] in test,
                     is_existing_curated_structure=key['tautomer_parent'] in curated,
                     is_new_activity_vs_archive=row['activity_id'] not in original_ids)
        measurement = '|'.join(str(entry[k]) for k in ['doi','SMILES','endpoint','value','relation','units','cell','assay_id'])
        entry['measurement_key'] = sha_bytes(measurement.encode())
        metadata.append(entry)
    audit = pd.DataFrame(metadata)
    bank = audit[audit.training_eligible].copy()
    bank = bank.drop_duplicates('activity_id')
    audit.to_csv(out / 'endpoint_audit.csv', index=False)
    bank.to_csv(out / 'endpoint_bank.csv', index=False)
    audit[~audit.training_eligible].to_csv(out / 'excluded_endpoints.csv', index=False)
    audit[audit.endpoint.str.lower().isin(D.SI_TYPES)].to_csv(out / 'reported_si_review.csv', index=False)
    # No newly accepted CC50/IC50 labels: direct reported ratios require an
    # explicit compatible definition; candidate pairs remain in their own table.
    candidates, _ = D.build_candidates(normalized)
    candidates.to_csv(out / 'candidate_si_pairs_review_only.csv', index=False)
    verified = bank[bank.si_definition.eq('CC50/IC50')].copy()
    verified.to_csv(out / 'verified_si.csv', index=False)
    per_task = []
    official_scaffolds = {T.gscaf(T.std(s)) for p in ['train.csv','test.csv']
                          for s in pd.read_csv(root/'data/raw'/p, usecols=['SMILES']).SMILES}
    curated_scaffolds = {T.gscaf(T.std(s)) for s in pd.read_csv(root/'data/external_curated/vacv_si.csv').SMILES}
    for task, group in bank.groupby('task'):
        novel = group[~group.tautomer_parent.isin(train | test | curated)]
        newsc = group[~group.scaffold.isin(official_scaffolds | curated_scaffolds)]
        per_task.append(dict(task=task, rows=len(group), molecules=group.tautomer_parent.nunique(),
                             scaffolds=group.scaffold.nunique(), publications=group.doi.nunique(),
                             new_molecules_vs_train_test_curated=novel.tautomer_parent.nunique(),
                             new_scaffolds_vs_train_test_curated=newsc.scaffold.nunique(),
                             raw_API_activities_new_to_archive=int(group.is_new_activity_vs_archive.sum())))
    summary = dict(source='ChEMBL', source_release=original_summary['chembl_db_version'],
                   retrieved_utc=datetime.now(timezone.utc).isoformat(), discovery=discovery,
                   archive_original_records=len(raw), unified_activity_rows=len(records),
                   rows_with_valid_structure=int(audit.InChIKey.ne('').sum()),
                   technically_valid_declared_endpoints=int((audit.endpoint_quality_status.eq('valid_endpoint') & audit.endpoint.isin(D.ENDPOINTS.split(','))).sum()),
                   other_endpoint_rows_not_declared_as_training_tasks=int((~audit.endpoint.isin(D.ENDPOINTS.split(','))).sum()),
                   official_test_structure_rows_excluded=int(audit.is_organizer_test_structure.sum()),
                   auxiliary_eligible_rows=len(bank), per_task=per_task,
                   verified_SI_CC50_over_IC50_rows=len(verified),
                   provisional_pairs_not_training_labels=len(candidates),
                   label_policy='Different endpoints are separate tasks; only exact positive values. No external 10000/zero code labels. Test structures excluded before export; fold scaffold exclusions are additionally required at model fit.',
                   novelty_warning='New-to-model auxiliary endpoints are not necessarily newly discovered measurements: the old export was previously used as a source-presence feature.',
                   license_url=LICENSE_URL,
                   license=LICENSE_CODE,
                   license_conditions='Attribute ChEMBL/EMBL-EBI, release37 and original publications; redistribute adapted ChEMBL tables under CC-BY-SA-3.0. Official database FAQ verifies this license.',
                   license_evidence_url='https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions',
                   raw_response_cache=str(Path(args.cache).resolve()))
    write_json(out / 'collection_summary.json', summary)
    result_path = root / 'results/external_20261107/chembl_collection.json'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, summary)
    task_table = '\n'.join('| {task} | {rows} | {molecules} | {scaffolds} | {new_molecules_vs_train_test_curated} | {new_scaffolds_vs_train_test_curated} |'.format(**row) for row in per_task)
    readme = f'''# ChEMBL: audited endpoint bank for the external-data study

Source: ChEMBL / EMBL-EBI, **{summary['source_release']}**. Retrieval: {summary['retrieved_utc']}.
Official API: https://www.ebi.ac.uk/chembl/api/data/ .
The API target census searches `organism__icontains=vaccinia`, retains ORGANISM
targets, and retrieves every endpoint type for those targets, with full pagination.
The frozen repository export additionally includes documented cytotoxicity and
ambiguous assay records, which remain explicitly identified.

The merged audit has **{summary['unified_activity_rows']} activities**, including
**{summary['technically_valid_declared_endpoints']} technically valid declared
endpoints**. The unrestricted target census added {discovery.get('new_activity_ids', 0)}
activity IDs to the old export; **none supplied a new eligible exact IC50/EC50/CC50
or compatible direct SI**. Other endpoints such as MIC, ED50, percent inhibition
and concentration-dependent activity are retained in the rejection audit, never
renamed IC50 or assigned a Kaggle technical code.

`endpoint_bank.csv` contains **{len(bank)} rows** for separate auxiliary tasks.
All external labels for official test structures were removed before this export.
Rows overlapping official training structures are flagged: each fold must also
exclude its validation structures and scaffolds at training time.

| Separate task | Rows | Molecules | Scaffolds | New molecules vs train/test/curated | New scaffolds vs train/test/curated |
|---|---:|---:|---:|---:|---:|
{task_table}

The four reported `CC50/EC50` ratios have an explicit database assay definition;
they are not CC50/IC50 labels. `verified_si.csv` contains **{len(verified)}** newly
accepted CC50/IC50 labels. `candidate_si_pairs_review_only.csv` contains
{len(candidates)} unapproved candidate ratios; matching molecule/document IDs alone
does not establish comparable biological measurements. The source provides
numeric auxiliary information that was already archived; the earlier model used
most of this archive only to derive source presence. These are not newly discovered
experimental measurements and must not be counted as independent replicates.

Files preserve full canonical structure and stereo, InChIKey, standardized parent
and tautomer/scaffold keys, exact relation and units, cell/strain where provided,
DOI, molecule/assay/activity/document IDs, API URL, source release, quality status,
and inclusion/exclusion reasons. No laboratory procedures are collected or generated.

`archive_manifest.json` identifies the existing source ZIP and member SHA256.
`request_manifest.json` identifies each additional raw API response in the cache,
its retrieval/access dates and SHA256. API pagination count/ID checks, bounded
retry, at least 0.5 seconds between fresh requests and persistent cache support
resumption. The complete source archive already lives at
`data/chembl_package/vacv_chembl_data.zip`; new API responses are in the configured
cache and can be archived by response hash. `files_sha256.json` hashes this export.

From the repository root:

```bash
python src/collect_chembl_research.py \\
  --cache work/external_source_cache/chembl/ChEMBL_37 \\
  --out data/external_research/chembl
python -m unittest discover -s tests -p test_chembl_research.py -v
```

`--offline` audits only the frozen archive, explicitly marked as such. It does
not claim a fresh target/endpoint census. A changed ChEMBL release requires a
complete reviewed refresh rather than silently mixing releases.

## License and attribution

Adapted ChEMBL data in this directory are **CC BY-SA 3.0 Unported**:
https://creativecommons.org/licenses/by-sa/3.0/ . Attribute ChEMBL / EMBL-EBI,
release 37 and the linked original publications, and preserve the ShareAlike terms
when redistributing adapted ChEMBL tables. The license is confirmed by the official
FAQ: https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions .
API access is also subject to EMBL-EBI terms. The scripts are project code; this
statement identifies the data license and does not relicense unrelated project files.
'''
    (out / 'README.md').write_text(readme)
    write_json(out / 'files_sha256.json', {p.name: sha_bytes(p.read_bytes()) for p in out.iterdir() if p.is_file() and p.name != 'files_sha256.json'})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='data/external_research/chembl')
    parser.add_argument('--cache',default='work/external_source_cache/chembl/ChEMBL_37')
    parser.add_argument('--timeout',type=float,default=60)
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    collect(args)


if __name__=='__main__':main()
