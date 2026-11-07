"""Collect a bounded, auditable PubChem VACV auxiliary-data pilot.

This is not a source of automatically approved SI labels. AC50 reporter assays,
cytotoxicity, censored endpoints and activity outcomes remain separate tasks.
Only measurement tables and non-operational metadata enter the export. Full API
description responses stay in the work cache, including any source protocol text.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlencode, urlparse

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import inchi

import train_v5 as T

PUG = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/'
EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
SEARCH = 'vaccinia NOT ChEMBL[sourcename]'
PROJECT_DOSE_AIDS = (1117284, 1117274, 1117273, 1117272, 1117271,
                     1053207, 1053168, 1053167, 1053166, 1053115,
                     977617, 977616, 977615)
PANEL_AID = 602175
REVIEW_AIDS = (1802287, 1802001, 1795508)
LICENSE_URL = 'https://pubchem.ncbi.nlm.nih.gov/docs/downloads'
ROW_COLUMNS = ['SMILES', 'raw_smiles', 'InChIKey', 'parent', 'scaffold',
               'pubchem_aid', 'pubchem_sid', 'pubchem_cid', 'result_tag',
               'endpoint', 'value_uM', 'relation', 'activity_outcome', 'assay_activity_outcome',
               'phenotype', 'channel', 'cell_line', 'strain', 'doi',
               'assay_name', 'source', 'source_url', 'license_url',
               'validation_status', 'rejection_reason']


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class CachedHTTP:
    """Persistent byte-exact GET cache, provenance, throttling and retry."""

    def __init__(self, cache, rate=2.5, timeout=75, refresh=False):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.interval = 1 / rate
        self.timeout = timeout
        self.refresh = refresh
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'SI-VACV-open-data-audit/1.0'
        self.last_request = 0.
        self.manifest = []

    def get(self, url, kind='json', allowed=(200,)):
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.netloc not in {
                'pubchem.ncbi.nlm.nih.gov', 'eutils.ncbi.nlm.nih.gov'}:
            raise ValueError('Unexpected source URL')
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.cache / f'{key}.{kind}'
        metadata = self.cache / f'{key}.meta.json'
        hit = path.exists() and metadata.exists() and not self.refresh
        if hit:
            raw = path.read_bytes()
            meta = json.loads(metadata.read_text())
            if hashlib.sha256(raw).hexdigest() != meta['sha256']:
                raise ValueError('Corrupt cached response: ' + str(path))
        else:
            for attempt in range(5):
                delay = self.interval - (time.monotonic() - self.last_request)
                if delay > 0:
                    time.sleep(delay)
                self.last_request = time.monotonic()
                try:
                    response = self.session.get(url, timeout=(15, self.timeout))
                    if response.status_code in (408, 429, 500, 502, 503, 504):
                        if attempt == 4:
                            response.raise_for_status()
                        retry = response.headers.get('Retry-After', '')
                        time.sleep(min(float(retry) if retry.isdigit() else 2 ** attempt, 30))
                        continue
                    response.raise_for_status() if response.status_code not in allowed else None
                    raw = response.content
                    meta = {'url': url, 'status': response.status_code,
                            'retrieved_utc': now(), 'sha256': hashlib.sha256(raw).hexdigest(),
                            'bytes': len(raw), 'content_type': response.headers.get('Content-Type', '')}
                    temp = path.with_suffix(path.suffix + '.part')
                    temp.write_bytes(raw)
                    temp.replace(path)
                    dump(metadata, meta)
                    break
                except (requests.ConnectionError, requests.Timeout):
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            else:
                raise RuntimeError('Retry loop exhausted')
        if meta['status'] not in allowed:
            raise ValueError('Cached error response: ' + str(meta['status']))
        self.manifest.append({**meta, 'cache_file': path.name,
                              'cache_hit': hit, 'accessed_utc': now()})
        return json.loads(raw) if kind == 'json' else raw

    def search(self, query, page_size=500):
        ids = []
        expected = None
        for start in range(0, 100000, page_size):
            url = EUTILS + 'esearch.fcgi?' + urlencode({
                'db': 'pcassay', 'term': query, 'retmode': 'json',
                'retstart': start, 'retmax': page_size})
            record = self.get(url)['esearchresult']
            if record.get('errorlist'):
                raise ValueError('Invalid ESearch query: ' + json.dumps(record['errorlist']))
            count = int(record['count'])
            expected = count if expected is None else expected
            if count != expected:
                raise RuntimeError('Search changed during pagination')
            ids.extend(map(int, record['idlist']))
            if len(ids) >= count:
                break
            if not record['idlist']:
                raise RuntimeError('Empty page before all records were retrieved')
        if len(ids) != expected or len(set(ids)) != expected:
            raise RuntimeError('Incomplete or duplicate ESearch result')
        return ids

    def descriptions(self, ids):
        result = {}
        for start in range(0, len(ids), 25):
            batch = ids[start:start + 25]
            url = PUG + 'assay/aid/' + ','.join(map(str, batch)) + '/description/JSON'
            data = self.get(url)
            for container in data['PC_AssayContainer']:
                descr = container['assay']['descr']
                result[int(descr['aid']['id'])] = descr
        if set(result) != set(ids):
            raise RuntimeError('Incomplete assay description batch')
        return result


def csv_records(raw):
    data = pd.read_csv(io.BytesIO(raw), dtype=str).fillna('')
    if 'PUBCHEM_RESULT_TAG' not in data:
        raise ValueError('Missing PubChem result identifiers')
    metadata = data[data.PUBCHEM_RESULT_TAG.str.startswith('RESULT_')]
    records = data[pd.to_numeric(data.PUBCHEM_RESULT_TAG, errors='coerce').notna()].copy()
    if not records.PUBCHEM_RESULT_TAG.is_unique:
        raise ValueError('Duplicate result tag within assay')
    units = metadata[metadata.PUBCHEM_RESULT_TAG.eq('RESULT_UNIT')]
    unit_map = units.iloc[0].to_dict() if len(units) == 1 else {}
    return records, unit_map


def finite_positive(text):
    try:
        number = float(text)
        return number if math.isfinite(number) and number > 0 else None
    except (ValueError, TypeError):
        return None


def structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid depositor SMILES')
    parent = T.std(smiles)
    return {'SMILES': Chem.MolToSmiles(mol, isomericSmiles=True),
            'raw_smiles': smiles, 'InChIKey': inchi.MolToInchiKey(mol),
            'parent': T.taut_key(parent), 'scaffold': T.gscaf(parent)}


def short_metadata(descr, status):
    """Never export protocols or full descriptions from an assay response."""
    name = descr['name']
    cell = next((c for c in ['A549', 'HEK293', 'HepG2', 'Vero', 'HeLa']
                 if c.lower() in name.lower()), '')
    return {'aid': int(descr['aid']['id']), 'version': descr['aid'].get('version'),
            'revision': descr.get('revision'), 'name': name,
            'depositor': descr['aid_source']['db']['name'],
            'source_id': descr['aid_source']['db'].get('source_id'),
            'cell_line': cell, 'status': status,
            'source_url': 'https://pubchem.ncbi.nlm.nih.gov/bioassay/' + str(descr['aid']['id']),
            'results': descr.get('results', []), 'license_url': LICENSE_URL,
            'license_status': 'PubChem reuse policy plus depositor-specific terms; no universal license inferred'}


def assay_status(aid, descr):
    if aid in PROJECT_DOSE_AIDS:
        return 'selected_project_dose_response_auxiliary'
    if aid == PANEL_AID:
        return 'selected_mixed_panel_only_explicit_VACV_rows'
    if aid in REVIEW_AIDS:
        return 'review_required_molecular_target_or_phenotype_ambiguous'
    text = descr['name'].lower()
    if any(k in text for k in ['vhr', 'phosphatase', 'protein-ligand', 'pdb', 'parkin']):
        return 'excluded_molecular_or_unrelated_target'
    if aid in (720579, 720580, 1053133):
        return 'not_downloaded_large_primary_screen'
    return 'not_selected_summary_single_point_or_other_context'


def row_base(record, metadata, aid, phenotype):
    base = {**structure(record['PUBCHEM_EXT_DATASOURCE_SMILES']),
            'pubchem_aid': aid, 'pubchem_sid': record['PUBCHEM_SID'],
            'pubchem_cid': record['PUBCHEM_CID'], 'result_tag': record['PUBCHEM_RESULT_TAG'],
            'activity_outcome': record['PUBCHEM_ACTIVITY_OUTCOME'],
            'assay_activity_outcome': record['PUBCHEM_ACTIVITY_OUTCOME'],
            'phenotype': phenotype, 'cell_line': metadata['cell_line'],
            'strain': '', 'doi': '', 'assay_name': metadata['name'],
            'source': metadata['depositor'], 'source_url': metadata['source_url'],
            'license_url': LICENSE_URL}
    return base


def parse_dose(raw, metadata, aid):
    records, units = csv_records(raw)
    phenotype = 'cytotoxicity' if 'cytotoxicity' in metadata['name'].lower() else 'VACV_reporter_or_plaque_activity'
    rows, rejected = [], []
    for record in records.to_dict('records'):
        try:
            base = row_base(record, metadata, aid, phenotype)
        except (ValueError, KeyError) as exc:
            rejected.append({'pubchem_aid': aid, 'result_tag': record['PUBCHEM_RESULT_TAG'],
                             'rejection_reason': str(exc)})
            continue
        value = finite_positive(record.get('AC50_uM', ''))
        relation = record.get('AC50_Qualifier', '').strip()
        status = ('exact_auxiliary_AC50' if value is not None and relation == '=' else
                  'censored_auxiliary_AC50' if value is not None and relation in ('>', '<', '>=', '<=') else
                  'activity_outcome_only')
        if units.get('AC50_uM') != 'MICROMOLAR':
            raise ValueError('Unexpected AC50 unit')
        rows.append({**base, 'endpoint': 'AC50', 'value_uM': value,
                     'relation': relation, 'channel': 'PubChem_primary_AC50',
                     'validation_status': status, 'rejection_reason': ''})
    return rows, rejected, len(records)


def parse_panel(raw, metadata, aid):
    records, units = csv_records(raw)
    rows, rejected = [], []
    for record in records.to_dict('records'):
        panel = record.get('Panel Name', '')
        if panel != 'Vaccinia virus WR':
            rejected.append({'pubchem_aid': aid, 'result_tag': record['PUBCHEM_RESULT_TAG'],
                             'rejection_reason': 'non_VACV_panel', 'panel_name': panel})
            continue
        try:
            base = row_base(record, metadata, aid, 'VACV_cellular_inhibition')
        except (ValueError, KeyError) as exc:
            rejected.append({'pubchem_aid': aid, 'result_tag': record['PUBCHEM_RESULT_TAG'],
                             'rejection_reason': str(exc)})
            continue
        base['strain'] = 'WR'
        for replicate in (1, 2, 3):
            for endpoint in ('IC50', 'TC50'):
                column = f'{endpoint} Rep {replicate}'
                if units.get(column) != 'MICROMOLAR':
                    raise ValueError('Unexpected panel concentration unit')
                value = finite_positive(record.get(column))
                modifier = record.get(f'{endpoint} Modifier Rep {replicate}', '').strip()
                # This depositor uses blank modifier for a finite exact number;
                # explicit '>' bounds are preserved, never treated as measurements.
                relation = modifier or '='
                status = 'exact_auxiliary_' + endpoint if value is not None and relation == '=' else 'censored_auxiliary_' + endpoint
                rows.append({**base, 'endpoint': endpoint, 'value_uM': value,
                             'relation': relation, 'channel': f'replicate_{replicate}',
                             'phenotype': 'cytotoxicity' if endpoint == 'TC50' else base['phenotype'],
                             'activity_outcome': '' if endpoint == 'TC50' else base['activity_outcome'],
                             'validation_status': status, 'rejection_reason': ''})
    return rows, rejected, len(records)


def overlap_audit(rows, root):
    def keys(path):
        source = pd.read_csv(path, usecols=['SMILES']).SMILES.dropna().unique()
        mols = [T.std(s) for s in source]
        return {T.taut_key(m) for m in mols if m is not None}, {T.gscaf(m) for m in mols if m is not None}
    train, train_sc = keys(root / 'data/raw/train.csv')
    test, test_sc = keys(root / 'data/raw/test.csv')
    previous, previous_sc = keys(root / 'data/external/external_endpoints.csv')
    data = pd.DataFrame(rows, columns=ROW_COLUMNS)
    data['test_overlap'] = data.parent.isin(test)
    data['train_overlap'] = data.parent.isin(train)
    data['existing_endpoint_structure_overlap'] = data.parent.isin(previous)
    data['new_scaffold_vs_official'] = ~data.scaffold.isin(train_sc | test_sc)
    accepted = data[~data.test_overlap].copy()
    return data, accepted, {
        'export_rows': len(data), 'structures': data.parent.nunique(), 'scaffolds': data.scaffold.nunique(),
        'train_overlap_structures': len(set(data.parent) & train),
        'test_overlap_structures': len(set(data.parent) & test),
        'existing_external_endpoint_overlap_structures': len(set(data.parent) & previous),
        'new_structures_vs_official_and_previous': len(set(data.parent) - train - test - previous),
        'new_scaffolds_vs_official': len(set(data.scaffold) - train_sc - test_sc),
        'new_scaffolds_vs_official_and_previous': len(set(data.scaffold) - train_sc - test_sc - previous_sc),
        'retained_auxiliary_rows_after_test_exclusion': len(accepted),
        'exact_positive_auxiliary_rows': int(((data.relation == '=') & (data.value_uM > 0)).sum()),
        'verified_SI_rows': 0, 'used_as_SI_training_labels': 0,
        'note': 'New structures/assay observations are not independent SI labels; the 3 panel compounds have censored antiviral IC50.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/external_research/pubchem')
    parser.add_argument('--report-out', default='results/external_20261107')
    parser.add_argument('--cache', default='work/external_source_cache/pubchem')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--rate', type=float, default=2.5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out, report = root / args.out, root / args.report_out
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    client = CachedHTTP(root / args.cache, args.rate, refresh=args.refresh)
    ids = client.search(SEARCH)
    descriptions = client.descriptions(sorted(set(ids) | set(PROJECT_DOSE_AIDS) | {PANEL_AID}))
    meta = {aid: short_metadata(descr, assay_status(aid, descr)) for aid, descr in descriptions.items()}
    dump(out / 'assay_metadata.json', list(meta.values()))
    selected = sorted(set(PROJECT_DOSE_AIDS) | {PANEL_AID})
    rows, exclusions, counts = [], [], []
    raw_dir = out / 'raw_numeric_tables'
    raw_dir.mkdir(exist_ok=True)
    for aid in selected:
        url = PUG + f'assay/aid/{aid}/CSV'
        raw = client.get(url, 'csv')
        # CSV contains measurement definitions/results, no source assay protocol.
        (raw_dir / f'AID_{aid}.csv').write_bytes(raw)
        sids = client.get(PUG + f'assay/aid/{aid}/sids/JSON')['InformationList']['Information']
        expected_sids = {str(sid) for entry in sids for sid in entry.get('SID', [])}
        expected = len(expected_sids)
        parsed, rejected, actual = (parse_panel if aid == PANEL_AID else parse_dose)(raw, meta[aid], aid)
        actual_sids = set(csv_records(raw)[0].PUBCHEM_SID)
        if actual_sids != expected_sids:
            raise RuntimeError(f'Incomplete assay {aid}: CSV SIDs do not match the SID resource')
        rows.extend(parsed)
        exclusions.extend(rejected)
        counts.append({'aid': aid, 'source_rows': actual, 'result_rows_exported': len(parsed),
                       'excluded_source_rows': len(rejected), 'expected_SIDs': expected,
                       'csv_unique_SIDs': len(actual_sids)})
        print('PUBCHEM', aid, counts[-1], flush=True)
    data, accepted, summary = overlap_audit(rows, root)
    data.to_csv(out / 'all_auxiliary_endpoints.csv', index=False)
    accepted.to_csv(out / 'auxiliary_endpoints.csv', index=False)
    excluded_overlap = data[data.test_overlap].copy()
    excluded_overlap['rejection_reason'] = 'official_test_structure_external_label_forbidden'
    pd.concat([pd.DataFrame(exclusions), excluded_overlap], ignore_index=True).to_csv(out / 'excluded_rows.csv', index=False)
    pd.DataFrame(columns=['SMILES', 'IC50_uM', 'CC50_uM', 'SI', 'doi', 'validation_status']).to_csv(out / 'verified_si.csv', index=False)
    pd.DataFrame([{'aid': aid, 'reason': meta[aid]['status'], 'source_url': meta[aid]['source_url']}
                  for aid in REVIEW_AIDS if aid in meta]).to_csv(out / 'review_required_assays.csv', index=False)
    pd.DataFrame(counts).to_csv(out / 'assay_counts.csv', index=False)
    dump(out / 'request_manifest.json', client.manifest)
    summary.update({'retrieved_utc': now(), 'discovery_query': SEARCH, 'discovery_AIDs': len(ids),
                    'selected_dose_response_AIDs': list(PROJECT_DOSE_AIDS), 'panel_AID': PANEL_AID,
                    'large_primary_screen_downloaded': False,
                    'activity_outcomes': accepted.activity_outcome.value_counts().to_dict(),
                    'endpoints': accepted.endpoint.value_counts().to_dict(),
                    'license_url': LICENSE_URL,
                    'raw_description_protocols_redistributed': False,
                    'doi_policy': 'Blank unless a primary measurement publication is unambiguously identified; background PMID xrefs are not DOI provenance.',
                    'sids_vs_csv_completeness_verified': True})
    dump(report / 'pubchem_summary.json', summary)
    dump(out / 'collection_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
