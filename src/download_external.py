"""Auditable ChEMBL VACV endpoint collection; compatible with curate_external.py.

Only organism-level VACV functional assays are accepted automatically.
Concentration ratios are candidates for publication review, never training labels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

import pandas as pd
from rdkit import Chem

BASE = 'https://www.ebi.ac.uk/chembl/api/data/'
ENDPOINTS = 'IC50,EC50,CC50,SI,Selectivity index'
UNIT_TO_UM = {'nM': .001, 'uM': 1., 'µM': 1., 'μM': 1.,
              'mM': 1000., 'M': 1e6, 'pM': 1e-6}
SI_TYPES = {'si', 'selectivity index'}
PAIR_COLS = [
    'SMILES', 'IC50_uM', 'EC50_uM', 'CC50_uM', 'SI', 'doi',
    'denominator_endpoint', 'antiviral_activity_id', 'cytotoxicity_activity_id',
    'cell_line_antiviral', 'cell_line_cyto', 'duration_antiviral', 'duration_cyto',
    'source_url', 'cyto_source_url', 'validation_status',
    'molecule_chembl_id', 'document_chembl_id', 'si_definition',
]
DIRECT_COLS = ['SMILES', 'IC50_uM', 'CC50_uM', 'SI', 'doi', 'source',
               'source_url', 'assay_chembl_id', 'document_chembl_id',
               'validation_status']


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class Client:
    """Paginated API client with retry, persistent cache and response hashes."""

    def __init__(self, cache, refresh=False, page_size=1000, workers=4, timeout=60):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh
        self.page_size = page_size
        self.workers = workers
        self.timeout = timeout
        self.manifest = []
        self.lock = threading.Lock()

    def get(self, url, use_cache=True):
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.netloc != 'www.ebi.ac.uk':
            raise ValueError(f'Unexpected API URL: {url}')
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.cache / (key + '.json')
        meta_path = self.cache / (key + '.meta.json')
        hit = use_cache and not self.refresh and path.exists()
        if hit:
            raw = path.read_bytes()
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            data = json.loads(raw)
        else:
            for attempt in range(5):
                try:
                    request = Request(url, headers={'User-Agent': 'VACV-ChEMBL-audit/2.0',
                                                    'Accept': 'application/json'})
                    with urlopen(request, timeout=self.timeout) as response:
                        raw = response.read()
                    data = json.loads(raw)
                    break
                except HTTPError as exc:
                    if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 4:
                        raise
                    retry_after = exc.headers.get('Retry-After', '')
                    delay = float(retry_after) if retry_after.isdigit() else 2 ** attempt
                    time.sleep(min(delay, 30))
                except (URLError, TimeoutError, json.JSONDecodeError):
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            meta = {'fetched_utc': utc_now()}
            if use_cache:
                temp = path.with_suffix('.tmp')
                temp.write_bytes(raw)
                temp.replace(path)
                write_json(meta_path, meta)
        entry = {'url': url, 'sha256': hashlib.sha256(raw).hexdigest(),
                 'cache_file': str(path) if use_cache else None,
                 'from_cache': bool(hit), 'fetched_utc': meta.get('fetched_utc'),
                 'accessed_utc': utc_now()}
        with self.lock:
            self.manifest.append(entry)
        return data

    def pages(self, resource, **filters):
        url = BASE + resource + '.json?' + urlencode(dict(filters, limit=self.page_size))
        rows, seen, expected = [], set(), None
        key = 'activities' if resource == 'activity' else resource + 's'
        while url:
            if url in seen:
                raise RuntimeError(f'Pagination loop: {url}')
            seen.add(url)
            data = self.get(url)
            meta = data['page_meta']
            if expected is None:
                expected = meta['total_count']
            if meta['total_count'] != expected:
                raise RuntimeError('ChEMBL result count changed during pagination; retry with --refresh')
            rows.extend(data[key])
            url = urljoin(BASE, meta['next']) if meta.get('next') else None
        if len(rows) != expected:
            raise RuntimeError(f'Incomplete {resource}: got {len(rows)}, expected {expected}')
        id_key = 'activity_id' if resource == 'activity' else resource + '_chembl_id'
        if len({r[id_key] for r in rows}) != len(rows):
            raise RuntimeError(f'Duplicate IDs during {resource} pagination')
        return rows

    def batches(self, resource, field, ids, extra=None):
        ids = sorted(set(ids))
        chunks = [ids[i:i + 30] for i in range(0, len(ids), 30)]

        def run(chunk):
            return self.pages(resource, **{field + '__in': ','.join(chunk), **(extra or {})})

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return [row for result in pool.map(run, chunks) for row in result]


def norm_cell(assay):
    return re.sub(r'[^a-z0-9]', '', ((assay or {}).get('assay_cell_type') or '').lower())


def duration(description):
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|days?|d)\b',
                         (description or '').lower())
    if len(matches) != 1:
        return ''
    value, unit = matches[0]
    hours = float(value) * (24 if unit.startswith('d') else 1)
    return f'{hours:g}'


def finite_number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def assay_decision(assay, targets):
    """A description mentioning VACV alone cannot establish an antiviral endpoint."""
    target = targets.get(assay.get('target_chembl_id'), {})
    if assay.get('assay_type') != 'F':
        return 'excluded_nonfunctional'
    if target.get('target_type') == 'ORGANISM' and 'vaccinia' in (target.get('organism') or '').lower():
        return 'included_vacv_organism_target'
    if target.get('target_type') in {'SINGLE PROTEIN', 'PROTEIN COMPLEX', 'PROTEIN FAMILY'}:
        return 'excluded_molecular_target'
    return 'review_required_virus_target_not_confirmed'


def discover_assays(client):
    targets = client.pages('target', organism__icontains='vaccinia')
    target_map = {r['target_chembl_id']: r for r in targets}
    organism_ids = [r['target_chembl_id'] for r in targets if r['target_type'] == 'ORGANISM']
    if not organism_ids:
        raise RuntimeError('No VACV organism targets found in ChEMBL')
    queries = [('assay_organism', {'assay_organism__icontains': 'vaccinia'}),
               ('description_vaccinia', {'description__icontains': 'vaccinia'}),
               ('description_vacv', {'description__icontains': 'VACV'})]
    found, routes = {}, {}
    for route, filters in queries:
        for row in client.pages('assay', **filters):
            aid = row['assay_chembl_id']
            found[aid] = row
            routes.setdefault(aid, set()).add(route)
    for row in client.batches('assay', 'target_chembl_id', organism_ids):
        aid = row['assay_chembl_id']
        found[aid] = row
        routes.setdefault(aid, set()).add('vacv_organism_target')
    other_ids = {r['target_chembl_id'] for r in found.values() if r.get('target_chembl_id')} - target_map.keys()
    target_map.update({r['target_chembl_id']: r for r in client.batches('target', 'target_chembl_id', other_ids)})
    audit = []
    for aid, row in sorted(found.items()):
        target = target_map.get(row.get('target_chembl_id'), {})
        audit.append({**row, 'search_routes': ';'.join(sorted(routes[aid])),
                      'selection_status': assay_decision(row, target_map),
                      'target_type': target.get('target_type'),
                      'target_organism': target.get('organism')})
    return found, target_map, pd.DataFrame(audit)


def normalize_record(record, assays, documents, vacv_ids):
    aid = str(record['activity_id'])
    assay = assays.get(record['assay_chembl_id'], {})
    doc = documents.get(record.get('document_chembl_id'), {})
    endpoint = record.get('standard_type') or ''
    value = finite_number(record.get('standard_value'))
    units = record.get('standard_units') or ''
    # The raw relation belongs to a different value; do not use it as a fallback.
    relation = record.get('standard_relation') or ''
    smiles = record.get('canonical_smiles') or ''
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    concentration = value * UNIT_TO_UM[units] if value is not None and units in UNIT_TO_UM else None
    if concentration is not None and not math.isfinite(concentration):
        concentration = None
    direct_si = value if endpoint.lower() in SI_TYPES and units.lower() in ('', 'dimensionless', 'ratio') else None
    bad = []
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        bad.append('invalid_or_missing_smiles')
    if relation != '=':
        bad.append('censored_or_unknown_relation')
    if record.get('data_validity_comment'):
        bad.append('chembl_validity_flag')
    if record.get('potential_duplicate'):
        bad.append('potential_duplicate')
    if value is None:
        bad.append('nonpositive_or_missing_value')
    if endpoint in ('IC50', 'EC50', 'CC50') and concentration is None:
        bad.append('unsupported_concentration_units')
    if endpoint.lower() in SI_TYPES and direct_si is None:
        bad.append('unsupported_si_units_or_value')
    doi = doc.get('doi') or ''
    return dict(
        SMILES=Chem.MolToSmiles(mol) if mol is not None else smiles,
        IC50_uM=concentration if endpoint == 'IC50' else None,
        EC50_uM=concentration if endpoint == 'EC50' else None,
        CC50_uM=concentration if endpoint == 'CC50' else None, SI=direct_si,
        doi=doi, endpoint=endpoint, value=record.get('standard_value'), units=units,
        relation=relation, activity_id=aid, molecule_chembl_id=record.get('molecule_chembl_id'),
        assay_chembl_id=record['assay_chembl_id'], document_chembl_id=record.get('document_chembl_id'),
        assay_organism=assay.get('assay_organism'), cell_line=assay.get('assay_cell_type') or '',
        cell_key=norm_cell(assay), duration_hours=duration(assay.get('description')),
        assay_description=assay.get('description') or record.get('assay_description') or '',
        is_vacv_assay=aid in vacv_ids,
        validation_status='valid_endpoint' if not bad else ';'.join(bad),
        source='ChEMBL', source_url=BASE + 'activity/' + aid + '.json',
        publication_url='https://doi.org/' + doi if doi else '',
        raw_type=record.get('type'), raw_value=record.get('value'),
        raw_units=record.get('units'), raw_relation=record.get('relation'),
        data_validity_comment=record.get('data_validity_comment'),
        potential_duplicate=record.get('potential_duplicate'),
        target_chembl_id=record.get('target_chembl_id'),
        assay_strain=assay.get('assay_strain') or '',
        cell_chembl_id=assay.get('cell_chembl_id') or '',
    )


def build_candidates(df):
    valid = df[df.validation_status.eq('valid_endpoint')]
    cc = valid[valid.endpoint.eq('CC50')]
    pairs, direct = [], []
    antiviral = valid[valid.is_vacv_assay & valid.endpoint.isin(['IC50', 'EC50', 'SI', 'Selectivity index'])]
    for _, row in antiviral.iterrows():
        if pd.notna(row.SI):
            direct.append(dict(SMILES=row.SMILES, IC50_uM=None, CC50_uM=None, SI=row.SI,
                               doi=row.doi, source='ChEMBL_direct_SI', source_url=row.source_url,
                               assay_chembl_id=row.assay_chembl_id, document_chembl_id=row.document_chembl_id,
                               validation_status='direct_reported_SI_needs_assay_review'))
            continue
        if not row.molecule_chembl_id or not row.document_chembl_id:
            continue
        cytotoxicity = cc[(cc.molecule_chembl_id == row.molecule_chembl_id) &
                         (cc.document_chembl_id == row.document_chembl_id)]
        for _, other in cytotoxicity.iterrows():
            denominator = row.IC50_uM if row.endpoint == 'IC50' else row.EC50_uM
            ratio = other.CC50_uM / denominator
            if not math.isfinite(ratio) or ratio <= 0:
                continue
            matched = bool(row.cell_key and row.cell_key == other.cell_key and
                           row.duration_hours and row.duration_hours == other.duration_hours)
            pairs.append(dict(
                SMILES=row.SMILES, IC50_uM=row.IC50_uM, EC50_uM=row.EC50_uM,
                CC50_uM=other.CC50_uM, SI=ratio, doi=row.doi,
                denominator_endpoint=row.endpoint, antiviral_activity_id=row.activity_id,
                cytotoxicity_activity_id=other.activity_id, cell_line_antiviral=row.cell_line,
                cell_line_cyto=other.cell_line, duration_antiviral=row.duration_hours,
                duration_cyto=other.duration_hours, source_url=row.source_url,
                cyto_source_url=other.source_url,
                validation_status='metadata_matched_manual_review_required' if matched else 'blocked_context_not_verified',
                molecule_chembl_id=row.molecule_chembl_id, document_chembl_id=row.document_chembl_id,
                si_definition='CC50/' + row.endpoint,
            ))
    return pd.DataFrame(pairs, columns=PAIR_COLS), pd.DataFrame(direct, columns=DIRECT_COLS)


def collect(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'collection_summary.json').unlink(missing_ok=True)
    client = Client(args.cache, refresh=args.refresh, page_size=args.page_size,
                    workers=args.workers, timeout=args.timeout)
    before = client.get(BASE + 'status.json', use_cache=False)
    version = before['chembl_db_version']
    client.cache = client.cache / re.sub(r'[^A-Za-z0-9_.-]', '_', version)
    client.cache.mkdir(parents=True, exist_ok=True)
    try:
        assays, targets, audit = discover_assays(client)
        usable = audit.loc[audit.selection_status.str.startswith('included_'), 'assay_chembl_id'].tolist()
        review_ids = audit.loc[audit.selection_status.str.startswith('review_required_'), 'assay_chembl_id'].tolist()
        if not usable:
            raise RuntimeError('No confirmed functional VACV assays found')
        print(f'{version}: {len(audit)} search hits; {len(usable)} accepted assays', flush=True)
        acts = client.batches('activity', 'assay_chembl_id', usable, {'standard_type__in': ENDPOINTS})
        if not acts:
            raise RuntimeError('No VACV endpoints found')
        review_acts = client.batches('activity', 'assay_chembl_id', review_ids, {'standard_type__in': ENDPOINTS})
        print(f'VACV endpoints: {len(acts)}; uncertain hits: {len(review_acts)}; retrieving publications and CC50', flush=True)
        docs = sorted({r['document_chembl_id'] for r in acts + review_acts if r.get('document_chembl_id')})
        documents = client.batches('document', 'document_chembl_id', docs)
        dmap = {r['document_chembl_id']: r for r in documents}
        cyto = client.batches('activity', 'document_chembl_id', docs, {'standard_type__in': 'CC50'})
        extra_assays = {r['assay_chembl_id'] for r in cyto} - assays.keys()
        assays.update({r['assay_chembl_id']: r for r in client.batches('assay', 'assay_chembl_id', extra_assays)})
        records = {str(r['activity_id']): r for r in acts + review_acts + cyto}
        vacv_ids = {str(r['activity_id']) for r in acts}
        df = pd.DataFrame([normalize_record(records[k], assays, dmap, vacv_ids)
                           for k in sorted(records, key=int)])
        decisions = audit.set_index('assay_chembl_id').selection_status
        df['vacv_selection_status'] = df.assay_chembl_id.map(decisions).fillna('cytotoxicity_context_only')
        # Additional cytotoxicity from ambiguous publications cannot supply labels
        # for a confirmed assay without the same publication/molecule match.
        pairs, direct = build_candidates(df)
        after = client.get(BASE + 'status.json', use_cache=False)
        if after['chembl_db_version'] != version:
            raise RuntimeError('ChEMBL release changed during download; rerun with --refresh')
        df.to_csv(out / 'external_endpoints.csv', index=False)
        df[df.vacv_selection_status.str.startswith('review_required_')].to_csv(
            out / 'review_required_endpoints.csv', index=False)
        audit.to_csv(out / 'assay_search_audit.csv', index=False)
        write_json(out / 'vacv_assays.json', [assays[k] for k in sorted(usable)])
        write_json(out / 'all_assays.json', [assays[k] for k in sorted(assays)])
        write_json(out / 'documents.json', documents)
        write_json(out / 'targets.json', [targets[k] for k in sorted(targets)])
        write_json(out / 'chembl_status.json', {'before': before, 'after': after})
        write_json(out / 'raw_activities.json', [records[k] for k in sorted(records, key=int)])
        pairs.to_csv(out / 'candidate_si_pairs.csv', index=False)
        pairs[pairs.denominator_endpoint.eq('IC50')].to_csv(out / 'external_si_database.csv', index=False)
        direct.to_csv(out / 'direct_si.csv', index=False)
        columns = ['SMILES', 'IC50_uM', 'CC50_uM', 'SI', 'doi', 'EC50_uM', 'endpoint',
                   'relation', 'validation_status', 'source_url']
        df[columns].to_csv(out / 'collected_database.csv', index=False)
        summary = dict(
            source='ChEMBL', chembl_db_version=version, retrieved_utc=utc_now(),
            assay_search_hits=len(audit), vacv_assays=len(usable), functional_assays=len(usable),
            assay_selection_counts=audit.selection_status.value_counts().to_dict(),
            vacv_endpoint_records=len(acts),
            review_required_endpoint_records=len(review_acts),
            additional_cc50_records=len({str(r['activity_id']) for r in cyto} -
                                        {str(r['activity_id']) for r in acts + review_acts}),
            all_unique_endpoint_records=len(df), endpoint_counts=df.endpoint.value_counts().to_dict(),
            valid_endpoint_rows=int(df.validation_status.eq('valid_endpoint').sum()),
            validation_counts=df.validation_status.value_counts().to_dict(),
            rows_with_doi=int(df.doi.ne('').sum()), direct_si_records=len(direct),
            candidate_pairs=len(pairs), automatically_approved_training_rows=0,
            policy='Confirmed organism-level VACV functional assays; positive exact endpoints for candidate ratios. '
                   'No automatic pairing approval. EC50 separate from IC50. Bounds stay bounds; zero SI rejected.',
            license_url='https://chembl.gitbook.io/chembl-interface-documentation/about/terms-of-use',
            queries=len(client.manifest),
        )
        write_json(out / 'collection_summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        write_json(out / 'request_manifest.json',
                   sorted(client.manifest, key=lambda r: (r['url'], r['accessed_utc'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='Directory for exported CSV/JSON')
    parser.add_argument('--cache', default='work/chembl_cache')
    parser.add_argument('--refresh', action='store_true', help='Ignore cached API responses')
    parser.add_argument('--page-size', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    if not 1 <= args.page_size <= 1000 or not 1 <= args.workers <= 8 or args.timeout <= 0:
        parser.error('Require page size 1..1000, workers 1..8, timeout > 0')
    collect(args)


if __name__ == '__main__':
    main()

