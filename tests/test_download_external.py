"""Offline regression checks for ChEMBL label integrity and API completeness."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from download_external import (BASE, Client, assay_decision, build_candidates,
                               duration, finite_number, normalize_record)


def activity(aid=1, endpoint='IC50', value='2000', unit='nM', **overrides):
    row = dict(activity_id=aid, assay_chembl_id='A', document_chembl_id='D',
               molecule_chembl_id='M', canonical_smiles='CCO',
               standard_type=endpoint, standard_value=value, standard_units=unit,
               standard_relation='=', potential_duplicate=False)
    return {**row, **overrides}


ASSAYS = {'A': {'assay_cell_type': 'Vero', 'description': 'after 4 days'}}
DOCS = {'D': {'doi': '10.1234/example'}}


def normalized(row):
    return normalize_record(row, ASSAYS, DOCS, {str(row['activity_id'])})


class EndpointIntegrityTests(unittest.TestCase):
    def test_unit_conversion_and_provenance(self):
        row = normalized(activity())
        self.assertEqual(row['IC50_uM'], 2)
        self.assertEqual(row['doi'], '10.1234/example')
        self.assertTrue(row['is_vacv_assay'])
        self.assertEqual(row['validation_status'], 'valid_endpoint')

    def test_si_zero_negative_nan_infinity_are_rejected(self):
        for value in ['0', '-1', 'nan', 'inf', None]:
            with self.subTest(value=value):
                row = normalized(activity(endpoint='Selectivity index', value=value, unit=''))
                self.assertNotEqual(row['validation_status'], 'valid_endpoint')
                self.assertIsNone(row['SI'])
                self.assertIsNone(finite_number(value))

    def test_no_raw_relation_fallback(self):
        row = normalized(activity(standard_relation=None, relation='='))
        self.assertIn('censored_or_unknown_relation', row['validation_status'])
        self.assertEqual(row['raw_relation'], '=')

    def test_unsupported_si_units_not_valid(self):
        row = normalized(activity(endpoint='SI', value='10', unit='nM'))
        self.assertIn('unsupported_si_units_or_value', row['validation_status'])

    def test_bounds_duplicates_and_validity_flags_do_not_form_pairs(self):
        for overrides in [dict(standard_relation='>'), dict(potential_duplicate=True),
                          dict(data_validity_comment='Outside typical range')]:
            with self.subTest(overrides=overrides):
                df = pd.DataFrame([normalized(activity(**overrides)),
                                   normalized(activity(2, 'CC50', '20', 'uM'))])
                pairs, _ = build_candidates(df)
                self.assertTrue(pairs.empty)

    def test_ec50_keeps_its_identity_and_pairs_require_review(self):
        df = pd.DataFrame([normalized(activity(endpoint='EC50')),
                           normalized(activity(2, 'CC50', '20', 'uM'))])
        pairs, direct = build_candidates(df)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(direct.empty)
        row = pairs.iloc[0]
        self.assertEqual(row.SI, 10)
        self.assertEqual(row.denominator_endpoint, 'EC50')
        self.assertEqual(row.si_definition, 'CC50/EC50')
        self.assertTrue(pd.isna(row.IC50_uM))
        self.assertEqual(row.validation_status, 'metadata_matched_manual_review_required')

    def test_no_pair_across_publications_or_molecules(self):
        antiviral = normalized(activity())
        for field in ['document_chembl_id', 'molecule_chembl_id']:
            cc = normalized(activity(2, 'CC50', '20', 'uM', **{field: 'other'}))
            pairs, _ = build_candidates(pd.DataFrame([antiviral, cc]))
            self.assertTrue(pairs.empty)

    def test_empty_candidates_keep_csv_schema(self):
        pairs, direct = build_candidates(pd.DataFrame([normalized(activity())]))
        self.assertTrue(pairs.empty)
        self.assertIn('denominator_endpoint', pairs)
        self.assertIn('SI', direct)

    def test_uncertain_antiviral_cannot_supply_training_candidates(self):
        row = normalize_record(activity(), ASSAYS, DOCS, set())
        cc = normalized(activity(2, 'CC50', '20', 'uM'))
        pairs, direct = build_candidates(pd.DataFrame([row, cc]))
        self.assertTrue(pairs.empty)
        self.assertTrue(direct.empty)

    def test_assay_disambiguation(self):
        targets = {'virus': {'target_type': 'ORGANISM', 'organism': 'Orthopoxvirus vaccinia'},
                   'enzyme': {'target_type': 'SINGLE PROTEIN', 'organism': 'Homo sapiens'},
                   'hiv': {'target_type': 'ORGANISM', 'organism': 'HIV-1'}}
        self.assertTrue(assay_decision({'assay_type': 'F', 'target_chembl_id': 'virus'}, targets).startswith('included_'))
        for target in ['enzyme', 'hiv']:
            self.assertFalse(assay_decision({'assay_type': 'F', 'target_chembl_id': target,
                                            'description': 'VACVase / vaccinia expression'}, targets).startswith('included_'))
        self.assertEqual(assay_decision({'assay_type': 'B', 'target_chembl_id': 'virus'}, targets),
                         'excluded_nonfunctional')

    def test_duration_compatibility(self):
        self.assertEqual(duration('after 4 days'), '96')
        self.assertEqual(duration('incubated 96 hrs'), '96')
        self.assertEqual(duration('after 2 h and 4 days'), '')


class PaginationTests(unittest.TestCase):
    def client(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return Client(temp.name)

    def test_all_pages_and_total_count(self):
        client = self.client()
        pages = [
            {'activities': [{'activity_id': 1}], 'page_meta': {'total_count': 2, 'next': '/chembl/api/data/activity.json?offset=1'}},
            {'activities': [{'activity_id': 2}], 'page_meta': {'total_count': 2, 'next': None}},
        ]
        with patch.object(client, 'get', side_effect=pages):
            self.assertEqual([r['activity_id'] for r in client.pages('activity')], [1, 2])

    def test_truncated_or_duplicate_pages_raise(self):
        for rows, count in [([{'activity_id': 1}], 2),
                            ([{'activity_id': 1}, {'activity_id': 1}], 2)]:
            client = self.client()
            with patch.object(client, 'get', return_value={
                    'activities': rows, 'page_meta': {'total_count': count, 'next': None}}):
                with self.assertRaises(RuntimeError):
                    client.pages('activity')

    def test_cache_hash_matches_response_and_no_second_network_call(self):
        client = self.client()
        raw = b'{"targets": [], "page_meta": {"total_count": 0, "next": null}}'
        with patch('download_external.urlopen', return_value=io.BytesIO(raw)) as network:
            self.assertEqual(client.pages('target'), [])
            self.assertEqual(client.pages('target'), [])
            self.assertEqual(network.call_count, 1)
        self.assertTrue(client.manifest[-1]['from_cache'])
        self.assertEqual(client.manifest[-1]['sha256'], hashlib.sha256(raw).hexdigest())

    def test_pagination_loop_and_release_count_change_raise(self):
        for second in [{'total_count': 2, 'next': BASE + 'activity.json?offset=1'},
                       {'total_count': 3, 'next': None}]:
            client = self.client()
            pages = [
                {'activities': [{'activity_id': 1}], 'page_meta': {'total_count': 2, 'next': BASE + 'activity.json?offset=1'}},
                {'activities': [{'activity_id': 2}], 'page_meta': second},
            ]
            with patch.object(client, 'get', side_effect=pages):
                with self.assertRaises(RuntimeError):
                    client.pages('activity')


if __name__ == '__main__':
    unittest.main()
