"""Semantic checks for the PubChem pilot, independent of network availability."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import collect_pubchem_research as P


class PubChemResearchTests(unittest.TestCase):
    def test_missing_activity_is_not_a_numeric_zero_or_an_SI(self):
        raw = (b'PUBCHEM_RESULT_TAG,PUBCHEM_SID,PUBCHEM_CID,PUBCHEM_EXT_DATASOURCE_SMILES,PUBCHEM_ACTIVITY_OUTCOME,AC50_Qualifier,AC50_uM\n'
               b'RESULT_UNIT,,,,,,MICROMOLAR\n'
               b'1,11,1,CCO,Inactive,,\n'
               b'2,12,2,CCN,Active,=,3\n'
               b'3,13,3,CCC,Inactive,>,30\n')
        meta = {'name': 'VACV reporter', 'cell_line': '',
                'depositor': 'Test', 'source_url': 'https://pubchem.ncbi.nlm.nih.gov/bioassay/1'}
        rows, bad, count = P.parse_dose(raw, meta, 1)
        self.assertEqual(count, 3)
        self.assertFalse(bad)
        self.assertIsNone(rows[0]['value_uM'])
        self.assertEqual(rows[0]['validation_status'], 'activity_outcome_only')
        self.assertEqual(rows[1]['value_uM'], 3)
        self.assertEqual(rows[2]['relation'], '>')
        self.assertTrue(all(r['endpoint'] == 'AC50' and 'SI' not in r for r in rows))

    def test_duplicate_result_tags_are_rejected(self):
        raw = b'PUBCHEM_RESULT_TAG,PUBCHEM_SID\n1,11\n1,12\n'
        with self.assertRaises(ValueError):
            P.csv_records(raw)

    def test_protocols_and_background_references_are_not_exported(self):
        a = {'aid': {'id': 1, 'version': 2}, 'name': 'A549 Cytotoxicity',
             'aid_source': {'db': {'name': 'NCGC'}},
             'protocol': ['not an exported field'], 'description': ['not exported'],
             'xref': [{'xref': {'pmid': 123}}]}
        exported = P.short_metadata(a, 'auxiliary')
        self.assertNotIn('protocol', exported)
        self.assertNotIn('description', exported)
        self.assertNotIn('doi', exported)
        self.assertEqual(exported['cell_line'], 'A549')

    def test_cache_integrity_is_checked_before_reuse(self):
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1/sids/JSON'
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            key = hashlib.sha256(url.encode()).hexdigest()
            Path(folder, key + '.json').write_text('{"bad":true}')
            Path(folder, key + '.meta.json').write_text(json.dumps({'sha256': 'wrong'}))
            with self.assertRaisesRegex(ValueError, 'Corrupt cached'):
                P.CachedHTTP(folder).get(url)


if __name__ == '__main__':
    unittest.main()
