"""Guard the errors that turn published rows into incorrect training labels."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import collect_pmc_research as P


class FakeJSON:
    def __init__(self,answers):self.answers=iter(answers)
    def json(self,url):return next(self.answers)


class TestPMCTables(unittest.TestCase):
    def test_bounds_do_not_become_exact_values(self):
        mean,relation,um=P.number('>100 nM')
        self.assertEqual(relation,'>');self.assertAlmostEqual(um,.1)
        self.assertEqual(P.number('not determined')[1],'missing')
        self.assertEqual(P.number('0')[1],'invalid')
        self.assertAlmostEqual(P.number('7.9 ± 0.5 nM')[2],.0079)

    def test_selected_table_bounds_excluded_and_no_unrelated_table(self):
        raw=b'''<article><front><article-meta>
        <article-id pub-id-type="doi">10.1099/jgv.0.001971</article-id>
        <permissions>Creative Commons Attribution</permissions>
        </article-meta></front><body>
        <table-wrap id="unrelated"><table><tbody><tr><td>ignore</td></tr></tbody></table></table-wrap>
        <table-wrap id="T2"><table><thead><tr><td>Compound</td><td>IC50</td><td>CC50</td><td>SI</td><td>x</td><td>y</td></tr></thead>
        <tbody><tr><td>Bufalin</td><td>1</td><td>&gt;100</td><td>&gt;100</td><td>0</td><td>0</td></tr></tbody></table></table-wrap>
        </body></article>'''
        rows,evidence,_=P.extract_rows(raw,'PMC10995631',P.SOURCES['PMC10995631'])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['decision'],'excluded_bound_or_missing')
        self.assertEqual(rows[0]['cell_line'],'HeLa')
        self.assertNotIn('unrelated',ET.tostring(evidence,encoding='unicode'))

    def test_multi_CID_is_review_not_first_hit(self):
        with self.assertRaisesRegex(ValueError,'Nonunique'):
            P.exact_structure(FakeJSON([{'IdentifierList':{'CID':[1,2]}}]),'compound')

    def test_near_synonym_is_rejected(self):
        api=FakeJSON([{'IdentifierList':{'CID':[1]}},
                      {'InformationList':{'Information':[{'Synonym':['compound sodium']} ]}}])
        with self.assertRaisesRegex(ValueError,'exact CID synonyms'):
            P.exact_structure(api,'compound')

    def test_inchikey_mismatch_rejected(self):
        api=FakeJSON([{'IdentifierList':{'CID':[1]}},
                      {'InformationList':{'Information':[{'Synonym':['compound']}]}},
                      {'PropertyTable':{'Properties':[{'CID':1,'SMILES':'CCO','InChIKey':'WRONG'}]}}])
        with self.assertRaisesRegex(ValueError,'SMILES/InChIKey'):
            P.exact_structure(api,'compound')

    def test_truncated_search_is_not_reported_complete(self):
        api=FakeJSON([{'hitCount':2,'resultList':{'result':[{'source':'MED','id':'1'}]}}])
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):
            P.search(api,'query')

    def test_corrupted_raw_cache_rejected(self):
        url=P.API+'test'
        with tempfile.TemporaryDirectory() as directory:
            key=hashlib.sha256(url.encode()).hexdigest()
            path=Path(directory)/(key+'.raw');path.write_bytes(b'changed')
            path.with_suffix('.json').write_text(json.dumps({'sha256':hashlib.sha256(b'original').hexdigest()}))
            with self.assertRaisesRegex(ValueError,'SHA256'):
                P.CachedHTTP(directory).get(url)


if __name__=='__main__':unittest.main()
