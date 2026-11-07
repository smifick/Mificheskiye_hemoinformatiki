"""Regression tests for endpoint semantics and test-structure exclusion."""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import collect_chembl_research as C


class ChEMBLResearchTests(unittest.TestCase):
    def test_explicit_ratio_definitions_and_ambiguity(self):
        self.assertEqual(C.si_definition('Selectivity index, CC50 in HFF cell/ EC50 against vaccinia virus'), 'CC50/EC50')
        self.assertEqual(C.si_definition('CC50 / IC50 against VACV'), 'CC50/IC50')
        self.assertEqual(C.si_definition('minimum toxic concentration / inhibitory dose'), 'unverified_definition')
        self.assertEqual(C.si_definition('Ratio of effective concentration to half maximal cytotoxic concentration'), 'unverified_definition')

    def test_preserve_stereochemistry_and_reject_empty_graph(self):
        a = C.graph_keys('N[C@H](C)C(=O)O')
        b = C.graph_keys('N[C@@H](C)C(=O)O')
        self.assertNotEqual(a['canonical_smiles'], b['canonical_smiles'])
        self.assertNotEqual(a['InChIKey'], b['InChIKey'])
        self.assertIsNone(C.graph_keys(''))
        self.assertIsNone(C.graph_keys('not_a_smiles'))

    def test_published_bank_has_no_test_labels_or_external_codes(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'data/external_research/chembl/endpoint_bank.csv'
        if not path.exists():
            self.skipTest('Run collect_chembl_research.py first')
        bank = pd.read_csv(path)
        test = {C.T.taut_key(C.T.std(s)) for s in pd.read_csv(root/'data/raw/test.csv', usecols=['SMILES']).SMILES}
        self.assertTrue(set(bank.tautomer_parent).isdisjoint(test))
        self.assertFalse(bank.is_organizer_test_structure.any())
        self.assertTrue(bank.endpoint_quality_status.eq('valid_endpoint').all())
        self.assertTrue(bank.relation.eq('=').all())
        self.assertTrue(set(bank.task).issubset({'VACV_IC50','VACV_EC50','cytotoxicity_CC50','reported_CC50/EC50','reported_CC50/IC50'}))
        self.assertTrue(bank.source_url.str.startswith(C.D.BASE+'activity/').all())
        self.assertEqual(bank.activity_id.nunique(), len(bank))
        self.assertEqual(set(bank.license), {'CC-BY-SA-3.0'})


if __name__=='__main__':unittest.main()
