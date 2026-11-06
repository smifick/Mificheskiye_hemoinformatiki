import sys,unittest
from pathlib import Path
import pandas as pd,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from download_external import duration
from train_external_v3 import canonical

class ExternalCurationTests(unittest.TestCase):
 def test_days_and_hours_metadata_agree(self):
  self.assertEqual(duration('after 4 days'),'96');self.assertEqual(duration('incubated 96 hrs'),'96')
 def test_reviewed_pairs_keep_cell_context(self):
  p=Path(__file__).resolve().parents[1];d=pd.read_csv(p/'data/external_curated/vacv_si.csv');ep=pd.read_csv(p/'data/external/external_endpoints.csv').set_index('activity_id')
  for _,r in d.iterrows():
   self.assertEqual(ep.loc[r.antiviral_activity_id,'cell_line'],r.cell_line)
   self.assertEqual(ep.loc[r.cytotoxicity_activity_id,'cell_line'],r.cell_line)
  np.testing.assert_allclose(d.CC50_uM/d.IC50_uM,d.SI)
 def test_tautomers_have_same_representation(self):
  self.assertEqual(canonical('O=c1cccc[nH]1',True)[0],canonical('Oc1ccccn1',True)[0])

if __name__=='__main__':unittest.main()
