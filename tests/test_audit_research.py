"""Regression checks for audited alignment and fold-label isolation."""
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import train_v5 as T
from audit_regression import empirical_floor
from research_heads import fit_head,predict_head
from smiles_style import raw_style_features


class AuditResearchTests(unittest.TestCase):
    def test_empirical_bound_is_group_conditional(self):
        result,_=empirical_floor(['a','a','b'],np.array([0.,2.,5.]))
        self.assertAlmostEqual(result['minimum_empirical_MSE'],2/3)
        result,_=empirical_floor(['a'],np.array([36000.]))
        self.assertEqual(result['minimum_empirical_MSE'],0.)
        self.assertEqual(result['minimum_empirical_MSE_with_output_cap_10000'],26000.**2)

    def test_world_offset_follows_row_counts_and_invalid_external_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data/raw').mkdir(parents=True)
            pd.DataFrame({'SMILES':['CCO','CCC','CCN'],'SI VACV':[1.,2.,10000.]}).to_csv(root/'data/raw/train.csv',index=False)
            pd.DataFrame({'index':[0,1],'SMILES':['CCCl','CCBr']}).to_csv(root/'data/raw/test.csv',index=False)
            ext=pd.DataFrame({'SMILES':['invalid!','CCF'],'SI':[999.,5.],
                              'doi':['x','y'],'cell':['a','b'],'tier':['V','V']})
            with patch.object(T,'ROOT',root),patch.object(T,'load_external',return_value=ext):
                w=T.World()
            self.assertEqual((w.n_tr,w.n_te,w.n_ext),(3,2,1))
            np.testing.assert_array_equal(w.ext_ids,[5])
            np.testing.assert_array_equal(w.ext_si,[5.])
            self.assertEqual(w.K.shape,(6,6))
            self.assertEqual(w.tau[5],T.taut_key(T.std('CCF')))
            np.testing.assert_array_equal(w.y,[1.,2.,10000.])

    def test_import_does_not_create_result_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'src').mkdir()
            path=root/'src/train_v5.py'
            shutil.copyfile(Path(T.__file__),path)
            spec=importlib.util.spec_from_file_location('audited_v5_import',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.assertFalse((root/'results').exists())

    def test_head_training_does_not_read_outer_validation_labels(self):
        rng=np.random.default_rng(2026)
        x=rng.normal(size=(14,7));kernel=x@x.T+np.eye(14)
        y=np.array([0,2,10000,5,10000,6,7,10000,8,9,1,3,10000,4.])
        flag=np.array([1,1,0,0,0,0,0,0,0,0,1,0,0,0],bool)
        world=SimpleNamespace(y=y,K=kernel)
        state={'y':y,'flag':flag,'world':world,'K_feature':kernel,'groups':np.array([str(i) for i in range(14)])}
        embeddings={'mlm_raw':rng.normal(size=(14,9))}
        tr=np.arange(10);va=np.arange(10,14)
        specs=[dict(name='linear',family='logistic',embedding='mlm_raw',C=.1),
               dict(name='kernel',family='fusion',embedding='mlm_raw',alpha=.5),
               dict(name='calibration',family='calibrated',C=1.)]
        specs.append(dict(name='paired',family='paired',components=specs[:2]))
        original=y.copy()
        for spec in specs:
            y[:]=original
            a=fit_head(spec,state,embeddings,tr);pa=predict_head(a,state,embeddings,va,tr)
            y[va]=[10000,0,36000,10000]
            b=fit_head(spec,state,embeddings,tr);pb=predict_head(b,state,embeddings,va,tr)
            np.testing.assert_allclose(pa,pb)

    def test_notation_features_preserve_written_form(self):
        from rdkit import Chem
        a='c1ccccc1';b='C1=CC=CC=C1'
        self.assertEqual(Chem.MolToSmiles(Chem.MolFromSmiles(a)),Chem.MolToSmiles(Chem.MolFromSmiles(b)))
        values=raw_style_features([a,b])
        self.assertEqual(values.shape,(2,9))
        self.assertFalse(np.array_equal(values[0],values[1]))


if __name__=='__main__':unittest.main()
