"""Features with a fixed column schema and no response-dependent fitting."""
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from sklearn.base import BaseEstimator, TransformerMixin
from .chemistry import canonicalize


class MolecularFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, morgan_bits=2048, radius=2, mordred=False, tautomers=True):
        self.morgan_bits = morgan_bits
        self.radius = radius
        self.mordred = mordred
        self.tautomers = tautomers

    def fit(self, smiles, y=None):
        self.rdkit_version_ = rdkit.__version__
        self.descriptor_names_ = tuple(name for name, _ in Descriptors._descList)
        names = [f'rdkit:{name}' for name in self.descriptor_names_]
        names += [f'maccs:{i}' for i in range(167)]
        names += [f'morgan_count:{i}' for i in range(self.morgan_bits)]
        self.mordred_names_ = ()
        if self.mordred:
            from mordred import Calculator, descriptors
            calc = Calculator(descriptors, ignore_3D=True)
            self.mordred_names_ = tuple(str(d) for d in calc.descriptors)
            names += [f'mordred:{name}' for name in self.mordred_names_]
        self.feature_names_ = tuple(names)
        return self

    def transform(self, smiles):
        if rdkit.__version__ != self.rdkit_version_:
            raise ValueError('RDKit version differs from training; use the saved environment lock')
        functions = dict(Descriptors._descList)
        if tuple(functions) != self.descriptor_names_:
            raise ValueError('RDKit descriptor schema changed')
        canon = [canonicalize(str(s), self.tautomers) for s in smiles]
        mols = [Chem.MolFromSmiles(s) for s in canon]
        result = np.full((len(mols), len(self.feature_names_)), np.nan, dtype=np.float64)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.morgan_bits, includeChirality=True)
        n = len(self.descriptor_names_)
        for i, mol in enumerate(mols):
            for j, name in enumerate(self.descriptor_names_):
                try:
                    result[i, j] = float(functions[name](mol))
                except (ValueError, TypeError, OverflowError, RuntimeError):
                    pass
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), result[i, n:n+167])
            counts = np.zeros(self.morgan_bits)
            DataStructs.ConvertToNumpyArray(gen.GetCountFingerprint(mol), counts)
            result[i, n+167:n+167+self.morgan_bits] = np.log1p(counts)
        if self.mordred:
            from mordred import Calculator, descriptors
            calc = Calculator(descriptors, ignore_3D=True)
            if tuple(str(d) for d in calc.descriptors) != self.mordred_names_:
                raise ValueError('Mordred descriptor schema changed')
            offset = n + 167 + self.morgan_bits
            for i, mol in enumerate(mols):
                for j, value in enumerate(calc(mol)):
                    try:
                        result[i, offset+j] = float(value)
                    except (ValueError, TypeError, OverflowError):
                        pass
        result[~np.isfinite(result)] = np.nan
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)
