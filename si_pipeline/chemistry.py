"""Shared, deterministic molecule normalization for collection and inference."""
from functools import lru_cache
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.warning')


@lru_cache(maxsize=32768)
def canonicalize(smiles, tautomers=True):
    """Keep the largest organic parent, uncharge, preserve specified stereo."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('Missing SMILES')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES')
    parts = Chem.GetMolFrags(mol, asMols=True)
    mol = max(parts, key=lambda m: (any(a.GetAtomicNum() == 6 for a in m.GetAtoms()), m.GetNumHeavyAtoms()))
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        raise ValueError('No organic parent')
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    if tautomers:
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def structure_keys(smiles):
    canonical = canonicalize(smiles)
    mol = Chem.MolFromSmiles(canonical)
    parent = Chem.MolToSmiles(mol, isomericSmiles=False)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    group = (Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold), isomericSmiles=False)
             if scaffold.GetNumAtoms() else 'ACYCLIC')
    return canonical, parent, group
