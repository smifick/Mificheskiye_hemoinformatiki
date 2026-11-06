"""Deterministic notation statistics, derived strictly from raw SMILES text.

These are acquisition/notation proxies as well as coarse structural counts;
they must not be interpreted as causal determinants of biological activity.
"""
import numpy as np
from rdkit import Chem

FEATURE_NAMES=('is_rdkit_canonical','kekule_aromatic_notation','bracket_fraction',
               'stereo_marker_fraction','branch_fraction','double_bond_fraction',
               'aromatic_carbon_fraction','uppercase_carbon_fraction','string_length')


def raw_style_features(smiles):
    result=[]
    for text in smiles:
        if not isinstance(text,str):raise ValueError('SMILES must be a string')
        mol=Chem.MolFromSmiles(text)
        if mol is None:raise ValueError('Invalid SMILES')
        length=max(len(text),1)
        result.append([text==Chem.MolToSmiles(mol),
            any(a.GetIsAromatic() for a in mol.GetAtoms()) and not any(c in text for c in 'cnops'),
            text.count('[')/length,text.count('@')/length,text.count('(')/length,text.count('=')/length,
            text.count('c')/length,text.count('C')/length,length])
    return np.asarray(result,float).reshape(-1,len(FEATURE_NAMES))
