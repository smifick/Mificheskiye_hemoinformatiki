"""Finite SMILES-only technical-code gate candidates for nested PR3 research.

The fit/predict interface uses state['y'], state['flag'], and original raw
state['smiles']; no organizer assay metadata or concentration enters any head.
Notation statistics are acquisition proxies, not causal biological features.
"""
import numpy as np
from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from smiles_style import raw_style_features


def extended_notation_features(smiles):
    """Deterministic counts and notation equality tests; no fitted statistics."""
    original = raw_style_features(smiles)
    prefixes = ('C', 'c', 'N', 'n', 'O', 'o', 'S', '[', 'F', 'Cl', 'Br')
    tokens = ('Cl', 'Br', 'F', 'I', 'N', 'O', 'P', 'S', 'n', 'o', 's',
              '1', '2', '3', '4', '5', '%', '#', '/', '\\', '.', '+', '-',
              '@@', '[C@H]', '[C@@H]', '[nH]', 'H]', '(=O)', '=O')
    positions = ('[', '@', 'c', '(', '=', '.', '1', '#')
    rows = []
    for s in smiles:
        n = max(len(s), 1)
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError('Invalid SMILES')
        try:
            kekule = Chem.MolToSmiles(mol, kekuleSmiles=True)
        except (ValueError, RuntimeError):
            kekule = None
        rows.append([float(s.startswith(x)) for x in prefixes]
                    + [s.count(x)/n for x in tokens]
                    + [(s.find(x)+1)/n for x in positions]
                    + [float(s == s.strip()),
                       float(s == Chem.MolToSmiles(mol, isomericSmiles=False)),
                       float(s == kekule),
                       float(any(a.GetIsAromatic() for a in mol.GetAtoms())),
                       mol.GetNumAtoms()/n, mol.GetNumHeavyAtoms()/n])
    return np.column_stack([original, np.asarray(rows, float)])


class NotationFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, extended=False):
        self.extended = extended

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return extended_notation_features(x) if self.extended else raw_style_features(x)


def candidate_specs():
    pool = [dict(name='style9_C1', family='style', C=1.)]
    pool += [dict(name=f'extended_style_C{c:g}', family='extended', C=c)
             for c in (.1, 1., 10.)]
    for family in ('char', 'char_style'):
        pool += [dict(name=f'{family}_C{c:g}', family=family, C=c)
                 for c in (1., 10., 100.)]
    pool += [dict(name='cat_style_d2_l20', family='catstyle', depth=2, l2=20.),
             dict(name='cat_style_d3_l50', family='catstyle', depth=3, l2=50.)]
    return pool


def make_estimator(spec):
    family = spec['family']
    # lbfgs preserves the old style9 control's intercept treatment exactly.
    lr = LogisticRegression(C=spec.get('C', 1.), max_iter=2000, random_state=42)
    if family in ('style', 'extended'):
        return make_pipeline(NotationFeatures(family == 'extended'), StandardScaler(), lr)
    char = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), min_df=2,
                          max_features=6000, sublinear_tf=True, lowercase=False)
    if family == 'char':
        return make_pipeline(char, lr)
    if family == 'char_style':
        features = FeatureUnion([('char', char),
                                 ('style', make_pipeline(NotationFeatures(True), StandardScaler()))])
        return make_pipeline(features, lr)
    if family == 'catstyle':
        from catboost import CatBoostClassifier
        return make_pipeline(NotationFeatures(True), CatBoostClassifier(
            iterations=300, depth=spec['depth'], l2_leaf_reg=spec['l2'],
            learning_rate=.035, loss_function='Logloss', random_seed=314,
            thread_count=1, allow_writing_files=False, verbose=False))
    raise ValueError(f'Unknown family: {family}')


def fit_gate(spec, state, train):
    """Fit head on unflagged training labels, flag rate on flagged training only."""
    y, flag = state['y'], state['flag']
    c = (y == 10000).astype(int)
    fit = train[~flag[train]]
    flagged = train[flag[train]]
    model = dict(spec=spec, p_flag=float((c[flagged].sum()+.5)/(len(flagged)+2)))
    if not len(fit) or len(np.unique(c[fit])) < 2:
        model['constant'] = float(c[fit].mean()) if len(fit) else float(c[train].mean())
    else:
        estimator = make_estimator(spec)
        estimator.fit(state['smiles'][fit], c[fit])
        model['estimator'] = estimator
    return model


def predict_gate(model, state, rows):
    if 'constant' in model:
        probability = np.full(len(rows), model['constant'])
    else:
        probability = model['estimator'].predict_proba(state['smiles'][rows])[:, 1]
    return np.where(state['flag'][rows], model['p_flag'], probability)
