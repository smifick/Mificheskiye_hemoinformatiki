"""Numerical invariants for the single PR3 source-routing intervention."""
from copy import deepcopy
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import soft_source as S
from smiles_style import raw_style_features


def table_fixture():
    # The fourth row saturates the complete predictor; the last two are
    # overridden by external values outside the output interval.
    return pd.DataFrame(dict(
        row_id=[41,7,99,12,73,8],
        source_present=[True,False,True,True,True,False],
        physical_prediction=[20.,30.,500.,40000.,900.,100.],
        zero_rate=[.01,.02,.15,0.,.01,.01],
        structure_group=['a','b','c','d','e','f'],
        scaffold_group=['g0','g1','g2','g3','g4','g5'],
        p_molecular_notation=[.8,.9,.1,.9,.7,.2],
        p_molecular_mlm=[.6,.3,.5,.7,.2,.8],
        p_group_notation=[.1,.1,.4,.1,.1,.1],
        p_group_mlm=[.1,.1,.4,.1,.1,.1],
        notation_weight=.5,mlm_weight=.5,
        lookup_applied=[False,False,False,False,True,True],
        lookup_value=[np.nan,np.nan,np.nan,np.nan,-7.,20000.],
    ),index=[30,10,50,20,70,60])


def test_alpha_zero_is_hard_override_with_exact_complete_and_lookup():
    t=table_fixture()
    final,p,leaves,numeric=S.apply_alpha(t,0,details=True)
    expected_p=np.array([.1,.6,.4,.1,.1,.5])
    expected_leaves=np.array([[.1,.1],[.9,.3],[.4,.4],[.1,.1],[.1,.1],[.2,.8]])
    # Includes inner probability clipping, final saturation and lookup clipping.
    expected_numeric=np.array([1017.8,6011.4,4225.,10000.,1801.,5049.])
    np.testing.assert_allclose(p,expected_p,rtol=0,atol=1e-14)
    np.testing.assert_allclose(leaves,expected_leaves,rtol=0,atol=1e-14)
    np.testing.assert_allclose(numeric,expected_numeric,rtol=0,atol=1e-10)
    np.testing.assert_allclose(final,np.array([1017.8,6011.4,4225.,10000.,0.,10000.]),rtol=0,atol=1e-10)


def test_alpha_one_restores_individual_leaf_probabilities():
    t=table_fixture();_,p,leaves,_=S.apply_alpha(t,1,details=True)
    expected=t[['p_molecular_notation','p_molecular_mlm']].to_numpy()
    np.testing.assert_array_equal(leaves,expected)
    np.testing.assert_allclose(p,expected.mean(axis=1),atol=1e-15)


@pytest.mark.parametrize('alpha',list(S.GRID)+[1.])
def test_interpolation_bounds_and_unaffected_rows(alpha):
    t=table_fixture();before=t.copy(deep=True)
    baseline=S.apply_alpha(t,0)
    final,p,leaves,_=S.apply_alpha(t,alpha,details=True)
    for col,h in enumerate(S.HEADS):
        pm=t['p_molecular_'+h].to_numpy();pg=t['p_group_'+h].to_numpy()
        source=t.source_present.to_numpy()
        assert np.all(leaves[source,col]>=np.minimum(pm,pg)[source]-1e-15)
        assert np.all(leaves[source,col]<=np.maximum(pm,pg)[source]+1e-15)
        np.testing.assert_array_equal(leaves[~source,col],pm[~source])
    unchanged=(~t.source_present.to_numpy())|t.lookup_applied.to_numpy()
    np.testing.assert_array_equal(final[unchanged],baseline[unchanged])
    assert np.isfinite(final).all() and np.isfinite(p).all() and np.isfinite(leaves).all()
    assert ((leaves>=0)&(leaves<=1)).all() and ((final>=0)&(final<=10000)).all()
    pd.testing.assert_frame_equal(t,before)


def test_alpha_is_applied_once_instead_of_squared_at_wrapper():
    t=table_fixture().iloc[:1]
    _,p,leaves,_=S.apply_alpha(t,.2,details=True)
    np.testing.assert_allclose(leaves,[[.24,.20]],atol=1e-15)
    np.testing.assert_allclose(p,[.22],atol=1e-15)
    assert abs(p[0]-(.1+.2**2*(.7-.1)))>.09


def test_row_ids_and_permutation_are_preserved():
    t=table_fixture();order=[4,0,5,2,1,3]
    expected=S.apply_alpha(t,.1);shuffled=t.iloc[order]
    original_ids=shuffled.row_id.to_list();original_index=shuffled.index.to_list()
    actual=S.apply_alpha(shuffled,.1)
    np.testing.assert_array_equal(actual,expected[order])
    assert shuffled.row_id.to_list()==original_ids and shuffled.index.to_list()==original_index


def test_invalid_alpha_is_rejected():
    for alpha in [-.1,1.01,np.nan]:
        with pytest.raises(ValueError):S.apply_alpha(table_fixture(),alpha)


def test_mismatched_group_probability_is_not_averaged():
    leaf=lambda pg,p:dict(spec=dict(family='logistic',embedding='x'),p_flag=pg,constant=p)
    base=dict(spec=dict(name=S.SPEC['name']),p_flag=.1,
              components=[leaf(.1,.8),leaf(.1,.6)])
    raw,groups=S.leaf_probabilities(base,{'x':np.ones((2,1))},np.arange(2))
    assert groups==[.1,.1]
    np.testing.assert_array_equal(raw,np.array([[.8,.8],[.6,.6]]))
    child_changed=deepcopy(base);child_changed['components'][1]['p_flag']=.2
    parent_changed=deepcopy(base);parent_changed['p_flag']=.2
    for bad in [child_changed,parent_changed]:
        with pytest.raises(AssertionError):S.leaf_probabilities(bad,{'x':np.ones((2,1))},np.arange(2))


def test_no_effect_or_insufficient_groups_falls_back_to_zero():
    t=table_fixture();t['observed']=S.apply_alpha(t,.2)
    t['lookup_applied']=True;t['lookup_value']=500.
    chosen,info=S.choose_alpha(t)
    assert chosen==0 and not info['information_sufficient']
    t=table_fixture().iloc[:1].copy();t['observed']=S.apply_alpha(t,.2)
    chosen,info=S.choose_alpha(t)
    assert chosen==0 and not info['information_sufficient']


def test_selection_uses_the_declared_small_grid_and_all_row_mse():
    t=table_fixture();t['observed']=S.apply_alpha(t,1)
    chosen,info=S.choose_alpha(t)
    assert chosen in S.GRID and 1. not in S.GRID
    assert info['information_sufficient']
    for alpha in S.GRID:
        expected=np.mean((S.apply_alpha(t,alpha)-t.observed.to_numpy())**2)
        assert info['MSE'][str(alpha)]==expected


@pytest.mark.skipif(not (ROOT/'work/soft_source_cache/mlm_raw.npy').exists(),reason='Frozen local features not prepared')
def test_actual_production_replay_and_model_hashes_unchanged():
    paths=[ROOT/'results/audit_20261107/selected_model.joblib',ROOT/'results/v7_final/model.joblib']
    hashes_before={str(p):S.sha(p) for p in paths}
    bundle=joblib.load(paths[0])
    train_count=len(pd.read_csv(ROOT/'data/raw/train.csv',usecols=['SMILES']))
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['index','SMILES'])
    emb=dict(raw_style=raw_style_features(test.SMILES),
             mlm_raw=np.load(ROOT/'work/soft_source_cache/mlm_raw.npy')[train_count:])
    exported=S.production_components(bundle,list(test.SMILES),emb)
    exported.insert(0,'index',test['index'].to_numpy())
    reference=pd.read_csv(ROOT/'results/audit_20261107/submission.csv')
    np.testing.assert_allclose(S.apply_alpha(exported,0),reference['SI VACV'],rtol=0,atol=1e-8)
    assert exported['index'].to_list()==reference['index'].to_list()
    counts=S.effect_counts(exported,.2)
    assert counts['source_present']==196 and counts['lookup_applied']==39
    assert counts['effective_rows']==157 and counts['effective_structures']==145
    assert counts['effective_scaffolds']==74
    for alpha in list(S.GRID)+[1.]:
        final,p,leaves,_=S.apply_alpha(exported,alpha,details=True)
        assert np.isfinite(final).all() and np.isfinite(p).all() and np.isfinite(leaves).all()
        lookup=exported.lookup_applied.to_numpy()
        np.testing.assert_array_equal(final[lookup],S.apply_alpha(exported,0)[lookup])
    assert {str(p):S.sha(p) for p in paths}==hashes_before
