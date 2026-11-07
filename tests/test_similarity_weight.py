"""Numeric regression tests for the isolated two-feature SI mixing gate."""
from pathlib import Path
import copy,json,sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import similarity_weight as G


def table():
    return pd.DataFrame(dict(row_id=[17,4,81,3,92,15],
        pred_S=[9000.,3000.,7000.,500.,6000.,2000.],
        pred_M=[1000.,7000.,4000.,2500.,9000.,8000.],
        s=[.9,.2,.6,.4,.8,.1],d=[.8,.4,.3,.2,.3,.6],
        lookup_applied=[False,False,False,False,True,True],
        lookup_value=[np.nan,np.nan,np.nan,np.nan,-50.,15000.],
        observed=[8000.,6500.,6000.,1000.,0.,10000.]),index=[2,8,1,4,3,9])


def params(**changes):
    result=dict(w0=.3,s0=.5,d0=.4,beta_s=0.,beta_d=0.,alpha_source=0.,disagreement_scale=10000.)
    result.update(changes);return result


def test_zero_coefficients_reproduce_exact_postexpert_constant_mixture():
    t=table();p,w=G.predict_gate(t,params())
    expected=.3*t.pred_S.to_numpy()+.7*t.pred_M.to_numpy()
    expected[t.lookup_applied]=[0.,10000.]
    np.testing.assert_allclose(p,expected,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(w,np.full(len(t),.3))


def test_closed_form_constant_accounts_for_lookup_and_uses_unchanged_targets():
    t=table();model=G.fit_gate(t,G.specs()[0])
    delta=np.where(t.lookup_applied,0,t.pred_S-t.pred_M)
    baseline=np.where(t.lookup_applied,np.clip(t.lookup_value,0,10000),t.pred_M)
    expected=float(np.clip(delta@(t.observed-baseline)/(delta@delta),0,1))
    assert model['w0']==expected and model['beta_s']==model['beta_d']==0
    assert model['alpha_source']==0


@pytest.mark.parametrize('w0',[0.,1.])
def test_boundary_constant_weight_does_not_prevent_departure(w0):
    t=table();_,w=G.predict_gate(t,params(w0=w0,beta_s=.5))
    assert np.any(abs(w-w0)>0)
    assert np.isfinite(w).all() and ((w>=0)&(w<=1)).all()


def test_row_permutation_batch_composition_and_labels_do_not_affect_inference():
    t=table();model=G.fit_gate(t,dict(family='sd',penalty=.1))
    p,w=G.predict_gate(t,model);order=[5,1,4,0,3,2]
    modified=t.iloc[order].copy();modified['observed']='NOT AN INPUT'
    actual,aw=G.predict_gate(modified,model)
    np.testing.assert_array_equal(actual,p[order]);np.testing.assert_array_equal(aw,w[order])
    for i in range(len(t)):
        pp,ww=G.predict_gate(t.iloc[[i]].drop(columns='observed'),model)
        np.testing.assert_array_equal(pp,p[[i]]);np.testing.assert_array_equal(ww,w[[i]])
    assert modified.row_id.to_list()==t.iloc[order].row_id.to_list()


@pytest.mark.parametrize('spec',G.specs())
def test_fitted_parameters_are_finite_bounded_json_serializable_and_input_unchanged(spec):
    t=table();before=t.copy(deep=True);spec_before=copy.deepcopy(spec)
    model=G.fit_gate(t,spec)
    pd.testing.assert_frame_equal(t,before);assert spec==spec_before
    assert 0<=model['w0']<=1 and 0<=model['beta_s']<=.5 and -.5<=model['beta_d']<=.5
    p,w=G.predict_gate(t,model)
    assert np.isfinite(p).all() and np.isfinite(w).all() and ((w>=0)&(w<=1)).all()
    assert p[-2]==0 and p[-1]==10000
    restored=json.loads(json.dumps(model,allow_nan=False));rp,rw=G.predict_gate(t,restored)
    np.testing.assert_array_equal(rp,p);np.testing.assert_array_equal(rw,w)
    constant=G.fit_gate(t,G.specs()[0])
    assert model['penalized_objective']<=constant['penalized_objective']+1e-12


def test_centers_are_training_only_and_reused_at_prediction():
    t=table();model=G.fit_gate(t,dict(family='sd',penalty=.01))
    assert model['s0']==t.s.mean() and model['d0']==t.d.mean()
    changed=t.copy();changed['s']=1.;changed['d']=0.
    G.predict_gate(changed,model)
    assert model['s0']==t.s.mean() and model['d0']==t.d.mean()


def test_identical_experts_or_only_lookup_rows_fall_back_to_constant_zero():
    for mode in ('same','lookup'):
        t=table()
        if mode=='same':t['pred_S']=t.pred_M
        else:t['lookup_applied']=True;t['lookup_value']=100.
        model=G.fit_gate(t,dict(family='sd',penalty=.01))
        assert model['w0']==model['beta_s']==model['beta_d']==0


def test_source_softening_and_nonfinite_data_are_rejected():
    with pytest.raises(ValueError):G.predict_gate(table(),params(alpha_source=.1))
    bad=table();bad.iloc[0,bad.columns.get_loc('s')]=np.nan
    with pytest.raises(ValueError):G.predict_gate(bad,params())
    bad=table();bad['observed']=np.nan
    with pytest.raises(ValueError):G.fit_gate(bad,G.specs()[0])


def test_fitted_gate_can_improve_a_real_directed_similarity_pattern():
    s=np.linspace(.1,.9,60);m=np.full(60,2000.);expert_s=np.full(60,8000.)
    expected_weight=np.clip(.3+.4*(s-.5),0,1)
    t=pd.DataFrame(dict(pred_S=expert_s,pred_M=m,s=s,d=.6,lookup_applied=False,
                        lookup_value=np.nan,observed=m+expected_weight*(expert_s-m)))
    constant=G.fit_gate(t,G.specs()[0]);directed=G.fit_gate(t,dict(family='s',penalty=.01))
    assert directed['beta_s']>0 and directed['training_MSE']<constant['training_MSE']*.05


@pytest.mark.parametrize('boundary',[0.,1.])
def test_optimization_can_leave_a_boundary_anchor(boundary):
    s=np.linspace(.1,.9,60);m=np.full(60,2000.);expert_s=np.full(60,8000.)
    target_weight=s-.6 if boundary==0 else 1+s-.4
    t=pd.DataFrame(dict(pred_S=expert_s,pred_M=m,s=s,d=.6,lookup_applied=False,
                        lookup_value=np.nan,observed=m+target_weight*(expert_s-m)))
    constant=G.fit_gate(t,G.specs()[0]);directed=G.fit_gate(t,dict(family='s',penalty=.01))
    assert constant['w0']==directed['w0']==boundary
    assert directed['beta_s']>0
    assert directed['training_MSE']<constant['training_MSE']
