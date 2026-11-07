"""Small, directed similarity/disagreement gates for two frozen SI experts.

This module fits only the mixing weight. The experts, source overrides,
physical component and clipping belong to the unchanged expert exporters.
No target is inspected by predict_gate, and inference-batch statistics are
never used for centering or scaling.
"""
from itertools import product
import numpy as np
from scipy.optimize import minimize

D=10000.
PENALTIES=(.01,.1,1.)
FAMILIES=('constant','s','d','sd')


def specs():
    """A prespecified constant control and nine small directed alternatives."""
    return [dict(name='constant',family='constant',penalty=0.)]+[
        dict(name=f'{family}_lambda{penalty:g}',family=family,penalty=penalty)
        for family in ('s','d','sd') for penalty in PENALTIES]


def _arrays(table,training=False):
    s=np.asarray(table['s'],float);d=np.asarray(table['d'],float)
    expert_s=np.asarray(table['pred_S'],float);expert_m=np.asarray(table['pred_M'],float)
    lookup=np.asarray(table['lookup_applied'],bool)
    lookup_value=np.asarray(table['lookup_value'],float)
    if not all(np.isfinite(a).all() for a in (s,d,expert_s,expert_m)):
        raise ValueError('Nonfinite expert prediction or gate feature')
    if not np.isfinite(lookup_value[lookup]).all():
        raise ValueError('A lookup-applied row has no finite lookup value')
    if not all(a.shape==s.shape for a in (d,expert_s,expert_m,lookup,lookup_value)) or s.ndim!=1:
        raise ValueError('Gate input columns must be one-dimensional and equally sized')
    base=np.where(lookup,np.clip(lookup_value,0,10000),expert_m)
    delta=np.where(lookup,0.,expert_s-expert_m)
    result=(s,d,base,delta)
    if training:
        y=np.asarray(table['observed'],float)
        if y.shape!=s.shape or not len(y) or not np.isfinite(y).all():
            raise ValueError('Finite training responses are required')
        return (*result,y)
    return result


def _parameters(spec):
    family=spec['family']
    if family not in FAMILIES:raise ValueError('Unknown gate family')
    penalty=float(spec.get('penalty',0.))
    if not np.isfinite(penalty) or penalty<0:raise ValueError('Invalid regularization')
    active=[name for name in ('s','d') if name in family] if family!='constant' else []
    bounds=[(0.,.5) if name=='s' else (-.5,.5) for name in active]
    return family,penalty,active,bounds


def fit_gate(table,spec):
    """Fit on TRAINING OOF expert outputs, with one fixed constant anchor.

    The parent procedure must construct fold-safe expert OOF predictions and
    select spec without using its outer validation labels. lambda regularizes
    the actual weight deviation, not predictions or the response.
    """
    family,penalty,active,bounds=_parameters(spec)
    s,d,base,delta,y=_arrays(table,training=True)
    denominator=float(delta@delta)
    w0=float(np.clip(delta@(y-base)/denominator,0,1)) if denominator>1e-16 else 0.
    s0=float(np.mean(s));d0=float(np.mean(d))
    centered={'s':s-s0,'d':d-d0}
    design=np.column_stack([centered[name] for name in active]) if active else np.zeros((len(y),0))

    def objective(beta):
        linear=w0+design@beta;w=np.clip(linear,0,1)
        prediction=base+w*delta;error=(prediction-y)/D
        deviation=w-w0
        loss=float(np.mean(error**2)+penalty*np.mean(deviation**2))
        if not len(beta):return loss,np.empty(0)
        # The function is piecewise smooth. Boundary-flat starts are paired
        # with deterministic interior starts so w0=0/1 does not lock the gate.
        interior=(linear>0)&(linear<1)
        dw=((prediction-y)*delta/(D*D)+penalty*deviation)*interior
        gradient=2*design.T@dw/len(y)
        return loss,gradient

    beta=np.zeros(len(active));best_value=objective(beta)[0]
    starts_count=0;converged=0
    if active and denominator>1e-16:
        grids=[(0.,.25,.5) if name=='s' else (-.5,0.,.5) for name in active]
        starts=[np.zeros(len(active))]+[np.asarray(start,float) for start in product(*grids)]
        # Deduplicate starts without randomized initialization.
        starts=list({tuple(start):start for start in starts}.values())
        starts_count=len(starts)
        for start in starts:
            result=minimize(objective,start,jac=True,method='L-BFGS-B',bounds=bounds,
                            options={'maxiter':400,'ftol':1e-13,'gtol':1e-9,'maxls':40})
            converged+=int(result.success)
            for candidate in (start,result.x):
                if not np.isfinite(candidate).all():continue
                value=objective(candidate)[0]
                # Prefer the smaller correction for practically exact ties.
                if value<best_value-1e-13 or (abs(value-best_value)<=1e-13 and np.dot(candidate,candidate)<np.dot(beta,beta)):
                    beta=np.asarray(candidate,float);best_value=value
    coefficients={'s':0.,'d':0.}
    coefficients.update(zip(active,map(float,beta)))
    model=dict(format_version=1,spec=dict(spec),family=family,penalty=penalty,
               w0=w0,s0=s0,d0=d0,beta_s=coefficients['s'],beta_d=coefficients['d'],
               disagreement_scale=D,alpha_source=0.,training_rows=len(y),
               optimizer='deterministic multistart bounded L-BFGS-B',optimization_starts=starts_count,
               optimizer_converged_starts=converged,penalized_objective=float(best_value))
    prediction,weights=predict_gate(table,model)
    model['training_MSE']=float(np.mean((prediction-y)**2))
    model['mean_squared_weight_deviation']=float(np.mean((weights-w0)**2))
    return model


def predict_gate(table,params):
    """Return final SI and weights; observed labels are deliberately not read."""
    if float(params.get('alpha_source',np.nan))!=0.:
        raise ValueError('PR8 result requires alpha_source=0')
    if float(params.get('disagreement_scale',D))!=D:
        raise ValueError('Unexpected disagreement scale')
    s,d,base,delta=_arrays(table)
    coefficients=np.asarray([params[name] for name in ('w0','s0','d0','beta_s','beta_d')],float)
    if not np.isfinite(coefficients).all():raise ValueError('Nonfinite gate parameters')
    w0,s0,d0,beta_s,beta_d=coefficients
    if not 0<=w0<=1:raise ValueError('Invalid constant weight')
    weights=np.clip(w0+beta_s*(s-s0)+beta_d*(d-d0),0,1)
    prediction=base+weights*delta
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite mixed prediction')
    return prediction,weights
