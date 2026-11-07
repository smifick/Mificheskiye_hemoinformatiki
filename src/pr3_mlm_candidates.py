"""Prespecified alternatives using frozen raw-MLM features, training-only fits."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import research_heads as H


def candidate_specs():
    return [dict(name='mlm_logistic_c1_pair',family='logistic',C=1.),
            dict(name='mlm_rbf_c1_pair',family='rbf',C=1.),
            dict(name='mlm_rbf_c10_pair',family='rbf',C=10.),
            dict(name='mlm_mse_c1_pair',family='mse',C=1.),
            dict(name='mlm_mse_c10_pair',family='mse',C=10.)]


def _svc(c):
    return make_pipeline(StandardScaler(),SVC(C=c,kernel='rbf',gamma='scale'))


def fit_gate(spec,state,embeddings,tr):
    fit=tr[~state['flag'][tr]];y=state['y'];code=(y==10000).astype(int)
    flagged=tr[state['flag'][tr]]
    result=dict(spec=spec,p_flag=float((code[flagged].sum()+.5)/(len(flagged)+2)))
    style=dict(name='style',family='logistic',embedding='raw_style',C=1.)
    result['style']=H.fit_head(style,state,embeddings,tr)
    if len(np.unique(code[fit]))<2:
        result['constant']=float(code[fit].mean());return result
    x=embeddings['mlm_raw']
    if spec['family']=='rbf':
        decisions=np.zeros(len(fit))
        for a,b in GroupKFold(3,shuffle=True,random_state=314).split(fit,groups=state['groups'][fit]):
            if len(np.unique(code[fit[a]]))<2:decisions[b]=float(code[fit[a]].mean())
            else:
                est=_svc(spec['C']).fit(x[fit[a]],code[fit[a]])
                decisions[b]=est.decision_function(x[fit[b]])
        result['calibration']=LogisticRegression(C=1.,max_iter=2000).fit(decisions[:,None],code[fit])
        result['estimator']=_svc(spec['C']).fit(x[fit],code[fit]);return result
    scaler=StandardScaler().fit(x[fit]);z=scaler.transform(x[fit])
    est=LogisticRegression(C=spec['C'],max_iter=2000,random_state=42).fit(z,code[fit])
    result['scaler']=scaler
    if spec['family']=='logistic':result['estimator']=est;return result
    # Direct normalized raw-MSE objective; targets SI/10000 are unchanged and
    # unbounded. L2 acts on coefficients, never on the target or output values.
    design=np.column_stack([np.ones(len(fit)),z]);target=y[fit]/10000.
    initial=np.r_[est.intercept_,est.coef_.ravel()]
    def objective(beta):
        p=expit(design@beta);delta=p-target
        value=.5*np.dot(delta,delta)+.5*np.dot(beta[1:],beta[1:])/spec['C']
        gradient=design.T@(delta*p*(1-p));gradient[1:]+=beta[1:]/spec['C']
        return value,gradient
    opt=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12,'gtol':1e-7})
    if not opt.success:raise RuntimeError('MSE gate optimization failed: '+str(opt.message))
    result.update(beta=opt.x,optimization_iterations=int(opt.nit));return result


def predict_gate(model,state,embeddings,rows,tr):
    family=model['spec']['family'];x=embeddings['mlm_raw'][rows]
    if 'constant' in model:p=np.full(len(rows),model['constant'])
    elif family=='rbf':p=model['calibration'].predict_proba(model['estimator'].decision_function(x)[:,None])[:,1]
    else:
        x=model['scaler'].transform(x)
        if family=='logistic':p=model['estimator'].predict_proba(x)[:,1]
        else:p=expit(model['beta'][0]+x@model['beta'][1:])
    p=np.where(state['flag'][rows],model['p_flag'],p)
    return .5*(p+H.predict_head(model['style'],state,embeddings,rows,tr))
