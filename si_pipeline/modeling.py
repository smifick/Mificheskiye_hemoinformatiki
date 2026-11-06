"""Regularized estimators and explicit physical/encoded label semantics."""
import warnings
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def estimator(family, classifier=False, seed=42, threads=2):
    if family=='ridge':
        return LogisticRegression(C=.1,max_iter=2000,random_state=seed) if classifier else Ridge(alpha=100.)
    if family=='extra':
        params=dict(n_estimators=250,max_depth=10,min_samples_leaf=5,max_features=.5,n_jobs=threads,random_state=seed)
        return ExtraTreesClassifier(**params) if classifier else ExtraTreesRegressor(**params)
    if family=='lightgbm':
        from lightgbm import LGBMClassifier,LGBMRegressor
        params=dict(n_estimators=300,learning_rate=.03,num_leaves=7,max_depth=3,
                    min_child_samples=15,min_split_gain=.01,reg_alpha=1.,reg_lambda=20.,
                    colsample_bytree=.7,subsample=.8,subsample_freq=1,
                    n_jobs=threads,random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True)
        return LGBMClassifier(**params) if classifier else LGBMRegressor(**params)
    if family=='catboost':
        from catboost import CatBoostClassifier,CatBoostRegressor
        params=dict(iterations=300,depth=3,learning_rate=.03,l2_leaf_reg=20.,random_strength=1.,
                    thread_count=threads,random_seed=seed,verbose=False,allow_writing_files=False)
        return CatBoostClassifier(loss_function='Logloss',**params) if classifier else CatBoostRegressor(loss_function='RMSE',**params)
    raise ValueError('Unknown family: '+family)


def pipeline(family, classifier=False, seed=42, threads=2):
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
    if family=='ridge':steps.append(('scale',StandardScaler()))
    steps.append(('estimator',estimator(family,classifier,seed,threads)))
    return Pipeline(steps)


def fit_model(spec,X,y,org_rows,external_X,external_y,external_keep,mode='encoded',sentinel=10000.,seed=42,threads=2):
    family=spec['family']
    encoded=mode=='encoded'
    c=(y[org_rows]==sentinel).astype(int)
    classifier=None
    if encoded:
        if np.unique(c).size==1:
            classifier=DummyClassifier(strategy='constant',constant=int(c[0])).fit(X[org_rows],c)
        else:
            classifier=pipeline(family,True,seed,threads)
            classifier.fit(X[org_rows],c)
        measured=org_rows[(y[org_rows]!=sentinel)&(y[org_rows]>0)]
    else:
        measured=org_rows
    keep=np.flatnonzero(external_keep) if spec['external_weight']>0 else np.array([],int)
    fit_X=np.vstack([X[measured],external_X[keep]])
    fit_y=np.r_[y[measured],external_y[keep]]
    weights=np.r_[np.ones(len(measured)),np.full(len(keep),spec['external_weight'])]
    if len(fit_y):
        scale=max(float(np.std(fit_y)),1.)
        regressor=pipeline(family,False,seed,threads)
        regressor.fit(fit_X,fit_y/scale,estimator__sample_weight=weights)
    else:
        # All training labels may be technical codes; this branch receives zero
        # mixture weight for the all-inactive case.
        scale=1.
        regressor=DummyRegressor(strategy='constant',constant=0.).fit(np.zeros((1,X.shape[1])),[0.])
    return {'classifier':classifier,'regressor':regressor,'target_scale':scale,'mode':mode,
            'sentinel':sentinel,'zero_rate':float(np.mean(y[org_rows]==0)) if encoded else 0.,'spec':spec}


def predict_model(model,X):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='X does not have valid feature names',category=UserWarning)
        physical=np.maximum(model['regressor'].predict(X)*model['target_scale'],0.)
        if model['mode']!='encoded':return physical
        clf=model['classifier']
        probs=clf.predict_proba(X)
    classes=clf.classes_
    where=np.flatnonzero(classes==1)
    p=probs[:,where[0]] if len(where) else np.zeros(len(X))
    # The sentinel is a benchmark label, not a physical SI or upper bound.
    return model['sentinel']*p+np.clip(1-p-model['zero_rate'],0,1)*physical
