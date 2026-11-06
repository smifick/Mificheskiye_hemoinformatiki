"""Nested grouped evaluation of SMILES-only kernels and missing-code mixture models."""
import argparse,json,hashlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import MACCSkeys,rdFingerprintGenerator
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.preprocessing import KernelCenterer
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier,CatBoostRegressor
from train_model import standardize,featurize,metrics,TARGET


def extended_features(smiles):
    m2,desc=featurize(smiles)
    m3=np.zeros_like(m2);count=np.zeros_like(m2);top=np.zeros_like(m2);maccs=np.zeros((len(smiles),167))
    g3=rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048,includeChirality=True)
    gc=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    for i,s in enumerate(smiles):
        m=Chem.MolFromSmiles(s)
        for fp,row in [(g3.GetFingerprint(m),m3[i]),(gc.GetCountFingerprint(m),count[i]),(Chem.RDKFingerprint(m,fpSize=2048),top[i]),(MACCSkeys.GenMACCSKeys(m),maccs[i])]:DataStructs.ConvertToNumpyArray(fp,row)
    return {'m2':m2,'m3':m3,'count':count,'top':top,'maccs':maccs,'tabular':np.column_stack([desc,maccs,m2])}


def kernel(A,B):
    dot=A@B.T
    return dot/np.maximum((A*A).sum(1)[:,None]+(B*B).sum(1)[None,:]-dot,1e-10)


def kernels(features,reference=None):
    ref=features if reference is None else reference
    result={k:kernel(features[k],ref[k]) for k in ['m2','m3','count','top','maccs']}
    result['multi']=(result['m2']+result['m3']+result['top'])/3
    return result


def candidates():
    result=[dict(name=f'krr_{key}_{alpha}',family='krr',key=key,alpha=alpha) for key,alphas in [('m2',[.03,.1,.3,1.]),('count',[.1,1.]),('multi',[.03,.1,.3,1.])] for alpha in alphas]
    result += [dict(name=f'catboost_raw_d{depth}',family='catboost',depth=depth) for depth in [4,6]]
    for classifier in ['krr','extra','catboost']:
        for regressor in ['krr_raw','extra_log']:
            result.append(dict(name=f'mixture_{classifier}_{regressor}',family='mixture',classifier=classifier,regressor=regressor))
    for classifier in ['krr','extra','catboost']:
        result.append(dict(name=f'blend_m2_mixture_{classifier}',family='blend',classifier=classifier))
    return result


def boost(klass=False,depth=4):
    common=dict(iterations=350,depth=depth,learning_rate=.04,l2_leaf_reg=8,random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
    return CatBoostClassifier(loss_function='MultiClass',**common) if klass else CatBoostRegressor(loss_function='RMSE',**common)


def stage_fit(spec,train_idx,y,codes,F,K):
    family=spec['family']
    if family=='centered':
        center=KernelCenterer().fit(K[spec['key']][np.ix_(train_idx,train_idx)])
        mean=float(y[train_idx].mean())
        est=KernelRidge(alpha=spec['alpha'],kernel='precomputed').fit(center.transform(K[spec['key']][np.ix_(train_idx,train_idx)]),y[train_idx]-mean)
        return dict(spec=spec,model=est,centerer=center,mean=mean,training_indices=train_idx)
    if family=='svm_mix':
        known=train_idx[codes[train_idx]==0]
        classifier=SVC(C=spec['C'],kernel='precomputed',probability=True,random_state=42).fit(K[spec['key']][np.ix_(train_idx,train_idx)],codes[train_idx]==1)
        reg=KernelRidge(alpha=1.,kernel='precomputed').fit(K['multi'][np.ix_(known,known)],np.log1p(y[known]) if spec['log'] else y[known])
        return dict(spec=spec,classifier=classifier,regressor=reg,training_indices=train_idx,known_indices=known)
    if family=='knn':return dict(spec=spec,training_indices=train_idx,training_targets=y[train_idx])
    if family=='krr':
        est=KernelRidge(alpha=spec['alpha'],kernel='precomputed').fit(K[spec['key']][np.ix_(train_idx,train_idx)],y[train_idx])
        return dict(spec=spec,model=est,training_indices=train_idx)
    A=F['tabular']
    if family=='catboost':
        est=boost(depth=spec['depth']).fit(A[train_idx],y[train_idx]/10000.)
        return dict(spec=spec,model=est)
    if family=='blend':
        base=stage_fit(dict(name='base',family='krr',key='m2',alpha=1.),train_idx,y,codes,F,K)
        other=stage_fit(dict(name='mix',family='mixture',classifier=spec['classifier'],regressor='extra_log'),train_idx,y,codes,F,K)
        return dict(spec=spec,base=base,other=other)
    # codes 0: known concentrations; 1: missing IC50; 2: missing CC50.
    known=train_idx[codes[train_idx]==0]
    if spec['classifier']=='krr':
        clf=KernelRidge(alpha=.3,kernel='precomputed').fit(K['multi'][np.ix_(train_idx,train_idx)],np.eye(3)[codes[train_idx]])
    elif spec['classifier']=='extra':
        clf=ExtraTreesClassifier(n_estimators=250,min_samples_leaf=2,max_features=.7,n_jobs=2,random_state=42).fit(np.nan_to_num(A[train_idx],nan=0,posinf=0,neginf=0),codes[train_idx])
    else:clf=boost(klass=True).fit(A[train_idx],codes[train_idx])
    if spec['regressor']=='krr_raw':
        reg=KernelRidge(alpha=1.,kernel='precomputed').fit(K['multi'][np.ix_(known,known)],y[known])
    else:
        reg=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,max_features=.7,n_jobs=2,random_state=42).fit(np.nan_to_num(A[known],nan=0,posinf=0,neginf=0),np.log1p(y[known]))
    return dict(spec=spec,classifier=clf,regressor=reg,known_indices=known,training_indices=train_idx)


def stage_predict(bundle,idx,F,K):
    spec=bundle['spec'];family=spec['family']
    if family=='centered':return np.maximum(bundle['model'].predict(bundle['centerer'].transform(K[spec['key']][np.ix_(idx,bundle['training_indices'])]))+bundle['mean'],0)
    if family=='svm_mix':
        probs=bundle['classifier'].predict_proba(K[spec['key']][np.ix_(idx,bundle['training_indices'])])[:,1]
        mu=bundle['regressor'].predict(K['multi'][np.ix_(idx,bundle['known_indices'])])
        mu=np.maximum(np.expm1(np.clip(mu,-20,20)) if spec['log'] else mu,0)
        return 10000*probs+(1-probs)*mu
    if family=='knn':
        S=K[spec['key']][np.ix_(idx,bundle['training_indices'])]
        near=np.argsort(-S,axis=1)[:,:spec['neighbors']]
        weights=np.take_along_axis(S,near,axis=1)**spec['power']
        denom=weights.sum(1)
        return np.divide((weights*bundle['training_targets'][near]).sum(1),denom,out=np.full(len(idx),bundle['training_targets'].mean()),where=denom>0)
    if family=='krr':return np.maximum(bundle['model'].predict(K[spec['key']][np.ix_(idx,bundle['training_indices'])]),0)
    if family=='catboost':return np.maximum(bundle['model'].predict(F['tabular'][idx])*10000.,0)
    if family=='blend':return .5*stage_predict(bundle['base'],idx,F,K)+.5*stage_predict(bundle['other'],idx,F,K)
    A=F['tabular'];tr=bundle['training_indices'];known=bundle['known_indices']
    if spec['classifier']=='krr':
        prob=np.maximum(bundle['classifier'].predict(K['multi'][np.ix_(idx,tr)]),0)
        total=prob.sum(1,keepdims=True);prob=np.divide(prob,total,out=np.tile([1.,0.,0.],(len(idx),1)),where=total>1e-12)
    else:
        values=A[idx] if spec['classifier']=='catboost' else np.nan_to_num(A[idx],nan=0,posinf=0,neginf=0)
        raw=bundle['classifier'].predict_proba(values);prob=np.zeros((len(idx),3))
        for col,c in enumerate(bundle['classifier'].classes_):prob[:,int(c)]=raw[:,col]
    if spec['regressor']=='krr_raw':
        physical=np.maximum(bundle['regressor'].predict(K['multi'][np.ix_(idx,known)]),0)
    else:physical=np.maximum(np.expm1(np.clip(bundle['regressor'].predict(np.nan_to_num(A[idx],nan=0,posinf=0,neginf=0)),-20,20)),0)
    return prob[:,1]*10000.+prob[:,0]*physical


def cross_validate(spec,indices,folds,y,codes,F,K):
    pred=np.zeros(len(indices))
    for fit,valid in folds:
        b=stage_fit(spec,indices[fit],y,codes,F,K)
        pred[valid]=stage_predict(b,indices[valid],F,K)
    return pred


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--out',default='results/v2');a=p.parse_args();root=Path(a.root);out=root/a.out;out.mkdir(parents=True,exist_ok=False)
    train=pd.read_csv(root/'data/raw/train.csv');test=pd.read_csv(root/'data/raw/test.csv');sample=pd.read_csv(root/'data/raw/sample_submit.csv')
    trstd=[standardize(s) for s in train.SMILES];teststd=[standardize(s) for s in test.SMILES]
    y=train[TARGET].to_numpy(float);codes=np.where(train.IC50_uM_VACV.isna(),1,np.where(train.CC50_uM.isna(),2,0))
    assert np.all(y[codes==1]==10000) and np.all(y[codes==2]==0)
    groups=np.array([x[1] for x in trstd]);smiles=[x[0] for x in trstd]
    F=extended_features(smiles);Ft=extended_features([x[0] for x in teststd]);K=kernels(F);Kt=kernels(Ft,F)
    specs=candidates();base=next(s for s in specs if s['name']=='krr_m2_1.0')
    ids=np.arange(len(y));outer=list(GroupKFold(n_splits=5,shuffle=True,random_state=2718).split(ids,y,groups))
    nested=np.zeros(len(y));baseline=np.zeros(len(y));foldrows=[];searchrows=[];splitrows=[]
    for fold,(fit,valid) in enumerate(outer):
        assert set(groups[fit]).isdisjoint(groups[valid])
        inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=314).split(fit,y[fit],groups[fit]));ranking=[]
        for spec in specs:
            pred=cross_validate(spec,fit,inner,y,codes,F,K);m=metrics(y[fit],pred)
            ranking.append((m['MSE'],spec));searchrows.append(dict(outer_fold=fold,model=spec['name'],**m))
        ranking.sort(key=lambda z:z[0]);chosen=ranking[0][1]
        b=stage_fit(chosen,fit,y,codes,F,K);nested[valid]=stage_predict(b,valid,F,K)
        b0=stage_fit(base,fit,y,codes,F,K);baseline[valid]=stage_predict(b0,valid,F,K)
        foldrows.append(dict(fold=fold,selected=chosen['name'],n_train=len(fit),n_valid=len(valid),selected_MSE=metrics(y[valid],nested[valid])['MSE'],baseline_MSE=metrics(y[valid],baseline[valid])['MSE']))
        splitrows.extend(dict(source_row=int(i+2),outer_fold=fold,scaffold_group=groups[i]) for i in valid)
        print('NESTED',fold,foldrows[-1],flush=True)
    # Candidate choice for the final refit uses grouped OOF MSE, never Kaggle labels.
    ranking=[];allpred={}
    for spec in specs:
        pred=cross_validate(spec,ids,outer,y,codes,F,K);allpred[spec['name']]=pred
        ranking.append(dict(model=spec['name'],**metrics(y,pred)))
        print('FULL_CV',spec['name'],metrics(y,pred)['MSE'],flush=True)
    table=pd.DataFrame(ranking).sort_values('MSE');table.to_csv(out/'candidate_cv.csv',index=False)
    selected_name=table.iloc[0].model;selected=next(s for s in specs if s['name']==selected_name)
    b=stage_fit(selected,ids,y,codes,F,K);pred=stage_predict(b,np.arange(len(test)),Ft,Kt)
    submission=sample[['index']].merge(pd.DataFrame({'index':test['index'],TARGET:pred}),on='index',how='left',validate='one_to_one',sort=False)
    assert len(submission)==len(sample) and submission['index'].tolist()==sample['index'].tolist() and np.isfinite(submission[TARGET]).all()
    submission.to_csv(out/'submission_v2.csv',index=False)
    # Structural novelty diagnostics are computed without test assay values.
    seen=set(groups);testgroups=[x[1] for x in teststd]
    report={'selected_model':selected_name,'nested_selection_pipeline':metrics(y,nested),'baseline_same_outer_folds':metrics(y,baseline),
            'selected_candidate_oof':table.iloc[0].to_dict(),'selection_bias_note':'Final candidate OOF scores used for selection; nested score estimates the selection procedure.',
            'outer_groups_overlap':0,'outer_folds':5,'inner_folds':3,'candidates':len(specs),'seed_outer':2718,'seed_inner':314,
            'test_rows':len(test),'test_scaffold_seen_fraction':float(np.mean([g in seen for g in testgroups])),
            'features':'SMILES only; concentration missingness used only as an auxiliary target in training.',
            'missing_codes':'10000 for missing IC50; two zero labels with missing CC50 handled as separate auxiliary class.',
            'external_used':False,'public_v2_score':None}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    import platform,catboost,rdkit,sklearn
    config={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'rdkit':rdkit.__version__,'sklearn':sklearn.__version__,'catboost':catboost.__version__,'selected_spec':selected,'candidate_specs':specs,'input_sha256':{name:hashlib.sha256((root/'data/raw'/name).read_bytes()).hexdigest() for name in ['train.csv','test.csv','sample_submit.csv']}}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    pd.DataFrame(foldrows).to_csv(out/'nested_fold_metrics.csv',index=False);pd.DataFrame(searchrows).to_csv(out/'inner_search.csv',index=False)
    pd.DataFrame(splitrows).to_csv(out/'validation_splits.csv',index=False)
    pd.DataFrame({'source_row':ids+2,'observed':y,'auxiliary_code':codes,'nested_prediction':nested,'baseline_prediction':baseline,'selected_oof_prediction':allpred[selected_name]}).to_csv(out/'oof_predictions.csv',index=False)
    joblib.dump({'bundle':b,'features':F,'training_smiles':smiles,'spec':selected},out/'model_v2.joblib',compress=3)
    print('FINAL',json.dumps(report),flush=True)

if __name__=='__main__':main()
