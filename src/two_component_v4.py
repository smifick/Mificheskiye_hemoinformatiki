"""Source-held-out external evaluation and grouped ablation of a two-component SI model.
Competition inputs: SMILES and response only. External SI never trains the code classifier.
"""
import argparse,json,hashlib,platform
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import nnls
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from train_model import standardize,featurize,tanimoto,metrics,TARGET
from train_external_v3 import canonical

HOLDOUT_DOI='10.1039/d0md00108b'
TRAIN_DOI='10.1016/j.ejmech.2023.115601'


def group_folds(indices,groups,n=3):
 return list(GroupKFold(n_splits=min(n,len(set(groups[indices])))).split(indices,groups=groups[indices]))


def fit_A(X,y,groups,tr):
 z=(y==10000).astype(int);oof=np.zeros(len(tr))
 for fit,val in group_folds(tr,groups):
  model=LogisticRegression(C=1.,max_iter=2000,random_state=42).fit(X[tr[fit]],z[tr[fit]])
  oof[val]=model.decision_function(X[tr[val]])
 calibrator=LogisticRegression(C=1.,max_iter=2000,random_state=42).fit(oof[:,None],z[tr])
 model=LogisticRegression(C=1.,max_iter=2000,random_state=42).fit(X[tr],z[tr])
 return {'classifier':model,'calibrator':calibrator}


def predict_A(bundle,X):
 return bundle['calibrator'].predict_proba(bundle['classifier'].decision_function(X)[:,None])[:,1]


def fit_B_core(K,targets,groups,idx,ext_idx,weight,log):
 # Each retained external observation receives the configured weight.
 ext_idx=np.asarray(ext_idx,dtype=int)
 fit=np.r_[idx,ext_idx]
 w=np.r_[np.ones(len(idx)),np.full(len(ext_idx),weight)]
 model=KernelRidge(alpha=1.,kernel='precomputed').fit(K[np.ix_(fit,fit)],np.log1p(targets[fit]) if log else targets[fit],sample_weight=w)
 return {'model':model,'fit_indices':fit,'log':log}


def predict_B_core(bundle,Krows):
 y=bundle['model'].predict(Krows[:,bundle['fit_indices']])
 return np.maximum(np.expm1(np.clip(y,-20,20)) if bundle['log'] else y,0)


def fit_B(K,targets,groups,known,ext_indices,weight,mode):
 known=np.asarray(known);ext_indices=np.asarray(ext_indices,dtype=int)
 if not weight:ext_indices=np.array([],dtype=int)
 correction=np.array([1.,0.]);calibration=[]
 if mode=='log_calibrated':
  oof=np.zeros(len(known))
  for fit,val in group_folds(known,groups):
   a,b=known[fit],known[val]
   keep=ext_indices[~np.isin(groups[ext_indices],groups[b])]
   assert set(groups[keep]).isdisjoint(groups[b])
   model=fit_B_core(K,targets,groups,a,keep,weight,True)
   oof[val]=predict_B_core(model,K[b])
  # Nonnegative affine correction is estimated only on grouped training OOF predictions.
  correction=nnls(np.column_stack([oof,np.ones(len(oof))]),targets[known])[0]
  calibration={'n_grouped_oof':len(known),'slope':float(correction[0]),'intercept':float(correction[1])}
 model=fit_B_core(K,targets,groups,known,ext_indices,weight,mode!='raw')
 model['correction']=correction;model['calibration']=calibration
 return model


def predict_B(bundle,Krows):
 return np.maximum(bundle['correction'][0]*predict_B_core(bundle,Krows)+bundle['correction'][1],0)


def combine(p,r):
 return 10000*p+(1-p)*r


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',default='.');parser.add_argument('--out',default='results/two_component_v4');args=parser.parse_args()
 root=Path(args.root);out=root/args.out;out.mkdir(parents=True,exist_ok=False)
 train=pd.read_csv(root/'data/raw/train.csv',usecols=['SMILES',TARGET]);test=pd.read_csv(root/'data/raw/test.csv',usecols=['index','SMILES']);sample=pd.read_csv(root/'data/raw/sample_submit.csv',usecols=['index'])
 ext=pd.read_csv(root/'data/external_curated/vacv_si.csv');ext=ext[ext.cell_line.eq('Vero')].copy()
 train_keys=np.array([canonical(s,True)[0] for s in train.SMILES]);test_keys=np.array([canonical(s,True)[0] for s in test.SMILES]);ext['key']=[canonical(s,True)[0] for s in ext.SMILES]
 # External test is a complete publication subset, after removal of all organizer overlaps.
 hold=ext[ext.doi.str.lower().eq(HOLDOUT_DOI) & ~ext.key.isin(set(train_keys)|set(test_keys))].copy()
 bank=ext[ext.doi.str.lower().eq(TRAIN_DOI) & ~ext.key.isin(set(test_keys)|set(hold.key))].copy()
 assert not set(hold.key)&set(train_keys);assert not set(hold.key)&set(bank.key);assert not set(bank.key)&set(test_keys)
 hold.to_csv(out/'external_holdout.csv',index=False);bank.to_csv(out/'external_training.csv',index=False)
 protocol={'holdout_doi':HOLDOUT_DOI,'training_doi':TRAIN_DOI,'holdout_rows':len(hold),'external_training_rows':len(bank),'holdout_organizer_structure_overlap':0,'holdout_external_training_structure_overlap':0,'cell_line':'Vero','split_selection':'DOI fixed in source before fitting or evaluation','retrospective_note':'These publications were inspected and used in earlier experiments; this is a source-disjoint retrospective check for the present run, not a globally untouched prospective test.','organizer_columns':['SMILES',TARGET],'test_columns':['index','SMILES'],'code_target':'response==10000, organizer rows only','zero_labels':'Excluded from physical SI fitting; retained in competition scoring. No claim that these are measured SI=0.','fingerprints':'Canonical tautomer; Morgan radius 2, 2048 bits, chirality','outer_folds':5,'outer_seed':2718,'inner_folds':3,'selection':'Organizer grouped OOF MSE only. External holdout metrics computed after selection and never used for tuning.','external_source_sha256':hashlib.sha256((root/'data/external_curated/vacv_si.csv').read_bytes()).hexdigest()}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 y=train[TARGET].to_numpy(float);targets=np.r_[y,bank.SI.to_numpy(float)];n=len(y)
 all_s=list(train_keys)+list(bank.key);X=featurize(all_s)[0];K=tanimoto(X,X)
 T=featurize(test_keys)[0];KT=tanimoto(T,X)
 H=featurize(list(hold.key))[0];KH=tanimoto(H,X)
 groups=np.array([standardize(s)[1] for s in list(train.SMILES)+list(bank.SMILES)])
 ids=np.arange(n);extids=np.arange(n,len(targets));folds=list(GroupKFold(n_splits=5,shuffle=True,random_state=2718).split(ids,y,groups[:n]))
 specs=[{'name':f'mixture_{mode}_external_{w}','mode':mode,'weight':w} for mode in ['raw','log_calibrated'] for w in [0.,.1,.3,1.]]
 specs +=[{'name':'mixture_log_uncalibrated_external_0.1','mode':'log_uncalibrated','weight':.1}]
 predictions={s['name']:np.zeros(n) for s in specs};baseline=np.zeros(n);fold_rows=[];cal_rows=[];physical_rows=[];split_rows=[]
 for fold,(tr,va) in enumerate(folds):
  assert set(groups[tr]).isdisjoint(groups[va])
  safe_ext=extids[~np.isin(groups[extids],groups[va])]
  known=tr[(y[tr]>0)&(y[tr]!=10000)]
  A=fit_A(X[:n],y,groups[:n],tr);p=predict_A(A,X[va])
  ref=KernelRidge(alpha=1.,kernel='precomputed').fit(K[np.ix_(tr,tr)],y[tr]);baseline[va]=np.maximum(ref.predict(K[np.ix_(va,tr)]),0)
  split_rows.extend({'source_row':int(i+2),'fold':fold,'group':groups[i]} for i in va)
  for s in specs:
   B=fit_B(K,targets,groups,known,safe_ext,s['weight'],s['mode']);r=predict_B(B,K[va]);pred=combine(p,r);predictions[s['name']][va]=pred
   fold_rows.append({'model':s['name'],'fold':fold,'external_rows':len(safe_ext) if s['weight'] else 0,**metrics(y[va],pred)})
   if B['calibration']:cal_rows.append({'model':s['name'],'fold':fold,**B['calibration']})
   valid=(y[va]>0)&(y[va]!=10000)
   physical_rows.extend({'model':s['name'],'source_row':int(i+2),'observed':float(y[i]),'prediction':float(r[j])} for j,i in enumerate(va) if valid[j])
  print('FOLD',fold,'train',len(tr),'valid',len(va),'safe_external',len(safe_ext),flush=True)
 rows=[{'model':'reference_tautomer_KRR',**metrics(y,baseline)}]+[{'model':s['name'],**metrics(y,predictions[s['name']])} for s in specs]
 table=pd.DataFrame(rows).sort_values('MSE');table.to_csv(out/'ablation.csv',index=False)
 # The best two-component configuration is selected before opening external holdout scores.
 selected=next(s for s in specs if s['name']==table[table.model!='reference_tautomer_KRR'].iloc[0].model)
 (out/'selection_frozen.json').write_text(json.dumps({'selected_spec':selected,'best_overall':table.iloc[0].to_dict(),'criterion':'Organizer scaffold OOF MSE only'},indent=2))
 A=fit_A(X[:n],y,groups[:n],ids);known=ids[(y>0)&(y!=10000)]
 external_scores=[];test_outputs={};physical_outputs={};final_b=None
 for s in [q for q in specs if q['name']==selected['name'] or q['weight']==0]:
  B=fit_B(K,targets,groups,known,extids,s['weight'],s['mode']);r=predict_B(B,KH)
  external_scores.append({'model':s['name'],'external_rows_fitted':len(extids) if s['weight'] else 0,**metrics(hold.SI.to_numpy(float),r)})
  physical_outputs[s['name']]=r
  if s['name']==selected['name']:final_b=B
 external=pd.DataFrame(external_scores);external.to_csv(out/'external_scores.csv',index=False)
 # External evaluation tests component B; encoding predictions are intentionally not compared with biological SI.
 pd.DataFrame({'compound_id':hold.compound_id,'SMILES':hold.SMILES,'observed_SI':hold.SI,**physical_outputs}).to_csv(out/'external_predictions.csv',index=False)
 p=predict_A(A,T);r=predict_B(final_b,KT);pred=combine(p,r)
 sub=sample.merge(pd.DataFrame({'index':test['index'],TARGET:pred}),on='index',validate='one_to_one',how='left',sort=False)
 assert len(sub)==len(sample) and sub['index'].tolist()==sample['index'].tolist() and np.isfinite(sub[TARGET]).all() and sub[TARGET].ge(0).all()
 sub.to_csv(out/'submission_v4.csv',index=False)
 pd.DataFrame({'index':test['index'],'p_technical_code':p,'physical_SI':r,'encoded_prediction':pred}).to_csv(out/'prediction_components.csv',index=False)
 pd.DataFrame({'source_row':ids+2,'observed':y,'baseline':baseline,**predictions}).to_csv(out/'oof_predictions.csv',index=False)
 pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv',index=False);pd.DataFrame(cal_rows).to_csv(out/'log_correction.csv',index=False);pd.DataFrame(physical_rows).to_csv(out/'physical_oof.csv',index=False);pd.DataFrame(split_rows).to_csv(out/'validation_splits.csv',index=False)
 # Export numeric parameters as transparent JSON, avoiding executable serialized objects.
 artifact={'classifier_coef':A['classifier'].coef_[0].tolist(),'classifier_intercept':float(A['classifier'].intercept_[0]),'calibrator_coef':float(A['calibrator'].coef_[0,0]),'calibrator_intercept':float(A['calibrator'].intercept_[0]),'regression_coefficients':final_b['model'].dual_coef_.tolist(),'regression_smiles':[all_s[i] for i in final_b['fit_indices']],'log':bool(final_b['log']),'correction':final_b['correction'].tolist(),'selected_spec':selected,'fingerprint_radius':2,'fingerprint_bits':2048,'canonical_tautomer':True}
 (out/'model_parameters.json').write_text(json.dumps(artifact))
 report={'selected_mixture':selected,'selected_oof':next(x for x in rows if x['model']==selected['name']),'reference_oof':rows[0],'best_overall_oof':table.iloc[0].to_dict(),'external_holdout_rows':len(hold),'external_training_rows':len(bank),'external_scores':external_scores,'public_v4_score':None,'classification_calibration':'Sigmoid calibration from grouped OOF logits within each training fold','log_correction':'Nonnegative affine correction from grouped physical-target OOF within training fold; no validation targets used','external_test_selection':'Source fixed before current fitting; external scores opened only after final selection','limitations':['Small physical-source holdout, retrospective data','Only two eligible external training rows remain after overlap/source filtering','OOF scores used for selection among configurations; no independent Kaggle improvement claim']}
 (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
