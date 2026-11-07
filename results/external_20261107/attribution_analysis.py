"""Recompute classifier-only attribution on the successful model's frozen folds.

All numeric measurements in this script are diagnostic. It does not modify
training labels, frozen models, or historical prediction files.
"""
from pathlib import Path
import sys,json,hashlib,platform,argparse
import numpy as np,pandas as pd,joblib
from rdkit import Chem
from rdkit.Chem import Descriptors,rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--cache',default='results/audit_20261107',help='Directory with unchanged frozen state_cache.joblib and embeddings_*.npz')
parser.add_argument('--out',default='results/external_20261107')
args=parser.parse_args()
sys.path.insert(0,str(root/'src'))
import research_heads as H
import train_v5 as T
import train_v7 as V7
old=root/'results/audit_20261107';cache=Path(args.cache)
if not cache.is_absolute():cache=root/cache
out=Path(args.out)
if not out.is_absolute():out=root/out
out.mkdir(parents=True,exist_ok=True)
state=joblib.load(cache/'state_cache.joblib');base=np.load(old/'baseline_oof.npz');emb=H.load_embeddings(cache)
y=state['y'];n=len(y);folds=base['fold'];ids=np.arange(n)
names=['mlm_raw_logistic_0.1','raw_smiles_notation_logistic','notation_mlm_probability_average']
specs=[s for s in H.candidate_specs() if s['name'] in names]
pred={'v7':base['v7'].copy()};probs={};frozen_B=base['r']
for s in specs:pred[s['name']]=np.zeros(n);probs[s['name']]=np.zeros(n)
for fold in np.unique(folds):
 tr=ids[folds!=fold];va=ids[folds==fold]
 for s in specs:
  head=H.fit_head(s,state,emb,tr)
  p=H.predict_head(head,state,emb,va,tr)
  probs[s['name']][va]=p
  pred[s['name']][va]=H.complete_predictions(state,tr,va,frozen_B[va],p)
# Check exact agreement with archived aggregate scores and paired row predictions.
dev=pd.read_csv(old/'targeted_development.csv');saved=pd.read_csv(old/'paired_new_model_errors.csv').sort_values('source_row')
metrics=[]
for name,p in pred.items():
 mse=float(np.mean((p-y)**2))
 archived=float(dev[(dev.scheme=='scaffold')&(dev.model==name)].MSE.iloc[0])
 assert abs(mse-archived)<1e-6,(name,mse,archived)
 metrics.append(dict(model=name,OOF_MSE=mse,archived_MSE=archived,max_aggregate_difference=abs(mse-archived),same_physical_branch=True,same_external_bank=True,same_structure_processing=True,same_target_transform=True))
chosen='notation_mlm_probability_average';p=pred[chosen]
assert np.max(abs(p-saved.new_prediction.to_numpy()))<1e-6
errors=saved[['source_row','SMILES','observed','label_type','fold','scaffold','max_train_tanimoto','nearest_train_source_rows','nearest_train_labels']].copy()
for name,prediction in pred.items():
 errors[name+'_prediction']=prediction
 errors[name+'_SSE']=(prediction-y)**2
 errors[name+'_SSE_reduction_vs_v7']=(pred['v7']-y)**2-(prediction-y)**2
errors['selected_signed_error']=p-y
errors['code_probability']=probs[chosen]
errors['physical_prediction']=frozen_B
errors['target_range']=pd.cut(y,[-np.inf,0,1,10,100,1000,10000,np.inf],labels=['0','(0,1]','(1,10]','(10,100]','(100,1000]','(1000,10000]','>10000']).astype(str)
errors.loc[y==10000,'target_range']='technical_10000'
errors['similarity_range']=pd.cut(errors.max_train_tanimoto,[-np.inf,.3,.5,.7,.9,np.inf],labels=['<=0.3','(0.3,0.5]','(0.5,0.7]','(0.7,0.9]','>0.9']).astype(str)
raw=pd.read_csv(root/'data/raw/train.csv')
exact=[];families=[]
for s in errors.SMILES:
 m=T.std(s)
 hetero=sum(a.GetAtomicNum() not in [1,6] for a in m.GetAtoms())
 rings=rdMolDescriptors.CalcNumRings(m);aromatic=rdMolDescriptors.CalcNumAromaticRings(m)
 atoms=m.GetNumHeavyAtoms();carbons=sum(a.GetAtomicNum()==6 for a in m.GetAtoms())
 signature=f'rings={rings};aromatic={aromatic};hetero={hetero};heavy_atoms_band={int(atoms//10)*10}s'
 families.append((signature,rings,aromatic,hetero,atoms,carbons/atoms))
 scf=MurckoScaffold.GetScaffoldForMol(m)
 exact.append(Chem.MolToSmiles(scf,isomericSmiles=True))
errors['exact_Murcko_scaffold']=exact
for j,k in enumerate(['chemotype_signature','n_rings','n_aromatic_rings','n_heteroatoms','n_heavy_atoms','carbon_fraction']):errors[k]=[a[j] for a in families]
errors.to_csv(out/'attribution_row_errors.csv',index=False)
pd.DataFrame(metrics).to_csv(out/'attribution_same_fold_metrics.csv',index=False)
for by in ['label_type','target_range','similarity_range','scaffold','exact_Murcko_scaffold','chemotype_signature']:
 agg=errors.groupby(by,dropna=False).agg(n=('source_row','size'),baseline_SSE=('v7_SSE','sum'),selected_SSE=(chosen+'_SSE','sum'),SSE_reduction=(chosen+'_SSE_reduction_vs_v7','sum'),mean_signed_error=('selected_signed_error','mean'),median_train_similarity=('max_train_tanimoto','median'),technical_codes=('label_type',lambda a:int((a=='technical_10000').sum()))).reset_index()
 agg['selected_MSE']=agg.selected_SSE/agg.n
 agg['selected_total_SSE_fraction']=agg.selected_SSE/errors[chosen+'_SSE'].sum()
 agg['gain_total_fraction']=agg.SSE_reduction/errors[chosen+'_SSE_reduction_vs_v7'].sum()
 agg.sort_values('selected_SSE',ascending=False).to_csv(out/f'attribution_by_{by}.csv',index=False)
# Concentration of improvement, plus wins/losses, not just high-impact anecdotes.
delta=errors[chosen+'_SSE_reduction_vs_v7'].to_numpy();total=float(delta.sum());se=errors[chosen+'_SSE'].to_numpy()
win_order=np.argsort(-delta)
summary={'public':{'successful_MSE':6327610.34997,'v7_MSE':9421022.05828,'v6_MSE':9649437.79854,'gain_vs_v7_pct':100*(1-6327610.34997/9421022.05828),'gain_vs_v6_pct':100*(1-6327610.34997/9649437.79854),'artifact_confirmation':'User explicitly confirmed submission(2).csv is the unmodified results/audit_20261107/submission.csv','successful_submission_sha256':hashlib.sha256((old/'submission.csv').read_bytes()).hexdigest(),'successful_model_sha256':hashlib.sha256((old/'selected_model.joblib').read_bytes()).hexdigest(),'commit':'b6c2e0c384d043582753993e442be7c042c3139a'},
 'primary_OOF':{'baseline_MSE':float(np.mean((pred['v7']-y)**2)),'selected_MSE':float(np.mean(se)),'relative_reduction_pct':100*(1-se.mean()/np.mean((pred['v7']-y)**2)),'SSE_reduction':total,'rows_improved':int((delta>1e-6).sum()),'rows_worsened':int((delta < -1e-6).sum()),'rows_equal_absolute_SSE_tolerance_1e_minus6':int((np.abs(delta)<=1e-6).sum()),'top_positive_gain_fractions':{str(k):float(delta[win_order[:k]].sum()/total) for k in [1,5,10,20]},'model_row_reproduction_max_error':float(np.max(abs(p-saved.new_prediction.to_numpy()))),'unchanged_source_flag_rows':int(np.sum(np.abs(delta[state['flag'][:n]])<=1e-6)),'source_flag_rows':int(state['flag'][:n].sum()),'rare_tail_rows':int((y>10000).sum()),'rare_tail_SSE_share':float(se[y>10000].sum()/se.sum()),'rare_tail_gain_share':float(delta[y>10000].sum()/total)},
 'attribution_limits':'Only classifier inputs/heads changed in this comparison. Physical branch, external bank, structure processing, official labels and target transform are held fixed. One Public score cannot identify the test-row errors or causally assign the Public improvement to a component.',
 'data_policy':'New user instruction prohibits external numerical labels for any competition-test structure. The successful historical baseline retained exact external lookup; it must be frozen as control, not silently reclassified as compliant with the new policy.',
 'python':platform.python_version()}
# Diagnostic audit of the successful historical full-fit model. These external
# numeric values are not inserted into new training or treated as Kaggle truth.
portable=joblib.load(old/'selected_model.joblib')['baseline_portable_model']
test=pd.read_csv(root/'data/raw/test.csv',usecols=['SMILES','index'])
keys=[V7.source_key(s) for s in test.SMILES]
covered=np.array([k in portable['lookup'] for k in keys])
lookup_rows=test[covered].copy()
lookup_rows['structure_key']=[k for k,c in zip(keys,covered) if c]
lookup_rows['legacy_exact_external_value']=[portable['lookup'][k] for k,c in zip(keys,covered) if c]
lookup_rows.to_csv(out/'attribution_legacy_test_lookup.csv',index=False)
w=state['world'];test_keys=set(w.tau[w.n_tr:w.n_tr+w.n_te])
external_overlap=np.array([w.tau[i] in test_keys for i in w.ext_ids])
v7_test=pd.read_csv(root/'submission_candidate_v7.csv')['SI VACV'].to_numpy()
selected_test=pd.read_csv(old/'submission.csv')['SI VACV'].to_numpy()
test_change=abs(selected_test-v7_test)
summary['legacy_test_external_label_overlap']={
 'external_measurement_rows':int(external_overlap.sum()),
 'by_tier':pd.Series(w.ext_tier[external_overlap]).value_counts().to_dict(),
 'tautomer_structures':len(set(w.tau[w.ext_ids[external_overlap]])),
 'test_rows_using_exact_lookup':int(covered.sum()),
 'test_predictions_unchanged_vs_v7':int((test_change<1e-6).sum()),
 'test_predictions_changed_vs_v7':int((test_change>=1e-6).sum()),
 'lookup_prediction_change':float(test_change[covered].max()),
 'note':'16 structures is only the reviewed Vero/all-cells subset, not the full V+C bank.'}
summary['input_sha256']={str(q.relative_to(root)):hashlib.sha256(q.read_bytes()).hexdigest()
 for q in [old/'baseline_oof.npz',old/'targeted_development.csv',old/'paired_new_model_errors.csv',root/'data/raw/train.csv',root/'data/raw/test.csv']}
(out/'attribution_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
