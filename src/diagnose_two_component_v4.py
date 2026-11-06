"""Fixed-protocol diagnostics; no hyperparameter selection or external-test tuning."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import brier_score_loss,roc_auc_score
from train_model import featurize,TARGET,metrics
from train_external_v3 import canonical
from two_component_v4 import fit_A,predict_A

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root);out=root/'results/two_component_v4'
 d=pd.read_csv(root/'data/raw/train.csv',usecols=['SMILES',TARGET]);splits=pd.read_csv(out/'validation_splits.csv').sort_values('source_row')
 X=featurize([canonical(s,True)[0] for s in d.SMILES])[0];y=d[TARGET].to_numpy(float);z=(y==10000).astype(int);g=splits.group.to_numpy();pcode=np.zeros(len(d));prior=np.zeros(len(d))
 for f in sorted(splits.fold.unique()):
  tr=np.where(splits.fold.to_numpy()!=f)[0];va=np.where(splits.fold.to_numpy()==f)[0]
  pcode[va]=predict_A(fit_A(X,y,g,tr),X[va]);prior[va]=z[tr].mean()
 classifier={'OOF_Brier':float(brier_score_loss(z,pcode)),'prior_Brier':float(brier_score_loss(z,prior)),'OOF_ROC_AUC':float(roc_auc_score(z,pcode))}
 pd.DataFrame({'source_row':np.arange(len(d))+2,'is_technical_code':z,'OOF_probability':pcode,'train_fold_prior':prior}).to_csv(out/'classifier_oof.csv',index=False)
 (out/'classifier_metrics.json').write_text(json.dumps(classifier,indent=2))
 ep=pd.read_csv(root/'data/external/external_endpoints.csv');cur=pd.read_csv(root/'data/external_curated/vacv_si.csv')
 matched=set(cur.antiviral_activity_id.astype(int))|set(cur.cytotoxicity_activity_id.astype(int))
 categories=np.select([ep.activity_id.isin(matched),ep.relation.isin(['>','<','>=','<=','≥','≤']),ep.endpoint.eq('Selectivity index'),ep.validation_status.eq('valid_endpoint')],['paired_to_publication_verified_SI','censored_endpoint','direct_SI_requires_review','valid_single_endpoint_without_verified_SI'],default='rejected_or_requires_review')
 ep[['activity_id','doi','endpoint','source_url']].assign(suitability=categories).to_csv(out/'external_suitability.csv',index=False)
 pd.Series(categories).value_counts().to_json(out/'suitability_summary.json',indent=2)
 hold=pd.read_csv(out/'external_predictions.csv');m=json.load(open(out/'metrics.json'));selected=m['selected_mixture']['name'];error=(hold.observed_SI-hold[selected])**2
 diagnostics={'holdout_rows':len(hold),'largest_error_fraction':float(error.max()/error.sum()),'largest_error_compound':str(hold.loc[error.idxmax(),'compound_id']),'max_holdout_SI':float(hold.observed_SI.max()),'full_holdout_MSE':float(error.mean()),'selection_used_holdout':False}
 (out/'external_error_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
 # Historical V2 fixed-candidate OOF predictions share the same organizer folds.
 old=pd.read_csv(root/'results/v2/oof_predictions.csv').sort_values('source_row');olds=pd.read_csv(root/'results/v2/validation_splits.csv').sort_values('source_row')
 assert np.array_equal(olds.outer_fold.to_numpy(),splits.fold.to_numpy())
 (out/'historical_v2_reference.json').write_text(json.dumps(metrics(old.observed.to_numpy(),old.selected_oof_prediction.to_numpy()),indent=2))
 print(json.dumps({'classifier':classifier,'external_diagnostics':diagnostics},indent=2))
if __name__=='__main__':main()
