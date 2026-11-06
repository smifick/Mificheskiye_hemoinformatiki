"""Independent validation of training, submission and externally collected endpoints."""
import argparse,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--sample',required=True);p.add_argument('--external',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    run=Path(a.run);ext=Path(a.external)
    d=pd.read_csv(run/'training_dataset.csv');s=pd.read_csv(run/'submission.csv');sample=pd.read_csv(a.sample);split=pd.read_csv(run/'validation_splits.csv')
    assert len(d)==len(split)
    assert all(Chem.MolFromSmiles(x) is not None for x in d.SMILES)
    assert list(s)==['index','SI VACV'] and s['index'].tolist()==sample['index'].tolist()
    assert np.isfinite(s['SI VACV']).all() and s['SI VACV'].ge(0).all()
    tr=split[split.outer_split=='train'];te=split[split.outer_split=='holdout']
    assert set(tr.scaffold_group).isdisjoint(te.scaffold_group)
    assert set(tr.canonical_smiles).isdisjoint(te.canonical_smiles)
    for fold in tr.inner_fold.unique():
        assert set(tr[tr.inner_fold==fold].scaffold_group).isdisjoint(tr[tr.inner_fold!=fold].scaffold_group)
    ep=pd.read_csv(ext/'external_endpoints.csv'); valid=ep[ep.validation_status=='valid_endpoint']
    assert valid.relation.eq('=').all()
    assert valid.SMILES.map(lambda x:Chem.MolFromSmiles(x) is not None).all()
    assert ep.source_url.str.startswith('https://www.ebi.ac.uk/chembl/api/data/activity/').all()
    doi=ep.doi.dropna().astype(str);bad_doi=doi[~doi.str.match(r'^10\.\d{4,9}/\S+$')]
    pairs=pd.read_csv(ext/'candidate_si_pairs.csv')
    if len(pairs):
        denom=pairs.IC50_uM.where(pairs.denominator_endpoint.eq('IC50'),pairs.EC50_uM)
        assert np.allclose(pairs.CC50_uM/denom,pairs.SI)
        assert pairs.validation_status.isin(['blocked_context_not_verified','metadata_matched_manual_review_required']).all()
    result={'status':'PASS','training_rows':len(d),'submission_rows':len(s),'outer_group_overlap':0,'inner_group_overlap':0,
            'submission_order_matches':True,'external_endpoint_rows':len(ep),'external_valid_endpoints':len(valid),
            'doi_format_warnings':bad_doi.tolist(),'candidate_pair_ratios_checked':len(pairs),
            'external_rows_approved_for_training':0,
            'scope':'Structural/schema/numeric validation; not independent verification of the published experiments.'}
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
