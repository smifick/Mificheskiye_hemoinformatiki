"""Independently recompute metrics, verify nested partitions and replay submissions."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from benchmark_si_tail_evidence import metrics,strata
from si_tail_evidence import structures


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',default='results/si_tail_evidence')
    args=parser.parse_args()
    root=Path(args.results)
    train=pd.read_csv('data/raw/train.csv',usecols=['SMILES','SI VACV'])
    y=train['SI VACV'].to_numpy(float)
    features,keys,groups=structures(train.SMILES)
    checked=0
    for mode in ['benchmark','extrapolation']:
        path=root/mode
        protocol=json.loads((path/'protocol.json').read_text())
        for name,digest in protocol['source_sha256'].items():
            if hashlib.sha256((Path('src')/name).read_bytes()).hexdigest()!=digest:
                raise AssertionError(f'Executed source changed: {name}')
        predictions=pd.read_csv(path/'oof_predictions.csv')
        saved=pd.read_csv(path/'metrics_by_stratum.csv').set_index(['seed','model','stratum'])
        splits=pd.read_csv(path/'validation_splits.csv')
        inner=pd.read_csv(path/'inner_selection_splits.csv')
        calibration=pd.read_csv(path/'residual_calibration_splits.csv')
        for (seed,name),frame in predictions.groupby(['seed','model']):
            frame=frame.sort_values('row_index')
            assert frame.row_index.tolist()==list(range(len(y)))
            np.testing.assert_array_equal(frame.observed,y)
            assert np.isfinite(frame.prediction).all()
            for stratum,mask in strata(y,frame.label_kind.to_numpy(),frame.tail_threshold_from_training.to_numpy()).items():
                recomputed=metrics(y,frame.prediction.to_numpy(),mask)
                row=saved.loc[seed,name,stratum]
                for metric in ['MSE','RMSE','MAE','bias','SSE']:
                    if recomputed[metric] is None:
                        assert pd.isna(row[metric])
                    else:
                        assert np.isclose(recomputed[metric],row[metric],rtol=1e-12,atol=1e-8)
                assert recomputed['n']==row['n']
                checked+=1
        for (seed,fold),part in splits.groupby(['seed','fold']):
            fit=part[part.role.eq('training')].row_index.to_numpy(int)
            valid=part[part.role.eq('validation')].row_index.to_numpy(int)
            assert set(groups[fit]).isdisjoint(groups[valid])
            assert set(np.asarray(keys)[fit]).isdisjoint(np.asarray(keys)[valid])
            ins=inner[(inner.seed==seed)&(inner.outer_fold==fold)]
            assert set(ins.row_index)==set(fit)
            assert ins.groupby('scaffold').inner_fold.nunique().max()==1
            for phase,cal in calibration[(calibration.seed==seed)&(calibration.outer_fold==fold)].groupby('inner_selection_fold'):
                relevant=fit if phase==-1 else fit[~np.isin(fit,ins[ins.inner_fold.eq(phase)].row_index)]
                expected=relevant[(y[relevant]>0)&(y[relevant]!=10000)]
                assert set(cal.row_index)==set(expected)
                assert cal.groupby('scaffold').calibration_fold.nunique().max()==1
                assert set(cal.row_index).isdisjoint(valid)
    bank=pd.read_csv(root/'external_audit/research_training_bank.csv')
    used=pd.read_csv(root/'external_ablation/external_training_rows.csv')
    primary=pd.read_csv(root/'benchmark/validation_splits.csv')
    assert not bank.organizer_test_overlap.any()
    for (name,fold),part in used.groupby(['model','fold']):
        va=primary[(primary.seed==2718)&(primary.fold==fold)&primary.role.eq('validation')].row_index.to_numpy(int)
        assert set(part.scaffold).isdisjoint(groups[va])
        assert set(part.canonical_smiles).isdisjoint(np.asarray(keys)[va])
    new=pd.read_csv(root/'external_audit/new_publication_exact.csv')
    assert len(new)==14 and new.CC50_relation.eq('=').all() and new.IC50_relation.eq('=').all()
    np.testing.assert_allclose(new.SI,new.CC50_uM/new.IC50_uM,rtol=1e-12)
    test=pd.read_csv('data/raw/test.csv',usecols=['index','SMILES'])
    sample=pd.read_csv('data/raw/sample_submit.csv',usecols=['index'])
    x_test,_,_=structures(test.SMILES)
    submission_checks=[]
    with threadpool_limits(limits=2):
        for directory in ['benchmark','final']:
            bundle=joblib.load(root/directory/'model.joblib')
            prediction=bundle['model'].predict(x_test)
            csv=pd.read_csv(root/directory/'submission.csv')
            assert list(csv.columns)==['index','SI VACV']
            assert len(csv)==len(sample)==597 and csv['index'].is_unique
            assert csv['index'].tolist()==sample['index'].tolist()
            assert np.isfinite(csv['SI VACV']).all() and csv['SI VACV'].ge(0).all()
            np.testing.assert_allclose(prediction,csv['SI VACV'],rtol=1e-12,atol=1e-9)
            submission_checks.append({'artifact':directory,'rows':len(csv),'valid':True,
                                      'max_abs_replay_difference':float(np.max(np.abs(prediction-csv['SI VACV']))),
                                      'model_sha256':hashlib.sha256((root/directory/'model.joblib').read_bytes()).hexdigest(),
                                      'submission_sha256':hashlib.sha256((root/directory/'submission.csv').read_bytes()).hexdigest()})
    snapshot=pd.read_csv(root/'final/training_data.csv')
    np.testing.assert_array_equal(snapshot.SI,y)
    assert snapshot.SMILES.tolist()==train.SMILES.tolist()
    report={'status':'passed','metric_rows_independently_recomputed':checked,
            'outer_scaffold_overlap':0,'outer_canonical_structure_overlap':0,
            'inner_selection_scaffold_overlap':0,'inner_residual_calibration_scaffold_overlap':0,
            'outer_validation_rows_in_calibration':0,'external_validation_scaffold_overlap':0,
            'external_test_identity_overlap':0,'training_rows':320,'new_exact_publication_rows':14,
            'source_hashes_match':True,'submissions':submission_checks}
    (root/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
