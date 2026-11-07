"""Integration checks for nested gating, reference banks and frozen PR8 rules."""
import argparse,copy,gzip,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import similarity_experts as E
import similarity_weight as G
import run_similarity_gate as R

def validate(root,cache,out,encoder_cache):
    frozen=json.loads((out/'baseline_frozen.json').read_text());before={p:E.sha(root/p) for p in frozen['file_sha256']}
    assert before==frozen['file_sha256'] and frozen['alpha_source']==0
    state,emb,smiles=E.prepare(cache);ids=np.arange(len(state['y']))
    provenance=json.loads(gzip.decompress((out/'model_provenance.json.gz').read_bytes()))
    for key,p in provenance.items():
        tr=np.array(p['training_source_rows'])-2;va=np.array(p['validation_source_rows'])-2;blocked=np.array(p['blocked_source_rows'],int)-2
        ref=np.array(p['reference_source_rows'])-2
        np.testing.assert_array_equal(ref,tr[~state['flag'][tr]])
        assert set(ref).isdisjoint(np.r_[va,blocked])
        assert set(state['groups'][ref]).isdisjoint(state['groups'][np.r_[va,blocked]])
        assert set(state['world'].tau[ref]).isdisjoint(state['world'].tau[np.r_[va,blocked]])
        assert p['reference_fingerprint_sha256']==E.array_sha(state['structural_bits'][ref])
        assert p['alpha_source']==0 and all(np.array_equal(np.array(x)-2,ref) for x in p['molecular_head_training_source_rows'])
    actual=pd.read_csv(out/'primary_outer_components.csv.gz');choices=json.loads((out/'fold_choices.json').read_text())
    primary_choices=[c for c in choices if c['scheme']=='primary']
    checked=max(primary_choices,key=lambda c:abs(c['selected_gate']['beta_s'])+abs(c['selected_gate']['beta_d'])+c['selected_gate']['w0'])
    f=checked['fold'];tr,va=R.folds(state,ids,'scaffold',2718,archived=True)[f]
    reference=actual[actual.outer_fold==f].sort_values('source_row');poison_errors={};specs={}
    with tempfile.TemporaryDirectory(dir=root/'work') as directory:
        directory=Path(directory)
        for name,value in [('NaN',np.nan),('large1e12',1e12)]:
            changed=state.copy();changed['world']=copy.copy(state['world'])
            changed['y']=state['y'].copy();changed['world'].y=state['world'].y.copy()
            changed['y'][va]=value;changed['world'].y[va]=value
            changed['train']=state['train'].copy();changed['world'].train=state['world'].train.copy()
            changed['train'].loc[va,'SI VACV']=value;changed['world'].train.loc[va,'SI VACV']=value
            fresh=directory/name;fresh.mkdir()
            outer,inner,calibration,decisions=R.nested(changed,emb,smiles,[(tr,va)],fresh,{},'poison_'+name)
            outer=outer.sort_values('source_row');chosen=decisions[0]['selected_gate'];specs[name]=decisions[0]['selected_spec']
            assert chosen==checked['selected_gate'] and specs[name]==checked['selected_spec']
            error=float(np.max(np.abs(outer.prediction_selected.to_numpy()-reference.prediction_selected.to_numpy())))
            assert error<1e-7;poison_errors[name]=error
            assert set(inner.source_row).isdisjoint(va+2) and set(calibration.source_row).isdisjoint(va+2)
        cli=check_cli(root,out,directory,encoder_cache,choices)
    finish_validation(root,out,production_before=before,provenance=provenance,checked=checked,
                      poison_errors=poison_errors,specs=specs,cli=cli)

def check_cli(root,out,directory,encoder_cache,choices):
    production=pd.read_csv(out/'production_components.csv.gz')
    # A nonzero already-trained outer gate exercises positive routing even if
    # the final candidate refits to zero. This is a temporary TEST config only.
    trained=max([c['family_gates']['sd'] for c in choices],key=lambda g:abs(g['beta_s'])+abs(g['beta_d'])+g['w0'])
    candidate_json=directory/'test_only_nonzero_gate.json'
    candidate_json.write_text(json.dumps(dict(recommended_gate=trained,statistical_acceptance_passed=True,
                                             validation_passed=True,alpha_source=0.,test_only=True)))
    differing=np.abs(production.pred_S-production.pred_M)>1e-9
    idx=np.r_[np.flatnonzero(differing)[:4],np.flatnonzero(production.source_present&~production.lookup_applied)[:2],
              np.flatnonzero(production.lookup_applied)[:2]]
    subset=production.iloc[idx];small=subset[['index','SMILES']]
    test=pd.read_csv(root/'data/raw/test.csv',usecols=['index','SMILES'])
    cases=[('fallback_clean',test,out/'gate.json',E.finish(production,production.pred_M.to_numpy()),production['index']),
           ('fallback_poison_reversed',test.assign(CC50_uM=1e99,IC50_uM_VACV=np.nan,**{'SI VACV':1e12},Title='ignored',Culture='ignored',strain='ignored').iloc[::-1],
            out/'gate.json',E.finish(production,production.pred_M.to_numpy()),production['index'])]
    expected,_=G.predict_gate(subset,trained)
    cases += [('nonzero_clean',small,candidate_json,expected,subset['index']),
              ('nonzero_poison_reversed',small.assign(CC50_uM=1e99,IC50_uM_VACV=np.nan,**{'SI VACV':1e12},Title='ignored',Culture='ignored',strain='ignored').iloc[::-1],candidate_json,expected,subset['index'])]
    errors={}
    for name,frame,config,expected,index in cases:
        inp=directory/(name+'.csv');target=directory/(name+'_prediction.csv');frame.to_csv(inp,index=False)
        subprocess.run([sys.executable,str(root/'src/predict_similarity_gate.py'),'--model',str(root/'results/audit_20261107/selected_model.joblib'),
                        '--gate-json',str(config),'--input',str(inp),'--out',str(target),'--encoder-cache',str(encoder_cache)],check=True)
        actual=pd.read_csv(target);assert actual['index'].tolist()==frame['index'].tolist()
        values=actual.set_index('index')['SI VACV'].loc[index].to_numpy()
        errors[name]=float(np.max(np.abs(values-expected)));assert errors[name]<1e-7
    return dict(max_error=errors,nonzero_test_gate=trained,nonzero_test_rows=len(subset),fallback_rows=len(test))

def finish_validation(root,out,production_before,provenance,checked,poison_errors,specs,cli):
    assert {p:E.sha(root/p) for p in production_before}==production_before
    config=json.loads((out/'gate.json').read_text());candidate=config['candidate_gate']
    restored=json.loads(json.dumps(candidate));production=pd.read_csv(out/'production_components.csv.gz')
    assert production['index'].tolist()==pd.read_csv(root/'data/raw/sample_submit.csv',usecols=['index'])['index'].tolist()
    p,w=G.predict_gate(production,candidate);p2,w2=G.predict_gate(production,restored)
    np.testing.assert_array_equal(p,p2);np.testing.assert_array_equal(w,w2)
    raw=E.finish(production,production.pred_M.to_numpy());source=production.source_present.to_numpy(bool);lookup=production.lookup_applied.to_numpy(bool)
    np.testing.assert_allclose(p[source],raw[source],atol=1e-9,rtol=0);np.testing.assert_array_equal(p[lookup],raw[lookup])
    assert config['alpha_source']==0 and candidate['alpha_source']==0
    checks=dict(status='PASS',reference_models_checked=len(provenance),reference_banks_exact_and_no_validation_overlap=True,
        outer_poison_refitted=True,checked_outer_fold=checked['fold'],checked_spec=checked['selected_spec'],outer_poison_selected_specs=specs,
        outer_poison_prediction_max_error=poison_errors,CLI=cli,production_hashes_unchanged=True,alpha_source=0.,
        JSON_roundtrip_exact=True,lookup_and_source_present_unchanged=True,IDs_preserved=True,submission_order_matches_sample=True)
    (out/'validation.json').write_text(json.dumps(checks,indent=2))
    config['validation_passed']=True
    if config['statistical_acceptance_passed']:
        config['recommended_gate']=config['candidate_gate'];config['recommendation']='validated dynamic gate'
        pending=out/'candidate_pending_validation.csv'
        if pending.exists():pending.replace(out/'submission_similarity_gate.csv')
    (out/'gate.json').write_text(json.dumps(config,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',default='work/similarity_gate_cache');p.add_argument('--out',default='results/similarity_gate_20261107')
    p.add_argument('--encoder-cache',default='work/chemberta_snapshots');args=p.parse_args();root=Path(__file__).resolve().parents[1]
    with threadpool_limits(limits=2):validate(root,root/args.cache,root/args.out,root/args.encoder_cache)
