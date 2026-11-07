"""Derived diagnostics only: no fitting or selection from leaderboard results."""
import argparse,importlib.metadata,json,shutil
from pathlib import Path
import numpy as np,pandas as pd
import soft_source as S

def report(root,out,cache):
    impacts=[];probability_metrics=[]
    for scheme,file in [('production','production_components.csv.gz'),('primary','primary_outer_components.csv.gz')]:
        t=pd.read_csv(out/file);base=S.apply_alpha(t,0)
        for a in (*S.GRID,1.):
            p,q,leaves,_=S.apply_alpha(t,a,details=True)
            raw=10000*q+np.clip(1-q-t.zero_rate.to_numpy(),0,1)*t.physical_prediction.to_numpy()
            impacts.append(dict(scheme=scheme,alpha=a,**S.effect_counts(t,a),
                                upper_numeric_clip_rows=int((raw>10000).sum()),
                                zero_mass_clip_rows=int((1-q-t.zero_rate.to_numpy()<0).sum()),
                                max_abs_SI_change=float(np.abs(p-base).max())))
    for scheme in ['primary','confirmation5511','confirmation5512']:
        t=pd.read_csv(out/(scheme+'_outer_components.csv.gz'));p0=S.apply_alpha(t,0,details=True)[1]
        selected=np.zeros(len(t))
        for a in sorted(t.selected_alpha.unique()):
            mask=t.selected_alpha.to_numpy()==a
            selected[mask]=S.apply_alpha(t.loc[mask],float(a),details=True)[1]
        for group,mask in [('all',np.ones(len(t),bool)),('source_present',t.source_present.to_numpy(bool))]:
            y=(t.observed.to_numpy()==10000).astype(float)
            probability_metrics.append(dict(scheme=scheme,group=group,n=int(mask.sum()),
                baseline_Brier=float(np.mean((p0[mask]-y[mask])**2)),selected_Brier=float(np.mean((selected[mask]-y[mask])**2))))
    pd.DataFrame(impacts).to_csv(out/'impact_by_alpha.csv',index=False)
    pd.DataFrame(probability_metrics).to_csv(out/'probability_metrics.csv',index=False)
    packages=['numpy','scipy','pandas','scikit-learn','rdkit','joblib','threadpoolctl','catboost','lightgbm','torch',
              'transformers','tokenizers','huggingface-hub','safetensors','requests','pytest']
    (out/'environment.json').write_text(json.dumps({p:importlib.metadata.version(p) for p in packages},indent=2))
    shutil.copy2(root/'results/audit_20261107/embeddings_mlm.json',out/'original_encoder_manifest.json')
    shutil.copy2(cache/'embedding_manifest.json',out/'embedding_manifest.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',default='work/soft_source_cache');p.add_argument('--out',default='results/soft_source_20261107')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];report(root,root/a.out,root/a.cache)
