"""Rebuild ignored PR3 comparison state and frozen raw-MLM feature cache."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import audit_regression as A
from extract_chemberta import encode_smiles, MODELS

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='work/pr3_half_mse')
    parser.add_argument('--encoder-cache',default='work/chemberta_snapshots')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/args.out;out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):state=A.make_state(root,out)
    smiles=list(state['train'].SMILES)+list(state['test'].SMILES)
    features=encode_smiles(smiles,'mlm','raw',root/args.encoder_cache,batch_size=1,threads=3)
    np.save(out/'mlm_raw.npy',features)
    manifest=dict(model=MODELS['mlm'],view='raw',batch_size=1,threads=3,shape=list(features.shape),
                  input_sha256=hashlib.sha256('\n'.join(smiles).encode()).hexdigest())
    (out/'embedding_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))

if __name__=='__main__':main()
