"""Cold preparation of frozen PR3 features and the exact structural bank bits.

No expert head or gate is fitted. The raw-SMILES MLM representation keeps the
PR3 snapshot, pooling, batch size and CPU thread count. Structural fingerprints
use exactly the legacy structural classifier's standardizer and generator.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import audit_regression as A
import train_v7 as V7


def prepare(root,cache,encoder_cache):
    from extract_chemberta import MODELS,encode_smiles
    cache.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        state=A.make_state(root,cache)
    smiles=list(state['train'].SMILES)+list(state['test'].SMILES)
    embeddings=encode_smiles(smiles,'mlm','raw',encoder_cache,batch_size=1,threads=3)
    if embeddings.shape!=(len(smiles),768) or not np.isfinite(embeddings).all():
        raise ValueError('Unexpected frozen MLM representation')
    bits=[]
    for text in smiles:
        mol=V7.T.std(text)
        if mol is None:raise ValueError('Invalid molecule in structural reference')
        bits.append(V7.T.mg.GetFingerprintAsNumPy(mol))
    bits=np.asarray(bits,float)
    if bits.shape!=(len(smiles),2048) or not np.isfinite(bits).all():
        raise ValueError('Unexpected structural Morgan representation')
    np.save(cache/'mlm_raw.npy',embeddings)
    np.save(cache/'structural_bits.npy',bits)
    manifest=dict(model=MODELS['mlm'],view='raw',batch_size=1,threads=3,
                  shape=list(embeddings.shape),
                  input_sha256=hashlib.sha256('\n'.join(smiles).encode()).hexdigest())
    (cache/'embedding_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',default='work/similarity_gate_cache')
    parser.add_argument('--encoder-cache',default='work/chemberta_snapshots')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    prepare(root,root/args.cache,root/args.encoder_cache)


if __name__=='__main__':main()
