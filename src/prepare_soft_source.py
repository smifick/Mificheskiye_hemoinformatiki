"""Rebuild ignored frozen features for the isolated PR3 source-routing study.

No task head is fitted here. The encoder snapshot, raw-SMILES view, batch size
and CPU thread count are exactly those used by the PR3 production candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import audit_regression as A


def prepare(root,cache,encoder_cache):
    from extract_chemberta import MODELS,encode_smiles
    cache.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        state=A.make_state(root,cache)
    smiles=list(state['train'].SMILES)+list(state['test'].SMILES)
    encoded=encode_smiles(smiles,'mlm','raw',encoder_cache,batch_size=1,threads=3)
    if encoded.shape!=(len(smiles),768) or not np.isfinite(encoded).all():
        raise ValueError('Unexpected frozen MLM representation')
    np.save(cache/'mlm_raw.npy',encoded)
    manifest=dict(model=MODELS['mlm'],view='raw',batch_size=1,threads=3,
                  shape=list(encoded.shape),
                  input_sha256=hashlib.sha256('\n'.join(smiles).encode()).hexdigest())
    (cache/'embedding_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',default='work/soft_source_cache')
    parser.add_argument('--encoder-cache',default='work/chemberta_snapshots')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    prepare(root,root/args.cache,root/args.encoder_cache)


if __name__=='__main__':main()
