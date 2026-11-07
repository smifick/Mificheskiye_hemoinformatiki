"""Infer from the benchmark's own trusted model bundle using index and SMILES only."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from si_tail_evidence import structures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.input, usecols=['index', 'SMILES'])
    bundle = joblib.load(args.model)
    if bundle['format'] != 'si_tail_evidence_v1':
        raise ValueError('Unsupported artifact')
    features, _, _ = structures(data.SMILES)
    with threadpool_limits(limits=2):
        prediction = bundle['model'].predict(features)
    if not data['index'].is_unique or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Invalid submission; predictions are not clipped')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'index': data['index'], 'SI VACV': prediction}).to_csv(out, index=False)


if __name__ == '__main__':
    main()
