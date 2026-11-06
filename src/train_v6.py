"""v6 = v5 + «двухрежимная» модель кода 10000 по присутствию молекулы в выгрузке ChEMBL.

Наблюдение (train): если для молекулы в ChEMBL есть точное (relation '=') измерение против VACV
или любое CC50 / точное IC50 / EC50, то код 10000 встречается 1 раз из 89 строк (при фоне 25%).
Это признак ПРОИСХОЖДЕНИЯ записи (строки с кодом, по-видимому, не приходят из курируемых ChEMBL-источников),
а не химии. Признак строится только по SMILES и открытой базе (data/external/external_endpoints.csv),
колонки CC50_uM / IC50_uM_VACV из train/test не используются.

Модель кода:  p(x) = p_flag (сглаженная доля кода среди помеченных обучающих строк)   если флаг = 1
              p(x) = KRR-классификатор, обученный только на НЕпомеченных строках       если флаг = 0
Оценка p_flag и классификатор считаются внутри каждого фолда. Далее как в v5:
ŷ = 10000·p + (1 − p − p0)·r(x), r — компонент B из v5 (log1p + smearing + внешние V/C, w=1);
для молекул с точным внешним значением SI (lookup, уровни V/C) подставляется оно (λ=1).

Запуск: python src/train_v6.py  ->  submission_v6.csv, results/v6/metrics.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v5 as T

OUT = T.ROOT / 'results' / 'v6'; OUT.mkdir(parents=True, exist_ok=True)
CFG = T.CONFIGS['15 log_smear + внешние V+C, w=1']

def chembl_flags(W):
    ep = pd.read_csv(T.ROOT/'data/external/external_endpoints.csv',
                     usecols=['SMILES', 'endpoint', 'relation', 'is_vacv_assay']).dropna(subset=['SMILES'])
    key = {}
    for s in ep.SMILES.drop_duplicates():
        m = T.std(s)
        if m is not None: key[s] = T.taut_key(m)
    ep['tau'] = ep.SMILES.map(key); ep = ep.dropna(subset=['tau'])
    ep['exact'] = ep.relation.eq('='); ep['vacv'] = ep.is_vacv_assay.astype(bool)
    good = ep[(ep.endpoint == 'CC50') | (ep.exact & (ep.vacv | ep.endpoint.isin(['IC50', 'EC50'])))]
    present = set(good.tau)
    return np.array([W.tau[i] in present for i in range(W.n_tr + W.n_te)])

def code_prob(W, flag, trn, val, a_c):
    y = W.y; c = (y == 10000).astype(float); K = W.K
    ft = trn[flag[trn]]; pf = (c[ft].sum() + 0.5) / (len(ft) + 2)          # сглаживание Лапласа
    tu = trn[~flag[trn]]; mu = c[tu].mean()
    a = np.linalg.solve(K[np.ix_(tu, tu)] + a_c*np.eye(len(tu)), c[tu] - mu)
    pu = np.clip(mu + K[np.ix_(val, tu)] @ a, 0, 1)
    return np.where(flag[val], pf, pu)

def predict_v6(W, flag, trn, val, ext_mask):
    _, _, r = T.predict(W, trn, val, CFG, ext_mask=ext_mask)
    p = code_prob(W, flag, trn, val, CFG['a_c']); p0 = float((W.y[trn] == 0).mean())
    return np.clip(10000*p + np.clip(1 - p - p0, 0, 1)*r, 0, 10000), p, r

def main():
    W = T.World(); flag = chembl_flags(W); n = W.n_tr; y = W.y
    print(f'помеченных: train {flag[:n].sum()}/{n} (кодов среди них {(y[flag[:n]] == 10000).sum()}), test {flag[n:].sum()}/{W.n_te}')
    # CV: v5-core против v6, одинаковые фолды
    rows = []
    for seed in range(T.N_SEEDS):
        o5 = np.zeros(n); o6 = np.zeros(n)
        for val in T.make_folds(W.groups, seed):
            trn = np.setdiff1d(np.arange(n), val); em = T.ext_allowed(W, val)
            o5[val] = T.predict(W, trn, val, CFG, ext_mask=em)[0]
            o6[val] = predict_v6(W, flag, trn, val, em)[0]
        rows.append(dict(seed=seed, v5=((o5 - y)**2).mean(), v6=((o6 - y)**2).mean()))
    cv = pd.DataFrame(rows); cv.to_csv(OUT/'cv.csv', index=False)
    d = cv.v6 - cv.v5
    print(f'CV MSE: v5 {cv.v5.mean()/1e6:.3f}M | v6 {cv.v6.mean()/1e6:.3f}M | {100*d.mean()/cv.v5.mean():.1f}% | лучше в {(d < 0).sum()}/{len(cv)} повторах')
    # финал
    org = np.arange(n); te = np.arange(n, n + W.n_te)
    yh, p, r = predict_v6(W, flag, org, te, np.ones(W.n_ext, bool))
    lt = T.lookup_table(W); cov = np.array([W.tau[i] in lt.index for i in te])
    e = np.array([lt.get(W.tau[i], np.nan) for i in te])
    final = np.clip(np.where(cov, np.nan_to_num(np.clip(e, 0, 10000)), yh), 0, 10000)
    sub = pd.DataFrame({'index': W.test['index'].values, 'SI VACV': final})
    assert len(sub) == 597 and sub['SI VACV'].between(0, 10000).all() and sub['SI VACV'].notna().all()
    sub.to_csv(T.ROOT/'submission_v6.csv', index=False)
    m = dict(cv_v5_mean=float(cv.v5.mean()), cv_v6_mean=float(cv.v6.mean()), cv_change_pct=float(100*d.mean()/cv.v5.mean()),
             seeds_better=int((d < 0).sum()), test_flagged=int(flag[n:].sum()), test_lookup=int(cov.sum()),
             pred_mean=float(final.mean()), p_mean_flagged=float(p[flag[n:]].mean()), p_mean_unflagged=float(p[~flag[n:]].mean()))
    json.dump(m, open(OUT/'metrics.json', 'w'), indent=2); print(m)

if __name__ == '__main__':
    main()
