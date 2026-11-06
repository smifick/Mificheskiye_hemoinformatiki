"""v5: двухкомпонентная модель + внешние данные по уровням пригодности.

ŷ(x) = 10000·p(x) + (1 − p(x) − p0)·r(x)
  A: p(x)  — мягкая вероятность технического кода 10000. Обучается ТОЛЬКО на организаторской
             целевой колонке (y==10000), по SMILES.
  B: r(x)  — физический SI по SMILES. Обучается на измеренных организаторских строках
             (код 10000 и код 0 исключены) + внешние пары уровней V/C с весами.
  p0       — доля строк с кодом 0 (неопределённый SI), отдельный постоянный член.

Не используются: CC50_uM, IC50_uM_VACV, strain, Culture, Title (ни как признаки, ни для калибровки).

Внешние записи (ChEMBL, происхождение построчно: DOI, assay/activity ID, линии клеток):
  V — 56 пар, сверенных с таблицами публикаций, линия клеток совпадает;
  C — прочие пары с IC50-знаменателем, линии известны и совпадают (таблицы не сверены);
  X — исключены: знаменатель EC50, линии не совпадают или неизвестны.

Честность валидации: в каждом фолде внешние записи, у которых таутомер-нормализованная структура
или обобщённый каркас Bemis–Murcko совпадает с валидационной частью, удаляются из обучения.

Запуск: python src/train_v5.py            (эксперименты + финальный submission_v5.csv)
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.MolStandardize import rdMolStandardize as S
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'v5'; OUT.mkdir(parents=True, exist_ok=True)
N_SEEDS = 8

# ------------------------------------------------------------------ данные
def norm_cell(s):
    if pd.isna(s): return None
    s = re.sub(r'\bcells?\b', '', str(s).lower().strip())
    s = re.sub(r'[^a-z0-9]', '', s)
    return s or None

_lf = S.LargestFragmentChooser(preferOrganic=True); _un = S.Uncharger()
_te = S.TautomerEnumerator(); _te.SetMaxTautomers(100); _te.SetMaxTransforms(100)
def std(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else _un.uncharge(_lf.choose(m))
def taut_key(m):
    try: return Chem.MolToSmiles(_te.Canonicalize(m))
    except Exception: return Chem.MolToSmiles(m)
def gscaf(m):
    try: return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(m)))
    except Exception: return ''

mg = G.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
tt = G.GetTopologicalTorsionGenerator(fpSize=2048)
def tanimoto(A, B):
    ab = A @ B.T; aa = (A*A).sum(1)[:, None]; bb = (B*B).sum(1)[None, :]
    return ab / np.maximum(aa + bb - ab, 1e-9)
def minmax(A, B):
    K = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        K[i] = np.minimum(a, B).sum(1) / np.maximum(np.maximum(a, B).sum(1), 1e-9)
    return K

def load_external():
    cur = pd.read_csv(ROOT/'data/external_curated/vacv_si.csv')
    V = pd.DataFrame(dict(SMILES=cur.SMILES, SI=cur.SI, doi=cur.doi, cell=cur.cell_line.map(norm_cell), tier='V'))
    pr = pd.read_csv(ROOT/'data/external/external_si_database.csv')
    pr = pr[pr.denominator_endpoint == 'IC50'].copy()
    ca = pr.cell_line_antiviral.map(norm_cell); cc = pr.cell_line_cyto.map(norm_cell)
    pr = pr[ca.notna() & (ca == cc)]
    C = pd.DataFrame(dict(SMILES=pr.SMILES, SI=pr.SI, doi=pr.doi, cell=ca[pr.index], tier='C'))
    ext = pd.concat([V, C], ignore_index=True)
    ext = ext[np.isfinite(ext.SI) & (ext.SI > 0)].reset_index(drop=True)
    return ext

class World:
    """Единое индексное пространство: 0..319 train, 320..916 test, 917.. внешние записи."""
    def __init__(self):
        self.train = pd.read_csv(ROOT/'data/raw/train.csv'); self.test = pd.read_csv(ROOT/'data/raw/test.csv')
        ext = load_external()
        smis = list(self.train.SMILES) + list(self.test.SMILES) + list(ext.SMILES)
        mols = [std(s) for s in smis]
        ok = np.array([m is not None for m in mols]); assert ok[:917].all()
        keep_ext = ok[917:]; ext = ext[keep_ext].reset_index(drop=True)
        mols = [m for m in mols if m is not None]
        self.ext = ext
        self.n_tr, self.n_te, self.n_ext = len(self.train), len(self.test), len(ext)
        self.y = self.train['SI VACV'].values.astype(float)
        self.tau = np.array([taut_key(m) for m in mols], dtype=object)
        self.scf = np.array([gscaf(m) for m in mols], dtype=object)
        # дубликат уровня C, уже входящий в V (та же структура+DOI+линия) — оставляем V
        key = pd.Series(list(zip(self.tau[917:], ext.doi, ext.cell)))
        dup = key.duplicated(keep='first').values
        self.ext = ext[~dup].reset_index(drop=True)
        self.ext_pos = np.where(~dup)[0] + 917
        keep_all = np.r_[np.arange(917), self.ext_pos]
        Xb = np.array([mg.GetFingerprintAsNumPy(mols[i]) for i in keep_all], float)
        Xt = np.array([tt.GetCountFingerprintAsNumPy(mols[i]) for i in keep_all], float)
        self.K = (tanimoto(Xb, Xb) + minmax(Xt, Xt)) / 2
        self.tau = self.tau[keep_all]; self.scf = self.scf[keep_all]
        self.n_ext = len(self.ext)
        self.ext_ids = np.arange(917, 917 + self.n_ext)
        self.ext_si = self.ext.SI.values.astype(float)
        self.ext_tier = self.ext.tier.values
        self.groups = pd.factorize(self.scf[:self.n_tr])[0]

# ------------------------------------------------------------------ вспомогательное
def make_folds(groups, seed, k=5):
    rng = np.random.RandomState(seed); ug = np.unique(groups)
    sizes = {u: (groups == u).sum() for u in ug}
    order = sorted(ug, key=lambda u: -sizes[u] + rng.rand()*0.5)
    bins = [[] for _ in range(k)]; load = np.zeros(k)
    for u in order:
        j = int(np.argmin(load + rng.rand(k)*0.1)); bins[j].append(u); load[j] += sizes[u]
    return [np.where(np.isin(groups, b))[0] for b in bins]

def ext_allowed(W, val_ids, use_scaffold=True):
    """Маска внешних записей, допустимых для обучения при данной валидационной части."""
    tset = set(W.tau[val_ids]); ok = np.array([W.tau[i] not in tset for i in W.ext_ids])
    if use_scaffold:
        sset = set(s for s in W.scf[val_ids] if s)
        ok &= np.array([(W.scf[i] not in sset) or (W.scf[i] == '') for i in W.ext_ids])
    return ok

def wkrr(K, ids, t, w, te_ids, alpha):
    m0 = np.average(t, weights=w)
    a = np.linalg.solve(K[np.ix_(ids, ids)] + alpha*np.diag(1.0/w), t - m0)
    return m0 + K[np.ix_(te_ids, ids)] @ a

def inner_folds(groups_str, k=5, seed=0):
    codes = pd.factorize(np.asarray(groups_str, dtype=object))[0]
    return make_folds(codes, seed, k)

# ------------------------------------------------------------------ компоненты
def fit_B_core(W, org_rows, ext_mask, ext_w, te_ids, transform, a_m, cap):
    """Сырой прогноз в пространстве преобразования для te_ids."""
    y = W.y
    ids_o = org_rows[(y[org_rows] != 10000) & (y[org_rows] != 0)]
    sel = np.where(ext_mask & (ext_w > 0))[0]
    ids = np.r_[ids_o, W.ext_ids[sel]]
    yy = np.r_[y[ids_o], W.ext_si[sel]]
    w = np.r_[np.ones(len(ids_o)), ext_w[sel]]
    t = np.minimum(yy, cap) if transform == 'raw' else np.log1p(yy)
    return wkrr(W.K, ids, t, w, te_ids, a_m)

def back(f, transform, scale=1.0, smear=1.0):
    if transform == 'raw': return np.maximum(f, 0)
    if transform == 'log_exp': return np.maximum(np.expm1(f), 0)
    if transform == 'log_smear': return np.maximum(smear*np.exp(f) - 1, 0)
    if transform == 'log_cal': return np.maximum(scale*np.expm1(f), 0)
    raise ValueError

def fit_B(W, org_rows, ext_mask, ext_w, te_ids, cfg):
    tr = cfg['B']; core_t = 'raw' if tr == 'raw' else 'log'
    if tr == 'raw':
        return back(fit_B_core(W, org_rows, ext_mask, ext_w, te_ids, 'raw', cfg['a_m'], cfg['cap']), 'raw')
    f = fit_B_core(W, org_rows, ext_mask, ext_w, te_ids, 'log', cfg['a_m'], cfg['cap'])
    if tr == 'log_exp': return back(f, 'log_exp')
    # внутренняя калибровка обратного преобразования на OOF внутри обучающей части фолда
    meas = org_rows[(W.y[org_rows] != 10000) & (W.y[org_rows] != 0)]
    oof = np.full(len(W.y), np.nan)
    for vi in inner_folds(W.scf[meas], 5, 0):
        v = meas[vi]; trn = np.setdiff1d(org_rows, v)
        em = ext_mask & ext_allowed(W, v)
        oof[v] = fit_B_core(W, trn, em, ext_w, v, 'log', cfg['a_m'], cfg['cap'])
    fo = oof[meas]; yo = W.y[meas]
    if tr == 'log_smear':
        smear = float(np.mean(np.exp(np.log1p(yo) - fo))); return back(f, 'log_smear', smear=smear)
    r = np.maximum(np.expm1(fo), 0); yc = np.minimum(yo, 5000.0)
    scale = float((r*yc).sum() / max((r*r).sum(), 1e-9)); scale = float(np.clip(scale, 0.3, 3.0))
    return back(f, 'log_cal', scale=scale)

def fit_A(W, org_rows, te_ids, cfg):
    y = W.y; c = (y == 10000).astype(float); K = W.K
    def raw(tr_rows, ids):
        mu = c[tr_rows].mean()
        a = np.linalg.solve(K[np.ix_(tr_rows, tr_rows)] + cfg['a_c']*np.eye(len(tr_rows)), c[tr_rows] - mu)
        return mu + K[np.ix_(ids, tr_rows)] @ a, mu
    p, mu = raw(org_rows, te_ids)
    if cfg.get('cal_p'):
        oof = np.zeros(len(y))
        for vi in inner_folds(W.scf[org_rows], 5, 0):
            v = org_rows[vi]; oof[v] = raw(np.setdiff1d(org_rows, v), v)[0]
        d = oof[org_rows] - mu; b = float((d*(c[org_rows]-mu)).sum() / max((d*d).sum(), 1e-9))
        p = mu + float(np.clip(b, 0.6, 1.5))*(p - mu)
    return np.clip(p, 0, 1)

def predict(W, org_rows, te_ids, cfg, ext_mask=None, ext_w_tier=None):
    if ext_mask is None: ext_mask = np.ones(W.n_ext, bool)
    wt = cfg.get('w', {})  # {'V':..,'C':..}
    ext_w = np.array([wt.get(t, 0.0) for t in W.ext_tier])
    p = fit_A(W, org_rows, te_ids, cfg)
    r = fit_B(W, org_rows, ext_mask, ext_w, te_ids, cfg)
    p0 = float((W.y[org_rows] == 0).mean())
    return np.clip(10000*p + np.clip(1 - p - p0, 0, 1)*r, 0, 10000), p, r

# ------------------------------------------------------------------ эксперименты
BASE = dict(a_c=0.5, a_m=3.0, cap=2000.0, B='raw', cal_p=False, w={})
def C(**kw): d = dict(BASE); d.update(kw); return d

CONFIGS = {
    '0 baseline KRR (v1-подобный, один компонент)': None,
    '1 v4-A: два компонента, raw-cap, без внешних': C(),
    '2 + log1p, простая экспонента': C(B='log_exp'),
    '3 + log1p, smearing (Duan)': C(B='log_smear'),
    '4 + log1p, калибровка масштаба во внутр. фолдах': C(B='log_cal'),
    '5 raw-cap + калибровка p(x)': C(cal_p=True),
    '6 log_cal + калибровка p(x)': C(B='log_cal', cal_p=True),
    '7 raw-cap + внешние V, w=0.5': C(w={'V': 0.5}),
    '8 raw-cap + внешние V, w=1': C(w={'V': 1.0}),
    '9 raw-cap + внешние V+C, w=0.25': C(w={'V': 0.25, 'C': 0.25}),
    '10 raw-cap + внешние V+C, w=0.5': C(w={'V': 0.5, 'C': 0.5}),
    '11 raw-cap + внешние V+C, w=1': C(w={'V': 1.0, 'C': 1.0}),
    '12 log_cal + внешние V+C, w=0.5': C(B='log_cal', w={'V': 0.5, 'C': 0.5}),
    '13 log_cal + внешние V+C, w=1': C(B='log_cal', w={'V': 1.0, 'C': 1.0}),
    '14 log_smear + внешние V+C, w=0.5': C(B='log_smear', w={'V': 0.5, 'C': 0.5}),
    '15 log_smear + внешние V+C, w=1': C(B='log_smear', w={'V': 1.0, 'C': 1.0}),
    '16 log_smear + внешние V+C, w=2': C(B='log_smear', w={'V': 2.0, 'C': 2.0}),
}

def baseline_krr(W, tri, tei):
    K = W.K; y = W.y; mu = y[tri].mean()
    return mu + K[np.ix_(tei, tri)] @ np.linalg.solve(K[np.ix_(tri, tri)] + np.eye(len(tri)), y[tri] - mu)

def run_cv(W, configs=CONFIGS):
    n = W.n_tr; rows = []; oofs = {k: [] for k in configs}
    for seed in range(N_SEEDS):
        folds = make_folds(W.groups, seed)
        for name, cfg in configs.items():
            oof = np.zeros(n)
            for val in folds:
                trn = np.setdiff1d(np.arange(n), val)
                if cfg is None: oof[val] = baseline_krr(W, trn, val); continue
                em = ext_allowed(W, val)
                oof[val] = predict(W, trn, val, cfg, ext_mask=em)[0]
            oofs[name].append(oof); rows.append(dict(config=name, seed=seed, mse=((oof - W.y)**2).mean()))
    df = pd.DataFrame(rows); return df, oofs

def summarize(df, ref_cfg):
    piv = df.pivot(index='seed', columns='config', values='mse'); ref = piv[ref_cfg]
    out = []
    for c in piv.columns:
        d = piv[c] - ref
        out.append(dict(config=c, mse_mean=piv[c].mean(), mse_std=piv[c].std(),
                        vs_ref_pct=100*d.mean()/ref.mean(), seeds_better=int((d < 0).sum())))
    return pd.DataFrame(out)

# ------------------------------------------------------------------ валидация по публикациям (B-компонент)
def run_pub_holdout(W, min_n=6):
    ext = W.ext; dois = ext.doi.value_counts(); dois = dois[dois >= min_n].index.tolist()
    org_all = np.arange(W.n_tr)
    cfgs = {'no external': C(B='log_cal'),
            'V, w=0.5': C(B='log_cal', w={'V': 0.5}),
            'V+C, w=0.5': C(B='log_cal', w={'V': 0.5, 'C': 0.5}),
            'V+C, w=1': C(B='log_cal', w={'V': 1.0, 'C': 1.0}),
            'raw-cap, no external': C(), 'raw-cap, V+C w=0.5': C(w={'V': 0.5, 'C': 0.5}),
            'log_smear, no external': C(B='log_smear'),
            'log_smear, V+C w=0.5': C(B='log_smear', w={'V': 0.5, 'C': 0.5}),
            'log_smear, V+C w=1': C(B='log_smear', w={'V': 1.0, 'C': 1.0}),
            'log_smear, V+C w=2': C(B='log_smear', w={'V': 2.0, 'C': 2.0})}
    res = {k: [] for k in cfgs}; truth = []; dd = []
    for d in dois:
        tgt = W.ext_ids[(ext.doi == d).values]
        mask = ext_allowed(W, tgt, use_scaffold=False) & ~(ext.doi == d).values   # вне публикации, без пересечения молекул
        tset = set(W.tau[tgt]); trn = np.array([i for i in org_all if W.tau[i] not in tset])
        for k, cfg in cfgs.items():
            ext_w = np.array([cfg['w'].get(t, 0.0) for t in W.ext_tier])
            res[k].append(fit_B(W, trn, mask, ext_w, tgt, cfg))
        truth.append(W.ext_si[tgt - 917]); dd += [d]*len(tgt)
    truth = np.concatenate(truth); rows = []
    const = np.log10(np.maximum(truth, 1e-9))
    for k in cfgs:
        p = np.concatenate(res[k]); lp = np.log10(np.maximum(p, 1e-3)); lt = np.log10(truth)
        rows.append(dict(config=k, n=len(truth), n_pubs=len(dois), rmse_log10=float(np.sqrt(np.mean((lp - lt)**2))),
                         spearman=float(spearmanr(p, truth)[0]), mse_raw_capped5000=float(np.mean((np.minimum(p, 5000) - np.minimum(truth, 5000))**2))))
    base = np.full(len(truth), np.median(W.y[(W.y != 0) & (W.y != 10000)]))
    rows.append(dict(config='constant (median of organizer measured)', n=len(truth), n_pubs=len(dois),
                     rmse_log10=float(np.sqrt(np.mean((np.log10(base) - np.log10(truth))**2))), spearman=float('nan'),
                     mse_raw_capped5000=float(np.mean((np.minimum(base, 5000) - np.minimum(truth, 5000))**2))))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ точный внешний lookup
def lookup_table(W, tiers=('V', 'C')):
    df = pd.DataFrame(dict(tau=W.tau[W.ext_ids], si=W.ext_si, tier=W.ext_tier))
    df = df[df.tier.isin(tiers)]
    return df.groupby('tau').si.agg(lambda s: float(np.exp(np.log(s).mean())))

def eval_lookup(W, oof_mean):
    lk = lookup_table(W); cov = np.array([W.tau[i] in lk.index for i in range(W.n_tr)])
    e = np.array([lk.get(W.tau[i], np.nan) for i in range(W.n_tr)]); y = W.y
    out = [dict(lam=lam, n_cov=int(cov.sum()),
                mse_cov=float(np.mean(((1 - lam)*oof_mean[cov] + lam*np.clip(e[cov], 0, 10000) - y[cov])**2)),
                mse_overall=float(np.mean(np.where(cov, (1 - lam)*oof_mean + lam*np.nan_to_num(np.clip(e, 0, 10000)), oof_mean) - y)**2) if False else
                float(np.mean((np.where(cov, (1 - lam)*oof_mean + lam*np.nan_to_num(np.clip(e, 0, 10000)), oof_mean) - y)**2)))
           for lam in [0, 0.25, 0.5, 0.75, 1.0]]
    agree = dict(n=int(cov.sum()), code_rate_covered=float(np.isin(y[cov], [0, 10000]).mean()),
                 code_rate_uncovered=float(np.isin(y[~cov], [0, 10000]).mean()),
                 median_abs_log10_ratio=float(np.median(np.abs(np.log10(e[cov]) - np.log10(np.maximum(y[cov], 1e-3))))))
    return pd.DataFrame(out), agree

# ------------------------------------------------------------------ основной запуск
def main():
    W = World()
    print(f'train {W.n_tr}, test {W.n_te}, внешние записи {W.n_ext}: ' + str(dict(pd.Series(W.ext_tier).value_counts())), flush=True)
    summary = {}
    # 1) ablation на одинаковых фолдах
    df, oofs = run_cv(W); df.to_csv(OUT/'ablation_cv_raw.csv', index=False)
    ref = '1 v4-A: два компонента, raw-cap, без внешних'
    sm = summarize(df, ref); sm.to_csv(OUT/'ablation_cv_summary.csv', index=False)
    print('\n=== Ablation (scaffold-CV, 8 повторов, внешние с общими структурами/каркасами удалены) ===')
    print(sm.assign(mse_mean=lambda d: (d.mse_mean/1e6).round(3), mse_std=lambda d: (d.mse_std/1e6).round(3), vs_ref_pct=lambda d: d.vs_ref_pct.round(2)).to_string(index=False), flush=True)
    summary['ablation'] = sm.to_dict('records')
    # 2) валидация по публикациям
    pub = run_pub_holdout(W); pub.to_csv(OUT/'pub_holdout.csv', index=False)
    print('\n=== Hold-out по публикациям (компонент B; log10 SI) ===\n' + pub.round(3).to_string(index=False), flush=True)
    summary['pub_holdout'] = pub.to_dict('records')
    # 3) выбор конфигурации: минимальный средний CV-MSE среди конфигураций >=1
    # Правило выбора (зафиксировано до просмотра Kaggle): B-преобразование — log_smear (лучшее в CV среди
    # вариантов); вес внешних данных — тот из {0, 0.5, 1, 2}, у которого потеря в организаторской CV не
    # превышает 0.25% (внутри шума), а MSE на отложенных публикациях минимален.
    cand = {0.0: ('3 + log1p, smearing (Duan)', 'log_smear, no external'),
            0.5: ('14 log_smear + внешние V+C, w=0.5', 'log_smear, V+C w=0.5'),
            1.0: ('15 log_smear + внешние V+C, w=1', 'log_smear, V+C w=1'),
            2.0: ('16 log_smear + внешние V+C, w=2', 'log_smear, V+C w=2')}
    ok = {}
    for w, (cn, pn) in cand.items():
        loss = float(sm[sm.config == cn].vs_ref_pct.iloc[0]); pm = float(pub[pub.config == pn].mse_raw_capped5000.iloc[0])
        if loss <= 0.25: ok[w] = pm
    wbest = min(ok, key=ok.get); best = cand[wbest][0]
    cfg = CONFIGS[best]; print('\nвыбрана:', best, flush=True)
    om = np.mean(oofs[best], 0)
    lk, agree = eval_lookup(W, om); lk.to_csv(OUT/'lookup_eval.csv', index=False)
    print('\n=== Точный внешний lookup на покрытых строках train ===\n', agree, '\n', lk.round(1).to_string(index=False), flush=True)
    summary['lookup_agreement'] = agree; summary['lookup_eval'] = lk.to_dict('records'); summary['chosen'] = best
    # 4) финал: обучение на всём train + все внешние записи
    te_ids = np.arange(W.n_tr, W.n_tr + W.n_te); org = np.arange(W.n_tr)
    yhat, p, r = predict(W, org, te_ids, cfg, ext_mask=np.ones(W.n_ext, bool))
    lam = float(lk.sort_values('mse_overall').iloc[0].lam)
    lt = lookup_table(W); cov = np.array([W.tau[i] in lt.index for i in te_ids])
    e = np.array([lt.get(W.tau[i], np.nan) for i in te_ids])
    final = np.where(cov, (1 - lam)*yhat + lam*np.nan_to_num(np.clip(e, 0, 10000)), yhat)
    final = np.clip(final, 0, 10000)
    sub = pd.DataFrame({'index': W.test['index'].values, 'SI VACV': final})
    assert len(sub) == 597 and sub['SI VACV'].notna().all() and sub['SI VACV'].between(0, 10000).all()
    sub.to_csv(ROOT/'submission_v5.csv', index=False)
    summary['final'] = dict(lookup_lambda=lam, test_rows_covered=int(cov.sum()), pred_mean=float(final.mean()),
                            p_mean=float(p.mean()), r_median=float(np.median(r)), cfg={k: v for k, v in cfg.items()})
    json.dump(summary, open(OUT/'metrics.json', 'w'), indent=2, ensure_ascii=False, default=str)
    print('\nfinal:', summary['final'])

if __name__ == '__main__':
    main()
