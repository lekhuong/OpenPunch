"""Revision v3 statistics for slabs with openings: series-balanced metrics, series-bootstrap CI of model-minus-EC2 differences,
non-conservative tail of Vu/Vpred, and a random-effects decomposition of ln(Vu/V_EC2)."""
import sys, json, warnings, numpy as np, pandas as pd
import os as _os; _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'))
warnings.filterwarnings('ignore')
from prep import load, add_codes
A, M = load(); M = add_codes(M)
y = M.Pu_kN.values; op = M.has_opening.values; ser = M.series.values
cands = {'ACI 318-19': M.Vu_ACI.values, 'Eurocode 2': M.Vu_EC2.values}
for tag, lab in [('CatBoost', 'CatBoost, data-driven'), ('CatBoost_datau', 'CatBoost, data-driven + perimeter ratio'), ('CatBoost_ec2', 'CatBoost, EC2-informed')]:
    if _os.path.exists(f'oof_{tag}.csv'): cands[lab] = pd.read_csv(f'oof_{tag}.csv').V_opt.values
rmse = lambda a, p: float(np.sqrt(np.mean((a - p) ** 2))); mape = lambda a, p: float(100 * np.mean(np.abs(a - p) / a))
def tail(a, p):
    r = a / p; ok = p > 0
    return dict(q05=float(np.percentile(r[ok], 5)), share_below_080=float(np.mean(r < 0.8) if ok.all() else np.nan), n_nonpos=int((~ok).sum()))
def per_series(mask, p):
    rows = []
    for k in np.unique(ser[mask]):
        i = mask & (ser == k); rows.append(dict(series=int(k), n=int(i.sum()), RMSE=rmse(y[i], p[i]), MAPE=mape(y[i], p[i])))
    return pd.DataFrame(rows)
rng = np.random.default_rng(42)
def boot_diff(mask, p, q, B=2000):
    ss = np.unique(ser[mask]); idx = {k: np.where(mask & (ser == k))[0] for k in ss}; d = []
    for _ in range(B):
        t = np.concatenate([idx[k] for k in rng.choice(ss, len(ss))])
        d.append([rmse(y[t], p[t]) - rmse(y[t], q[t]), mape(y[t], p[t]) - mape(y[t], q[t])])
    d = np.array(d); return dict(dRMSE_CI=np.percentile(d[:, 0], [2.5, 97.5]).tolist(), dMAPE_CI=np.percentile(d[:, 1], [2.5, 97.5]).tolist(),
                                 p_dMAPE_below0=float(np.mean(d[:, 1] < 0)), p_dRMSE_below0=float(np.mean(d[:, 0] < 0)))
out = {}
for nm, p in cands.items():
    PS = per_series(op, p)
    out[nm] = dict(openings=dict(RMSE=rmse(y[op], p[op]), MAPE=mape(y[op], p[op]), **tail(y[op], p[op])),
                   all=dict(**tail(y, p)),
                   series_balanced=dict(RMSE=float(PS.RMSE.mean()), MAPE=float(PS.MAPE.mean()), MAPE_median=float(PS.MAPE.median())),
                   per_series=PS.to_dict('records'))
    if nm != 'Eurocode 2':
        out[nm]['diff_vs_EC2_openings'] = dict(dRMSE=rmse(y[op], p[op]) - rmse(y[op], M.Vu_EC2.values[op]), dMAPE=mape(y[op], p[op]) - mape(y[op], M.Vu_EC2.values[op]),
                                               **boot_diff(op, p, M.Vu_EC2.values))
        e = per_series(op, M.Vu_EC2.values)
        out[nm]['series_better_than_EC2_MAPE'] = int((PS.MAPE.values < e.MAPE.values).sum())
# random-effects decomposition of ln(Vu/V_EC2) for slabs with openings
import statsmodels.formula.api as smf
O = M[op].copy(); O['lr'] = np.log(O.Pu_kN / O.Vu_EC2)
def icc(D):
    f = smf.mixedlm('lr ~ 1', D, groups=D['series']).fit(reml=True)
    tau2 = float(f.cov_re.iloc[0, 0]); s2 = float(f.scale); return tau2, s2, tau2 / (tau2 + s2)
tau2, s2, rho = icc(O)
bs = []
ss = O.series.unique()
for _ in range(1000):
    pick = rng.choice(ss, len(ss)); D = pd.concat([O[O.series == k].assign(series=j) for j, k in enumerate(pick)])
    try: bs.append(icc(D)[2])
    except Exception: pass
# equal weight per series: one-way ANOVA estimator with the unbalanced-design n0
g = O.groupby('series').lr; n_i = g.size().values; k = len(n_i); N = n_i.sum()
msb = (n_i * (g.mean().values - O.lr.mean()) ** 2).sum() / (k - 1); msw = ((O.lr - g.transform('mean')) ** 2).sum() / (N - k)
n0 = (N - (n_i ** 2).sum() / N) / (k - 1); tau2_a = max((msb - msw) / n0, 0)
out['variance_components'] = dict(n=int(len(O)), series=int(k), REML_tau2=tau2, REML_sigma2=s2, ICC_REML=rho, ICC_CI=np.percentile(bs, [2.5, 97.5]).tolist(), n_boot=len(bs),
                                  ICC_ANOVA=float(tau2_a / (tau2_a + msw)), simple_between_fraction=json.load(open('paired.json'))['ln_ratio_var']['between_fraction'])
json.dump(out, open('stats_v3.json', 'w'), indent=1)
for nm, v in out.items():
    if nm == 'variance_components': print(nm, {a: (np.round(b, 3) if not isinstance(b, list) else np.round(b, 3).tolist()) for a, b in v.items()}); continue
    print(nm, {a: round(b, 3) for a, b in v['openings'].items()}, 'all', {a: round(b, 3) for a, b in v['all'].items()}, 'balanced', {a: round(b, 2) for a, b in v['series_balanced'].items()})
    if 'diff_vs_EC2_openings' in v: print('   diff', {a: (np.round(b, 2).tolist() if isinstance(b, list) else round(b, 3)) for a, b in v['diff_vs_EC2_openings'].items()}, 'series better MAPE', v['series_better_than_EC2_MAPE'], '/ 18')
# leave-one-group-out over the slabs with openings
L = {}
for f, lab in [('data', 'CatBoost, data-driven'), ('datau', 'CatBoost, data-driven + perimeter ratio'), ('ec2', 'CatBoost, EC2-informed')]:
    D = pd.read_csv(f'lopo_CatBoost_{f}.csv'); assert (D.row754.values == M.row754.values).all(); L[lab] = D.V_pred.values
grpv = D.group.values; L = {'ACI 318-19': M.Vu_ACI.values, 'Eurocode 2': M.Vu_EC2.values, **L}
def by_group(p):
    return pd.DataFrame([dict(group=int(k), n=int((op & (grpv == k)).sum()), RMSE=rmse(y[op & (grpv == k)], p[op & (grpv == k)]), MAPE=mape(y[op & (grpv == k)], p[op & (grpv == k)]))
                         for k in np.unique(grpv[op])])
def boot_g(p, q, B=2000):
    ss = np.unique(grpv[op]); idx = {k: np.where(op & (grpv == k))[0] for k in ss}; d = []
    for _ in range(B):
        t = np.concatenate([idx[k] for k in rng.choice(ss, len(ss))]); d.append([rmse(y[t], p[t]) - rmse(y[t], q[t]), mape(y[t], p[t]) - mape(y[t], q[t])])
    d = np.array(d); return dict(dRMSE_CI=np.percentile(d[:, 0], [2.5, 97.5]).tolist(), dMAPE_CI=np.percentile(d[:, 1], [2.5, 97.5]).tolist())
E = by_group(M.Vu_EC2.values); out['lopo'] = {}
for nm, p in L.items():
    G = by_group(p); r = dict(RMSE=rmse(y[op], p[op]), MAPE=mape(y[op], p[op]), R2=float(1 - ((y[op] - p[op]) ** 2).sum() / ((y[op] - y[op].mean()) ** 2).sum()),
                          balanced_MAPE=float(G.MAPE.mean()), balanced_RMSE=float(G.RMSE.mean()), groups=len(G), **tail(y[op], p[op]))
    if nm != 'Eurocode 2':
        r['groups_better_MAPE'] = int((G.MAPE.values < E.MAPE.values).sum()); r.update(boot_g(p, M.Vu_EC2.values))
    out['lopo'][nm] = r
    print('LOPO', nm, {a: (np.round(b, 2).tolist() if isinstance(b, list) else round(b, 3)) for a, b in r.items()})
json.dump(out, open('stats_v3.json', 'w'), indent=1)
