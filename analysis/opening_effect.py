"""Opening effect relative to the companion control slabs of the same test series: measured vs EC2 perimeter ratio vs models."""
import sys, json, numpy as np, pandas as pd
from prep import load, add_codes
from codes_impl import vu_ec2
import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'), exist_ok=True); _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'))  # outputs go to analysis/results
models = sys.argv[1:] if len(sys.argv) > 1 else ['CatBoost', 'CatBoost_ec2']
A, M = load(); M = add_codes(M)
M['Vec2s'] = [vu_ec2(r.fc_prime_MPa, r.rho_percent, r.C_mm, r.d_mm) for _, r in M.iterrows()]
for m in models: M['P_' + m] = pd.read_csv(f'oof_{m}.csv').V_opt.values
res = {}; rows = []
for s, G in M[M.source == 'opening_programme'].groupby('series'):
    C = G[~G.has_opening]; O = G[G.has_opening]
    if len(C) == 0 or len(O) == 0: continue
    for i, r in O.iterrows():
        d = dict(series=s, adj=r.Opening_Dist_mm == 0, meas=(r.Pu_kN / r.Vec2s) / (C.Pu_kN / C.Vec2s).mean(), ec2=r.u_ratio)
        for m in models: d[m] = (r['P_' + m] / r.Vec2s) / (C['P_' + m] / C.Vec2s).mean()
        rows.append(d)
P = pd.DataFrame(rows); P.to_csv('paired_openings.csv', index=False)
def st(pred, meas):
    q = meas / pred
    return dict(mean_ratio=float(q.mean()), cov=float(q.std(ddof=1) / q.mean()), corr=float(np.corrcoef(pred, meas)[0, 1]), mae=float(np.abs(pred - meas).mean()))
res['n'] = len(P); res['series'] = int(P.series.nunique())
res['measured_mean'] = {'all': float(P.meas.mean()), 'adjacent': float(P[P.adj].meas.mean()), 'distance': float(P[~P.adj].meas.mean())}
for col in ['ec2'] + models:
    res[col] = {'all': st(P[col], P.meas), 'adjacent': st(P[P.adj][col], P[P.adj].meas), 'distance': st(P[~P.adj][col], P[~P.adj].meas),
                'pred_mean': float(P[col].mean())}
# variance of ln(Vu/V_EC2) for slabs with openings: between and within test series
O = M[M.has_opening].copy(); O['lr'] = np.log(O.Pu_kN / O.Vu_EC2)
tot = O.lr.var(ddof=0); within = O.groupby('series').lr.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum() / len(O)
res['ln_ratio_var'] = {'total': float(tot), 'within_series': float(within), 'between_fraction': float(1 - within / tot)}
C = M[(M.source == 'opening_programme') & ~M.has_opening]; r = C.Pu_kN / C.Vu_EC2
res['controls_ec2_ratio'] = {'n': len(C), 'mean': float(r.mean()), 'cov': float(r.std(ddof=1) / r.mean())}
json.dump(res, open('paired.json', 'w'), indent=1); print(json.dumps(res, indent=1))
