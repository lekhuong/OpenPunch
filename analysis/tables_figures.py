"""Tables 2-3 and Figs. 4-5 from the nested series-grouped CV of both formulations."""
import json, numpy as np, pandas as pd, matplotlib
import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'), exist_ok=True); _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'))  # outputs go to analysis/results
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from prep import load, add_codes
ORDER = ['CatBoost', 'GBRT', 'XGBoost', 'LightGBM', 'AdaBoost', 'RF']
FORMS = [('data', '', 'Data-driven'), ('ec2', '_ec2', 'EC2-informed')]
A, M = load(); M = add_codes(M)
def met(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); r = y / p
    return dict(n=len(y), R2=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), RMSE=float(np.sqrt(((y - p) ** 2).mean())),
                MAE=float(np.abs(y - p).mean()), MAPE=float(100 * np.mean(np.abs((y - p) / y))), ratio_mean=float(r.mean()), ratio_cov=float(r.std(ddof=1) / r.mean()))
P, J = {}, {}
for m in ORDER:
    for f, suf, lab in FORMS:
        P[(m, f)] = pd.read_csv(f'oof_{m}{suf}.csv'); J[(m, f)] = json.load(open(f'nested_{m}{suf}.json'))
        assert (P[(m, f)].row754.values == M.row754.values).all()
op = M.has_opening.values
rows = []
for m in ORDER:
    for f, suf, lab in FORMS:
        d = P[(m, f)]; a = met(d.V_exp, d.V_opt); o = met(d.V_exp[op], d.V_opt[op])
        rows.append(dict(model=m, form=lab, R2=a['R2'], RMSE=a['RMSE'], MAE=a['MAE'], MAPE=a['MAPE'], MAE_fold_sd=float(np.std(J[(m, f)]['opt_fold_MAE'], ddof=1)),
                         oR2=o['R2'], oRMSE=o['RMSE'], oMAE=o['MAE'], oMAPE=o['MAPE']))
T2 = pd.DataFrame(rows); T2.to_csv('table2.csv', index=False); print(T2.round(3).to_string())
best = {lab: T2[T2.form == lab].sort_values('MAE').iloc[0].model for _, _, lab in FORMS}
sel = T2.sort_values('MAE').iloc[0]; print('best per form', best, 'selected overall', sel.model, sel.form)
json.dump({'best': best, 'selected': [sel.model, sel.form]}, open('selection.json', 'w'))
# Table 3: codes vs best models, all specimens / slabs with openings / adjacent / at distance, series bootstrap CI
rng = np.random.default_rng(42)
cands = {'ACI 318-19': M.Vu_ACI.values, 'Eurocode 2': M.Vu_EC2.values,
         f"{best['Data-driven']} (data-driven)": P[(best['Data-driven'], 'data')].V_opt.values,
         f"{best['EC2-informed']} (EC2-informed)": P[(best['EC2-informed'], 'ec2')].V_opt.values}
y = M.Pu_kN.values; ser = M.series.values
subs = {'all': np.ones(len(y), bool), 'openings': op, 'adjacent': op & (M.Opening_Dist_mm.values == 0), 'distance': op & (M.Opening_Dist_mm.values > 0)}
def boot(mask, p, B=2000):
    ss = np.unique(ser[mask]); idx = {k: np.where(mask & (ser == k))[0] for k in ss}; out = []
    for _ in range(B):
        t = np.concatenate([idx[k] for k in rng.choice(ss, len(ss))]); out.append([met(y[t], p[t])[k] for k in ('R2', 'RMSE', 'MAPE')])
    return np.percentile(np.array(out), [2.5, 97.5], 0).T.tolist()
T3 = {}
for nm, p in cands.items():
    T3[nm] = {k: met(y[mk], p[mk]) for k, mk in subs.items()}
    T3[nm]['ci_openings'] = boot(op, p); T3[nm]['ci_all'] = boot(np.ones(len(y), bool), p)
json.dump(T3, open('table3.json', 'w'), indent=1)
for nm, v in T3.items():
    print(nm, {k: (x['n'], round(x['R2'], 3), round(x['RMSE'], 1), round(x['MAPE'], 1), round(x['ratio_mean'], 3), round(x['ratio_cov'], 3)) for k, x in v.items() if k in subs})
pd.DataFrame({'row754': M.row754, 'series': ser, 'has_opening': op, 'Vu': y, **{k: v for k, v in cands.items()}}).to_csv('benchmark_predictions.csv', index=False)
# Fig. 4: out-of-series MAE by model and formulation (mean and SD over outer folds)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, ax = plt.subplots(figsize=(6.8, 3.6)); x = np.arange(len(ORDER)); w = 0.36
for j, (f, suf, lab) in enumerate(FORMS):
    v = np.array([J[(m, f)]['opt_fold_MAE'] for m in ORDER])
    ax.bar(x + (j - 0.5) * w, v.mean(1), w, yerr=v.std(1, ddof=1), capsize=3, color=['#e0913a', '#1e5bb8'][j], label=lab, error_kw=dict(lw=0.8))
ax.axhline(np.abs(M.Vu_EC2 - M.Pu_kN).mean(), color='0.3', ls='--', lw=0.9, label='Eurocode 2')
ax.set_xticks(x); ax.set_xticklabels(ORDER); ax.set_ylabel('MAE on held-out series (kN)'); ax.legend(frameon=False, fontsize=8)
ax.grid(axis='y', alpha=0.3, lw=0.4); fig.tight_layout(); fig.savefig('fig4_models.png', dpi=300, bbox_inches='tight'); plt.close(fig)
# Fig. 5: predicted vs measured, EC2 and best model of each formulation
vmax = 1.04 * max(y.max(), max(v.max() for v in cands.values()))
fig, axs = plt.subplots(1, 3, figsize=(11, 4))
for ax, nm, tag in zip(axs, ['Eurocode 2', f"{best['Data-driven']} (data-driven)", f"{best['EC2-informed']} (EC2-informed)"], 'abc'):
    p = cands[nm]
    ax.scatter(y[~op], p[~op], s=8, c='#e07b24', edgecolors='none', label='Slabs without openings')
    ax.scatter(y[op], p[op], s=13, marker='^', c='#7a1f1f', edgecolors='none', label='Slabs with openings')
    ax.plot([0, vmax], [0, vmax], 'k--', lw=0.8); ax.set_xlim(0, vmax); ax.set_ylim(0, vmax); ax.set_aspect('equal')
    a = T3[nm]['all']; b = T3[nm]['openings']
    ax.text(0.96, 0.05, f"All: $R^2$ = {a['R2']:.3f}, MAPE = {a['MAPE']:.1f}%\nOpenings: $R^2$ = {b['R2']:.3f}, MAPE = {b['MAPE']:.1f}%",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5, bbox=dict(fc='white', ec='0.6', lw=0.5))
    ax.set_title(f'({tag}) {nm}', fontsize=10); ax.set_xlabel(r'Measured $V_u$ (kN)'); ax.set_ylabel(r'Predicted $V_u$ (kN)'); ax.grid(alpha=0.3, lw=0.4)
axs[0].legend(loc='upper left', fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig('fig5_pred.png', dpi=300, bbox_inches='tight'); plt.close(fig)
