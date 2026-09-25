import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from prep import load, FEATS
A, M = load()
ORDER = ['CatBoost', 'GBRT', 'XGBoost', 'LightGBM', 'AdaBoost', 'RF']
R = {m: json.load(open(f'res_{m}.json')) for m in ORDER}
def met(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return dict(n=len(y), R2=1-((y-p)**2).sum()/((y-y.mean())**2).sum(), RMSE=np.sqrt(((y-p)**2).mean()), MAE=np.abs(y-p).mean(), MAPE=100*np.mean(np.abs((y-p)/y)))
rows = []; openm = {}
for m in ORDER:
    r = R[m]
    for s, cv, te in (('Default', r['default_cv'], r['default_test']), ('Optimized', r['opt_cv'], r['opt_test'])):
        rows.append([m, s, cv['R2'], cv['RMSE'], cv['MAE'], cv['MAPE'], te['R2'], te['RMSE'], te['MAE'], te['MAPE']])
    P = pd.read_csv(f'pred_{m}.csv'); P['has_opening'] = M.loc[P.row, 'has_opening'].values
    T = P[P.subset == 'test']
    openm[m] = {'test_all': met(T.V_exp, T.V_pred), 'test_open': met(T[T.has_opening].V_exp, T[T.has_opening].V_pred),
                'test_solid': met(T[~T.has_opening].V_exp, T[~T.has_opening].V_pred)}
T2 = pd.DataFrame(rows, columns=['model', 'setting', 'cvR2', 'cvRMSE', 'cvMAE', 'cvMAPE', 'tR2', 'tRMSE', 'tMAE', 'tMAPE'])
T2.to_csv('table2.csv', index=False); print(T2.round(3).to_string())
json.dump(openm, open('opening_metrics.json', 'w'), indent=1, default=float)
for m in ORDER: print(m, {k: (v['n'], round(v['R2'], 3), round(v['RMSE'], 1), round(v['MAE'], 1), round(v['MAPE'], 1)) for k, v in openm[m].items()})
# Fig 4 optimisation history
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
mk = dict(RF='D', GBRT='o', CatBoost='s', XGBoost='^', LightGBM='p', AdaBoost='v')
for m in ORDER:
    h = np.minimum.accumulate(R[m]['history']); ax.plot(np.arange(1, len(h) + 1), h, marker=mk[m], ms=3, markevery=5, lw=1.4, label=m)
ax.set_xlabel('Trial'); ax.set_ylabel('Best 10-fold CV MAE (kN)'); ax.grid(alpha=0.4, lw=0.4); ax.legend(ncol=3, fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig('fig4_opt.png', bbox_inches='tight'); plt.close(fig)
# Fig 5 predicted vs measured
vmax = 1.04 * max(M.Pu_kN.max(), max(pd.read_csv(f'pred_{m}.csv').V_pred.max() for m in ORDER))
fig, axs = plt.subplots(3, 2, figsize=(7.2, 10.2), dpi=300)
for ax, m, tag in zip(axs.flat, ORDER, 'abcdef'):
    P = pd.read_csv(f'pred_{m}.csv'); P['has_opening'] = M.loc[P.row, 'has_opening'].values
    tr = P[P.subset == 'train']; te = P[P.subset == 'test']
    ax.scatter(tr.V_exp, tr.V_pred, s=7, c='#b9c9e8', edgecolors='none', label='Training subset')
    ax.scatter(te[~te.has_opening].V_exp, te[~te.has_opening].V_pred, s=13, c='#e07b24', edgecolors='none', label='Test: solid/control slabs')
    ax.scatter(te[te.has_opening].V_exp, te[te.has_opening].V_pred, s=15, marker='^', c='#7a1f1f', edgecolors='none', label='Test: slabs with openings')
    ax.plot([0, vmax], [0, vmax], 'k--', lw=0.8)
    ax.set_xlim(0, vmax); ax.set_ylim(0, vmax); ax.set_aspect('equal')
    t = R[m]['opt_test']
    ax.text(0.96, 0.05, f"Test: $R^2$ = {t['R2']:.3f}\nRMSE = {t['RMSE']:.2f} kN", transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(fc='white', ec='0.6', lw=0.5))
    ax.set_title(f'({tag}) {m}', fontsize=10); ax.set_xlabel(r'Measured $V_u$ (kN)'); ax.set_ylabel(r'Predicted $V_u$ (kN)'); ax.grid(alpha=0.3, lw=0.4)
axs.flat[0].legend(loc='upper left', fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig('fig5_pred.png', bbox_inches='tight'); plt.close(fig)
print('best params', {m: R[m]['best_params'] for m in ORDER})
