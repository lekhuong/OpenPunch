import joblib, numpy as np, pandas as pd, shap, matplotlib, json
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from prep import load, FEATS
A, M = load()
model = joblib.load('best_catboost_model.joblib')
X = pd.DataFrame(M[FEATS].values, columns=[r'$d$ (mm)', r'$c$ (mm)', r'$\sqrt{f^\prime_c}$', r'$\rho$ (%)', r'$a/d$', r'$D_{op}$ (mm)', r'$S_{op}$ (mm)'])
ex = shap.TreeExplainer(model); sv = ex.shap_values(X.values)
imp = np.abs(sv).mean(0); order = np.argsort(imp)
res = {'base_value': float(np.ravel(ex.expected_value)[0]), 'mean_abs': dict(zip(FEATS, imp.round(2).tolist())),
       'd_shap_max': float(sv[:, 0].max()), 'dop_min_openings': float(sv[M.has_opening.values, 5].min()),
       'dop_meanabs_openings': float(np.abs(sv[M.has_opening.values, 5]).mean()), 'sop_meanabs_openings': float(np.abs(sv[M.has_opening.values, 6]).mean()),
       'meanabs_openings': dict(zip(FEATS, np.abs(sv[M.has_opening.values]).mean(0).round(2).tolist()))}
json.dump(res, open('shap.json', 'w'), indent=1); print(res)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig = plt.figure(figsize=(12, 4.3), dpi=300)
ax1 = fig.add_axes([0.07, 0.12, 0.33, 0.8])
ax1.barh(np.array(X.columns)[order], imp[order], color='#1e88e5'); ax1.set_xlabel('mean(|SHAP value|) (kN)'); ax1.set_title('(a) Mean absolute SHAP value', fontsize=10)
ax1.grid(axis='x', alpha=0.3)
plt.sca(fig.add_axes([0.52, 0.12, 0.42, 0.8]))
shap.summary_plot(sv, X, show=False, plot_size=None, color_bar=True)
plt.gca().set_title('(b) SHAP summary', fontsize=10); plt.gca().set_xlabel('SHAP value (kN)')
fig.savefig('fig6_shap.png', bbox_inches='tight'); plt.close(fig)
