"""Final model for a given algorithm and formulation: hyperparameters tuned by series-grouped 5-fold CV on all specimens,
model fitted on all specimens. Also: random vs grouped CV contrast, SHAP and counterfactual opening curves."""
import sys, json, warnings, numpy as np, pandas as pd, optuna, joblib
import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'), exist_ok=True); _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'))  # outputs go to analysis/results
warnings.filterwarnings('ignore'); optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedGroupKFold, KFold
from prep import load, add_codes, FEATS
from codes_impl import vu_ec2, shadowed_length
from models import make, space, SEED
name = sys.argv[1]; FORM = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
FIXED = {'CatBoost': {'bootstrap_type': 'Bernoulli'}, 'LightGBM': {'subsample_freq': 1}}.get(name, {})
A, M = load(); M = add_codes(M)
COLS = FEATS + (['u_ratio'] if FORM == 'ec2' else [])
X = M[COLS].values; y = M.Pu_kN.values; g = M.series.values; s = M.has_opening.values.astype(int); base = M.Vu_EC2.values
T = np.log(y / base) if FORM == 'ec2' else y
back = (lambda p, idx: base[idx] * np.exp(p)) if FORM == 'ec2' else (lambda p, idx: p)
def cvpred(p, grouped=True, k=5):
    P = np.zeros(len(y)); idx = np.arange(len(y))
    sp = StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(X, s, g) if grouped else KFold(k, shuffle=True, random_state=SEED).split(X)
    for a, b in sp: P[b] = back(make(name, p).fit(X[a], T[a]).predict(X[b]), idx[b])
    return P
st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
st.optimize(lambda t: np.abs(cvpred(space(name, t)) - y).mean(), n_trials=N)
bp = {**st.best_params, **FIXED}
def met(yy, p): return dict(R2=float(1 - ((yy - p) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()), RMSE=float(np.sqrt(((yy - p) ** 2).mean())), MAPE=float(100 * np.mean(np.abs((yy - p) / yy))))
contrast = {lab: (lambda P: {'all': met(y, P), 'openings': met(y[s == 1], P[s == 1])})(cvpred(bp, gr)) for lab, gr in (('grouped', True), ('random', False))}
model = make(name, bp).fit(X, T)
tag = f'{name}_{FORM}'
joblib.dump(model, f'final_{tag}.joblib')
if name == 'CatBoost': model.save_model(f'final_{tag}.cbm'); model.save_model(f'final_{tag}.json', format='json')
# prediction for arbitrary opening geometry (recomputes the EC2 baseline and perimeter ratio)
pos = M.Opening_Pos.astype(str).str.lower().str.replace('paralllel', 'parallel').values
shp = np.where(M.Opening_Shape == 'Circular', 'Circular', 'Square')
def predict_row(r, i, Dop, Sop, n):
    if FORM == 'data':
        x = [r.d_mm, r.C_mm, r.sqrt_fc, r.rho_percent, r.a_over_d, Dop if n else 0, Sop if n else 1000.0]
        return float(model.predict(np.array([x]))[0])
    Vb = vu_ec2(r.fc_prime_MPa, r.rho_percent, r.C_mm, r.d_mm, Dop, Sop if n else 0, n, pos[i], shp[i])
    u1 = 4 * r.C_mm + 4 * np.pi * r.d_mm
    red = shadowed_length(r.C_mm, 2 * r.d_mm, True, Dop, Sop, n, pos[i], shp[i]) if (n and Sop <= 6 * r.d_mm) else 0.0
    x = [r.d_mm, r.C_mm, r.sqrt_fc, r.rho_percent, r.a_over_d, Dop if n else 0, Sop if n else 1000.0, (u1 - red) / u1]
    return float(Vb * np.exp(model.predict(np.array([x]))[0]))
O = M[M.has_opening]
sop = np.arange(0, 451, 25.0); dop = np.arange(100, 701, 50.0)
cs, cd, es, ed = [], [], [], []
ec2r = lambda r, i, D, S, n: vu_ec2(r.fc_prime_MPa, r.rho_percent, r.C_mm, r.d_mm, D, S, n, pos[i], shp[i]) / vu_ec2(r.fc_prime_MPa, r.rho_percent, r.C_mm, r.d_mm)
for i, r in O.iterrows():
    n = int(r.Num_Openings); V0 = predict_row(r, i, 0, 0, 0)
    cs.append([predict_row(r, i, r.Opening_Size_mm, v, n) / V0 for v in sop])
    cd.append([predict_row(r, i, v, r.Opening_Dist_mm, n) / V0 for v in dop])
    es.append([ec2r(r, i, r.Opening_Size_mm, v, n) for v in sop]); ed.append([ec2r(r, i, v, r.Opening_Dist_mm, n) for v in dop])
cs = np.array(cs); cd = np.array(cd); es = np.array(es); ed = np.array(ed)
q = lambda a, p: np.percentile(a, p, 0).tolist()
out = {'model': name, 'form': FORM, 'params': bp, 'history': [t.value for t in st.trials], 'contrast': contrast,
       'cf': {'sop': sop.tolist(), 'sop_med': q(cs, 50), 'sop_q25': q(cs, 25), 'sop_q75': q(cs, 75),
              'dop': dop.tolist(), 'dop_med': q(cd, 50), 'dop_q25': q(cd, 25), 'dop_q75': q(cd, 75),
              'ec2_sop_med': q(es, 50), 'ec2_dop_med': q(ed, 50)}}
try:
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    np.save(f'shap_{tag}.npy', sv)
    out['shap_meanabs_all'] = dict(zip(COLS, np.abs(sv).mean(0).round(4).tolist()))
    out['shap_meanabs_open'] = dict(zip(COLS, np.abs(sv[s == 1]).mean(0).round(4).tolist()))
except Exception as e:
    out['shap_error'] = str(e)
json.dump(out, open(f'final_{tag}.json' if name != 'CatBoost' else f'final_{tag}_summary.json', 'w'), indent=1, default=float)
print(tag, bp); print(json.dumps(contrast, indent=1))
