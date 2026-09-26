"""Nested, series-grouped cross-validation.
Outer: 5-fold StratifiedGroupKFold by test series (stratified by presence of an opening) -> out-of-series predictions for every specimen.
Inner: Optuna TPE, N trials, minimising 5-fold grouped CV MAE on the outer training data."""
import sys, json, warnings, time, numpy as np, pandas as pd, optuna
import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'), exist_ok=True); _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'))  # outputs go to analysis/results
warnings.filterwarnings('ignore'); optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedGroupKFold
from prep import load, add_codes, FEATS
from models import make, space, metrics, SEED
name = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
FORM = sys.argv[3] if len(sys.argv) > 3 else 'data'   # 'data': target Vu; 'ec2': target ln(Vu/V_EC2)
A, M = load(); M = add_codes(M)
X = M[FEATS + (['u_ratio'] if FORM == 'ec2' else [])].values; y = M.Pu_kN.values; base = M.Vu_EC2.values; T = np.log(y / base) if FORM == 'ec2' else y
g = M.series.values; s = M.has_opening.values.astype(int)
def back(p, idx): return base[idx] * np.exp(p) if FORM == 'ec2' else p
FIXED = {'CatBoost': {'bootstrap_type': 'Bernoulli'}, 'LightGBM': {'subsample_freq': 1}}.get(name, {})
def gcv(idx, p, k=5):
    P = np.zeros(len(idx))
    for a, b in StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(X[idx], s[idx], g[idx]):
        P[b] = back(make(name, p).fit(X[idx][a], T[idx][a]).predict(X[idx][b]), idx[b])
    return P
outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
pred_def = np.zeros(len(y)); pred_opt = np.zeros(len(y)); fold = np.zeros(len(y), int); params = []; t0 = time.time()
for k, (tr, te) in enumerate(outer.split(X, s, g)):
    fold[te] = k
    pred_def[te] = back(make(name).fit(X[tr], T[tr]).predict(X[te]), te)
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(lambda t: np.abs(gcv(tr, space(name, t)) - y[tr]).mean(), n_trials=N)
    bp = {**st.best_params, **FIXED}; params.append(bp)
    pred_opt[te] = back(make(name, bp).fit(X[tr], T[tr]).predict(X[te]), te)
    print(name, 'fold', k, 'done', round(time.time() - t0), flush=True)
out = {'model': name, 'form': FORM, 'trials': N, 'params': params, 'secs': time.time() - t0,
       'default_all': metrics(y, pred_def), 'opt_all': metrics(y, pred_opt),
       'opt_open': metrics(y[s == 1], pred_opt[s == 1]), 'opt_solid': metrics(y[s == 0], pred_opt[s == 0]),
       'default_fold_MAE': [float(np.abs(pred_def[fold == k] - y[fold == k]).mean()) for k in range(5)],
       'opt_fold_MAE': [float(np.abs(pred_opt[fold == k] - y[fold == k]).mean()) for k in range(5)]}
pd.DataFrame({'row754': M.row754, 'series': g, 'has_opening': s, 'fold': fold, 'V_exp': y, 'V_def': pred_def, 'V_opt': pred_opt}).to_csv(f'oof_{name}' + ('_ec2' if FORM == 'ec2' else '') + '.csv', index=False)
json.dump(out, open(f'nested_{name}' + ('_ec2' if FORM == 'ec2' else '') + '.json', 'w'), indent=1, default=float)
print(name, 'FINISHED', out['opt_all'], out['opt_open'])
