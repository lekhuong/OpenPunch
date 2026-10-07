"""Leave-one-group-out over the slabs with openings: each held-out group is one opening test series with its control slabs
(the three consecutive series 9-11, which share the effective depth, are held out together). The model is developed on all other
tests, with hyperparameters tuned inside the training data by series-grouped 5-fold CV."""
import sys, json, warnings, time, numpy as np, pandas as pd, optuna
import os as _os; _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'))
warnings.filterwarnings('ignore'); optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedGroupKFold
from prep import load, add_codes, FEATS
from models import make, space, SEED
name = sys.argv[1]; FORM = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
A, M = load(); M = add_codes(M)
X = M[FEATS + (['u_ratio'] if FORM in ('ec2', 'datau') else [])].values; y = M.Pu_kN.values; base = M.Vu_EC2.values
T = np.log(y / base) if FORM == 'ec2' else y; g = M.series.values; s = M.has_opening.values.astype(int)
FIXED = {'CatBoost': {'bootstrap_type': 'Bernoulli'}, 'LightGBM': {'subsample_freq': 1}}.get(name, {})
back = (lambda p, idx: base[idx] * np.exp(p)) if FORM == 'ec2' else (lambda p, idx: p)
def holdout_groups():
    op_series = sorted(set(M.series[M.has_opening]))
    grp = {k: k for k in op_series}
    for k in (10, 11): grp[k] = 9          # consecutive series with the same effective depth (rows 107-141)
    return grp
grp = holdout_groups(); H = np.array([grp.get(k, -1) for k in g])
def gcv(idx, p, k=5):
    P = np.zeros(len(idx))
    for a, b in StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(X[idx], s[idx], g[idx]):
        P[b] = back(make(name, p).fit(X[idx][a], T[idx][a]).predict(X[idx][b]), idx[b])
    return P
pred = np.full(len(y), np.nan); t0 = time.time(); params = {}
for h in sorted(set(H[H >= 0])):
    te = np.where(H == h)[0]; tr = np.where(H != h)[0]
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(lambda t: np.abs(gcv(tr, space(name, t)) - y[tr]).mean(), n_trials=N)
    bp = {**st.best_params, **FIXED}; params[int(h)] = bp
    pred[te] = back(make(name, bp).fit(X[tr], T[tr]).predict(X[te]), te)
    print(name, FORM, 'group', h, 'done', round(time.time() - t0), flush=True)
tag = f'lopo_{name}_{FORM}'
pd.DataFrame({'row754': M.row754, 'series': g, 'group': H, 'has_opening': s, 'V_exp': y, 'V_EC2': base, 'V_pred': pred}).to_csv(tag + '.csv', index=False)
json.dump({'model': name, 'form': FORM, 'trials': N, 'params': params, 'secs': time.time() - t0}, open(tag + '.json', 'w'), indent=1, default=float)
print(tag, 'FINISHED', round(time.time() - t0))
