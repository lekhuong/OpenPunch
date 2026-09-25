import json, joblib, numpy as np, pandas as pd
from prep import load, FEATS
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
A, M = load()
NAMES = ['d', 'c', 'sqrt_fc', 'rho', 'a_over_d', 'Dop', 'Sop']
X = pd.DataFrame(M[FEATS].values, columns=NAMES); y = M.Pu_kN.values
itr, ite = train_test_split(np.arange(len(M)), test_size=0.2, random_state=42)
bp = json.load(open('res_CatBoost.json'))['best_params']
g = CatBoostRegressor(random_seed=42, verbose=0, thread_count=2, **bp).fit(X.iloc[itr], y[itr])
p = g.predict(X.iloc[ite]); ref = pd.read_csv('pred_CatBoost.csv').query("subset=='test'").V_pred.values
print('identical to reported test predictions:', np.allclose(p, ref, atol=1e-6), np.abs(p-ref).max())
joblib.dump(g, 'best_catboost_model.joblib'); g.save_model('best_catboost_model.cbm')
case = pd.DataFrame([[135, 200, np.sqrt(30), 1.2, 850/135, 200, 100]], columns=NAMES)
print('example opening case:', g.predict(case))
big = pd.DataFrame([[275, 200, np.sqrt(111.72), 1.49, 1150/275, 0, 1000]], columns=NAMES); print('large solid example (measured 2450):', g.predict(big))
