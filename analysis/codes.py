import numpy as np, pandas as pd, json
from prep import load
from sklearn.model_selection import train_test_split
A, M = load()
idx = np.arange(len(M)); itr, ite = train_test_split(idx, test_size=0.2, random_state=42)
M['subset'] = 'train'; M.loc[ite, 'subset'] = 'test'
def met(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return dict(n=len(y), R2=1-((y-p)**2).sum()/((y-y.mean())**2).sum(), RMSE=np.sqrt(((y-p)**2).mean()), MAE=np.abs(y-p).mean(), MAPE=100*np.mean(np.abs((y-p)/y)),
                ratio_mean=np.mean(y/p), ratio_cov=np.std(y/p)/np.mean(y/p))
out = {}
for c in ['Vu_ACI_kN', 'Vu_EC2_kN', 'Vu_El_Shafiey_kN', 'Vu_Koroglu_kN']:
    out[c] = {'db_all': met(M.Pu_kN, M[c]), 'db_open': met(M[M.has_opening].Pu_kN, M[M.has_opening][c]),
              'test_all': met(M[M.subset=='test'].Pu_kN, M[M.subset=='test'][c]),
              'test_open': met(M[(M.subset=='test')&M.has_opening].Pu_kN, M[(M.subset=='test')&M.has_opening][c])}
json.dump(out, open('codes.json', 'w'), indent=1, default=float)
for c, d in out.items():
    print(c, {k: (v['n'], round(v['R2'],3), round(v['RMSE'],1), round(v['MAPE'],1)) for k, v in d.items()})
print('test composition:', M[M.subset=='test'].groupby(['source','has_opening']).size().to_dict(), 'test Vu max', M[M.subset=='test'].Pu_kN.max())
