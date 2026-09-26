import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
SEED = 42
def make(n, p=None):
    p = p or {}
    if n == 'RF': return RandomForestRegressor(random_state=SEED, n_jobs=2, **p)
    if n == 'GBRT': return GradientBoostingRegressor(random_state=SEED, **p)
    if n == 'XGBoost': return XGBRegressor(random_state=SEED, n_jobs=2, verbosity=0, **p)
    if n == 'LightGBM': return LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1, **p)
    if n == 'CatBoost': return CatBoostRegressor(random_seed=SEED, verbose=0, thread_count=2, **p)
    if n == 'AdaBoost':
        p = dict(p); md = p.pop('max_depth', 3)
        return AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=md, random_state=SEED), random_state=SEED, **p)
def space(n, t):
    if n == 'RF': return dict(n_estimators=t.suggest_int('n_estimators', 100, 1000, step=50), max_depth=t.suggest_int('max_depth', 3, 30),
        min_samples_split=t.suggest_int('min_samples_split', 2, 10), min_samples_leaf=t.suggest_int('min_samples_leaf', 1, 5),
        max_features=t.suggest_float('max_features', 0.3, 1.0))
    if n == 'GBRT': return dict(n_estimators=t.suggest_int('n_estimators', 100, 1000, step=50), learning_rate=t.suggest_float('learning_rate', 0.01, 0.3, log=True),
        max_depth=t.suggest_int('max_depth', 2, 8), subsample=t.suggest_float('subsample', 0.6, 1.0), min_samples_leaf=t.suggest_int('min_samples_leaf', 1, 10))
    if n == 'XGBoost': return dict(n_estimators=t.suggest_int('n_estimators', 100, 1000, step=50), learning_rate=t.suggest_float('learning_rate', 0.01, 0.3, log=True),
        max_depth=t.suggest_int('max_depth', 2, 10), subsample=t.suggest_float('subsample', 0.6, 1.0), colsample_bytree=t.suggest_float('colsample_bytree', 0.6, 1.0),
        min_child_weight=t.suggest_float('min_child_weight', 1, 10), reg_alpha=t.suggest_float('reg_alpha', 1e-3, 10, log=True), reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True))
    if n == 'LightGBM': return dict(n_estimators=t.suggest_int('n_estimators', 100, 1000, step=50), learning_rate=t.suggest_float('learning_rate', 0.01, 0.3, log=True),
        num_leaves=t.suggest_int('num_leaves', 8, 128), max_depth=t.suggest_int('max_depth', 3, 12), min_child_samples=t.suggest_int('min_child_samples', 5, 30),
        subsample=t.suggest_float('subsample', 0.6, 1.0), subsample_freq=1, colsample_bytree=t.suggest_float('colsample_bytree', 0.6, 1.0),
        reg_alpha=t.suggest_float('reg_alpha', 1e-3, 10, log=True), reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True))
    if n == 'CatBoost': return dict(iterations=t.suggest_int('iterations', 200, 1500, step=100), learning_rate=t.suggest_float('learning_rate', 0.01, 0.3, log=True),
        depth=t.suggest_int('depth', 3, 10), l2_leaf_reg=t.suggest_float('l2_leaf_reg', 1, 10, log=True), subsample=t.suggest_float('subsample', 0.6, 1.0), bootstrap_type='Bernoulli')
    if n == 'AdaBoost': return dict(n_estimators=t.suggest_int('n_estimators', 50, 500, step=25), learning_rate=t.suggest_float('learning_rate', 0.01, 2.0, log=True),
        loss=t.suggest_categorical('loss', ['linear', 'square', 'exponential']), max_depth=t.suggest_int('max_depth', 3, 12))
def metrics(yt, p):
    return dict(R2=1 - ((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum(), RMSE=float(np.sqrt(((yt - p) ** 2).mean())),
                MAE=float(np.abs(yt - p).mean()), MAPE=float(100 * np.mean(np.abs((yt - p) / yt))))
