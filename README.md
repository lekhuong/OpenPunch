# OpenPunch: punching shear of RC flat slabs with openings

This repository accompanies the manuscript

> H.Q. Nguyen, X.H. Nguyen, Q.S. Nguyen, D.D. Le, N.K. Le. *Prediction of punching shear strength of reinforced concrete flat slabs with openings using ensemble machine learning.*

It contains the experimental database, the analysis scripts and results, the trained CatBoost model, and the web application.

**Web application:** https://lekhuong.github.io/OpenPunch/

The web application runs entirely in the browser. It evaluates the exported CatBoost model (`model.js`), and its predictions are identical to the Python model for all 744 specimens.

## Contents

| Path | Content |
|---|---|
| `index.html`, `model.js` | Web application (static; served by GitHub Pages) |
| `data/Full_Merged_754_Samples.xlsx` | Compiled database of 754 tests: 232 specimens from test programmes on slabs with openings (195 with openings, 37 control slabs) and 522 solid slabs from the authors' earlier database |
| `data/modelling_dataset_744.csv` | The 744 specimens used for modelling, with the model inputs, the train/test assignment and the code predictions used in the benchmark |
| `analysis/` | Scripts for the full workflow (data preparation, six ensemble models with Bayesian optimization, code benchmark, figures, SHAP) |
| `analysis/results/` | Metrics, best hyperparameters and specimen-level predictions of all six models |
| `streamlit/` | Streamlit version of the application (`app.py`) and the trained model (`best_catboost_model.joblib`) |

## Modelling dataset

- Ten specimens with two to four openings whose distance to the column was not reported are excluded. This leaves 744 specimens: 185 slabs with openings and 559 slabs without openings.
- **Inputs:**
  - effective depth *d* (mm);
  - column dimension *c* (mm; circular columns are converted to the square of equal perimeter, *c* = π*D*/4);
  - √*f*′c (√MPa);
  - flexural reinforcement ratio *ρ* (%);
  - shear span-to-depth ratio *a*/*d*;
  - opening size *D*op (mm);
  - clear distance from the column face to the opening *S*op (mm).
- **Target:** measured punching shear capacity *V*u (kN).
- **Slabs without openings,** including the control slabs, are encoded as *D*op = 0 and *S*op = 1000 mm.
- **Split:** 80/20 random train/test split (seed 42), giving 595 training and 149 test specimens. The test subset contains 44 slabs with openings.

## Results (held-out test subset)

| Model | R² | RMSE (kN) | MAE (kN) | MAPE (%) |
|---|---|---|---|---|
| CatBoost (selected) | 0.963 | 61.97 | 34.70 | 12.83 |
| GBRT | 0.966 | 58.96 | 37.20 | 13.82 |
| XGBoost | 0.958 | 66.13 | 38.05 | 13.45 |

For the 44 test slabs with openings, CatBoost gives R² = 0.890 and MAPE = 18.6%.

## Reproducing the analysis

```
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna shap openpyxl matplotlib scipy
cd analysis
for m in RF GBRT CatBoost XGBoost LightGBM AdaBoost; do python run_model.py $m 100; done
python post.py          # Table 2, Figs. 4 and 5, metrics for slabs with openings
python codes.py         # code and empirical benchmark on the same specimens
python final_model.py   # final CatBoost model with named features
python shap_rf.py       # SHAP analysis of the final model (Fig. 6)
python figs23.py        # Figs. 2 and 3
```

## Use and limitations

The model is intended for research and preliminary assessment within the range of the database. It does not replace code-based design verification.
