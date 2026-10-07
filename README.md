# OpenPunch: punching shear of RC flat slabs with openings

This repository accompanies the manuscript

> H.Q. Nguyen, X.H. Nguyen, Q.S. Nguyen, D.D. Le, K. Le-Nguyen. *Do ensemble models for punching shear transfer to new test series? Slabs with openings and a Eurocode 2-informed correction.*

It contains the experimental database, the analysis scripts and results, the final model and the web application.

**Web application:** https://lekhuong.github.io/OpenPunch/

The application computes the Eurocode 2 punching resistance, with the control perimeter reduced for openings, and multiplies it by a correction factor predicted by a CatBoost model. The computation runs in the browser and reproduces the Python model exactly for all 742 tests.

## Contents

| Path | Content |
|---|---|
| `index.html`, `model.js` | Web application (static; served by GitHub Pages) |
| `data/Full_Merged_754_Samples.xlsx` | Compiled database of 754 tests: 232 specimens from test programmes on slabs with openings and 522 slabs from the authors' earlier database |
| `data/modelling_dataset_742.csv` | The 742 modelled tests, with the test-series assignment and the ACI 318-19 and Eurocode 2 resistances |
| `analysis/` | Scripts for the whole workflow |
| `analysis/results/` | Out-of-series predictions and metrics of all models, the benchmark, the opening-effect analysis and the final model |

## Modelling dataset

- **Exclusions.** From the 754 tests, 10 multi-opening slabs without a reported opening distance were excluded, and 2 duplicate records were removed.
- **Composition.** The resulting 742 tests comprise 185 slabs with openings, 36 control slabs and 521 slabs from the earlier database.
- **Test series.** The compiled workbook does not record the source publication of each test, so test series are reconstructed: consecutive specimens with the same shear span form a series, series with an identical set-up (d, c, a) are merged, and specimens with identical inputs share a series. This gives 258 series, 18 of which contain slabs with openings (listed in Supplementary Table S1 of the manuscript).
- **Inputs.**
  - *d*, *c*, √*f*′c, *ρ*, *a*/*d*, *D*op and *S*op;
  - for the Eurocode 2-informed model and for the control model (direct prediction of *V*u with the same extra input), also the ratio of the reduced to the full Eurocode 2 control perimeter.
- **Target.** *V*u. The Eurocode 2-informed model predicts ln(*V*u / *V*R,c,EC2).
- **Code resistances.** `analysis/codes_impl.py` computes ACI 318-19 and Eurocode 2 (EN 1992-1-1:2004) from the specimen variables, with mean strengths and no partial factors. Openings reduce the control perimeter by the part between the radial lines from the column centre tangent to each opening.

## Validation

- **Nested cross-validation grouped by test series.**
  - *Outer loop:* five folds, stratified by the presence of an opening, so every test receives a prediction from a model that has not seen its series.
  - *Inner loop:* Bayesian optimization, 50 trials, on a five-fold cross-validation that is also grouped by series.
- **Six algorithms** (RF, GBRT, XGBoost, LightGBM, CatBoost, AdaBoost) in three formulations: data-driven; data-driven with the perimeter ratio (control); Eurocode 2-informed.
- **Random versus grouped splitting.** Nested cross-validation of CatBoost repeated with random folds in both loops.
- **Leave-one-group-out for slabs with openings.** Each opening series is held out in turn with its control slabs (series 9-11 of Supplementary Table S1 together, 16 groups); hyperparameters tuned inside the training data.
- **Further statistics** (`stats_v3.py`): series-weighted errors, series-bootstrap intervals of model-minus-Eurocode 2 differences, the 5% fractile of *V*u/*V*pred and the share below 0.8, and a random-effects (REML) decomposition of ln(*V*u / *V*R,c,EC2).

Out-of-series performance of the selected model and of Eurocode 2:

| Method | All tests: R² | RMSE (kN) | MAPE (%) | Slabs with openings: R² | RMSE (kN) | MAPE (%) |
|---|---|---|---|---|---|---|
| Eurocode 2 | 0.902 | 106.7 | 21.4 | 0.634 | 115.5 | 25.6 |
| CatBoost, data-driven | 0.883 | 116.4 | 21.6 | 0.758 | 93.9 | 33.9 |
| CatBoost, Eurocode 2-informed (selected) | 0.935 | 86.8 | 17.2 | 0.800 | 85.4 | 27.8 |

## Reproducing the analysis

```
pip install -r analysis/requirements.txt
cd analysis
for m in RF GBRT XGBoost LightGBM CatBoost AdaBoost; do
  python nested_cv.py $m 50 data     # data-driven formulation
  python nested_cv.py $m 50 datau    # control: data-driven + perimeter ratio
  python nested_cv.py $m 50 ec2      # Eurocode 2-informed formulation
done
python nested_cv.py CatBoost 50 data random   # random folds in both loops
python nested_cv.py CatBoost 50 ec2 random
for f in data datau ec2; do python lopo.py CatBoost $f 20; done
python tables_figures.py             # Tables 2-3, Figs. 4-5, benchmark with series-bootstrap intervals
python final_model.py CatBoost ec2 100
python final_model.py CatBoost data 100
python opening_effect.py CatBoost CatBoost_ec2
python stats_v3.py
python fig1.py; python fig2_fig3.py; python fig6.py
python export_model.py results/final_CatBoost_ec2.json ../model.js
```

All outputs are written to `analysis/results/`.

## Use and limitations

The model estimates mean capacities and is intended for research and for the assessment of tests and existing slabs within the range of the database. It overestimates the capacity more often than Eurocode 2 and is not a design tool; it does not replace code-based design verification.

## License

- Code: MIT License, see `LICENSE`.
- Data and model: Creative Commons Attribution 4.0 International (CC BY 4.0). The test results were compiled from the published experimental studies cited in the manuscript. Please cite the manuscript and the original sources when using the database.
