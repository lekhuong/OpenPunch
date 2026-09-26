import matplotlib
import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'), exist_ok=True); _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'))  # outputs go to analysis/results
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family'] = 'Liberation Serif'
fig, ax = plt.subplots(figsize=(10, 7.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
TC = '#1a33c4'; AC = '#4472c4'
def box(x, y, w, h, title, body, fs=11):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='square,pad=0', fc='white', ec='black', lw=1.0))
    ax.text(x, y + h / 2 - 3.2, title, ha='center', va='center', fontsize=12.5, fontweight='bold', color=TC)
    ax.text(x, y - 1.9, body, ha='center', va='center', fontsize=fs, linespacing=1.3)
def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=13, lw=1.4, color=AC, shrinkA=0, shrinkB=0))
def line(xs, ys): ax.plot(xs, ys, color=AC, lw=1.4)
box(50, 92, 70, 12, 'Experimental database', '742 tests (185 slabs with openings, 36 control slabs, 521 solid slabs)\ngrouped into 258 test series (18 with openings)', 10.5)
box(24, 71, 44, 14, 'Data-driven formulation', 'target $V_u$\nseven input variables')
box(76, 71, 44, 14, 'EC2-informed formulation', r'target $\ln(V_u / V_{R,c,EC2})$' + '\ninputs + control perimeter ratio')
box(50, 49, 74, 14, 'Nested cross-validation by test series', 'six ensemble algorithms; Bayesian hyperparameter optimization\nin the inner loop; out-of-series predictions for every test', 10.5)
box(21, 26, 38, 13, 'Benchmark', 'ACI 318-19 and Eurocode 2\nwith opening deductions')
box(71, 26, 50, 13, 'Opening effect', 'measured versus predicted reduction\nrelative to companion control slabs')
box(50, 6, 60, 10, 'Selected model', 'SHAP interpretation and web application')
arrow((50, 86), (50, 82)); line([24, 76], [82, 82]); arrow((24, 82), (24, 78)); arrow((76, 82), (76, 78))
line([24, 24], [64, 60]); line([76, 76], [64, 60]); line([24, 76], [60, 60]); arrow((50, 60), (50, 56))
line([50, 50], [42, 37]); line([21, 71], [37, 37]); arrow((21, 37), (21, 32.5)); arrow((71, 37), (71, 32.5))
line([21, 21], [19.5, 15]); line([71, 71], [19.5, 15]); line([21, 71], [15, 15]); arrow((50, 15), (50, 11))
plt.savefig('fig1_workflow.png', dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
plt.savefig('fig1_workflow.pdf', bbox_inches='tight', pad_inches=0.05, facecolor='white')
