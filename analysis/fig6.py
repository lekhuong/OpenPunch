import json, numpy as np, pandas as pd, matplotlib
import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'), exist_ok=True); _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'results'))  # outputs go to analysis/results
matplotlib.use('Agg'); import matplotlib.pyplot as plt
E = json.load(open('final_CatBoost_ec2_summary.json')); P = pd.read_csv('paired_openings.csv')
names = {'d_mm': r'$d$', 'C_mm': r'$c$', 'sqrt_fc': r'$\sqrt{f^\prime_c}$', 'rho_percent': r'$\rho$', 'a_over_d': r'$a/d$',
         'Opening_Size_mm': r'$D_{op}$', 'Opening_Dist_mm': r'$S_{op}$', 'u_ratio': r'$u_{1,red}/u_1$'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
ax = axs[0]; keys = list(E['shap_meanabs_all'].keys()); order = np.argsort([E['shap_meanabs_open'][k] for k in keys])
y = np.arange(len(keys)); h = 0.38
ax.barh(y - h / 2, [100 * E['shap_meanabs_all'][keys[i]] for i in order], h, color='#b9c9e8', label='All specimens')
ax.barh(y + h / 2, [100 * E['shap_meanabs_open'][keys[i]] for i in order], h, color='#1e5bb8', label='Slabs with openings')
ax.set_yticks(y); ax.set_yticklabels([names[keys[i]] for i in order]); ax.set_xlabel('mean |SHAP| of the correction (%)')
ax.set_title('(a) Contributions to the correction of EC2', fontsize=10); ax.legend(frameon=False, fontsize=8); ax.grid(axis='x', alpha=0.3, lw=0.4)
for ax, col, lab, tag in ((axs[1], 'ec2', 'EC2 perimeter ratio $u_{1,red}/u_1$', '(b)'), (axs[2], 'CatBoost_ec2', 'EC2-informed CatBoost', '(c)')):
    a = P[P.adj]; b = P[~P.adj]
    ax.scatter(a[col], a.meas, s=14, marker='^', c='#7a1f1f', edgecolors='none', label='Adjacent openings')
    ax.scatter(b[col], b.meas, s=14, marker='o', c='#1e5bb8', edgecolors='none', label='Openings at a distance')
    ax.plot([0.2, 1.4], [0.2, 1.4], 'k--', lw=0.8); ax.set_xlim(0.2, 1.4); ax.set_ylim(0.2, 1.4); ax.set_aspect('equal')
    q = P.meas / P[col]
    ax.text(0.97, 0.05, f'measured/predicted: mean {q.mean():.2f}, CoV {q.std(ddof=1) / q.mean():.2f}\nr = {np.corrcoef(P[col], P.meas)[0, 1]:.2f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5, bbox=dict(fc='white', ec='0.6', lw=0.5))
    ax.set_xlabel(f'Predicted reduction ({lab})', fontsize=8.5); ax.set_ylabel('Measured reduction relative to control slabs'); ax.grid(alpha=0.3, lw=0.4)
    ax.set_title(f'{tag} ' + ('EC2 control perimeter' if col == 'ec2' else 'EC2-informed model'), fontsize=10)
axs[1].legend(frameon=False, fontsize=7.5, loc='upper left')
fig.tight_layout(); fig.savefig('fig6_interp.png', dpi=300, bbox_inches='tight'); plt.close(fig)
