import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from prep import load
A, M = load()
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
panels = [('d_mm', r'$d$ (mm)', M.d_mm), ('C_mm', r'$c$ (mm)', M.C_mm), ('sqrt_fc', r'$\sqrt{f^\prime_c}$ ($\sqrt{\mathrm{MPa}}$)', M.sqrt_fc),
          ('rho_percent', r'$\rho$ (%)', M.rho_percent), ('a_over_d', r'$a/d$', M.a_over_d), ('Dop', r'$D_{op}$ (mm)', M.Opening_Size_mm),
          ('Sop', r'$S_{op}$ (mm), slabs with openings', M.loc[M.has_opening, 'Opening_Dist_mm']), ('Vu', r'$V_u$ (kN)', M.Pu_kN)]
fig, axs = plt.subplots(2, 4, figsize=(12, 5.4), dpi=300)
for ax, (k, lab, s) in zip(axs.flat, panels):
    s = s.dropna().values
    ax.hist(s, bins=20, color='#4f6fd6', edgecolor='white', linewidth=0.6)
    if s.std() > 0:
        xs = np.linspace(s.min(), s.max(), 300); kde = gaussian_kde(s)
        ax.plot(xs, kde(xs) * len(s) * (s.max() - s.min()) / 20, color='#d62728', lw=1.6)
    ax.set_title(lab, fontsize=10); ax.tick_params(labelsize=8)
    ax.text(0.97, 0.95, f'n = {len(s)}', transform=ax.transAxes, ha='right', va='top', fontsize=8)
fig.tight_layout(); fig.savefig('fig2_hist.png', bbox_inches='tight'); plt.close(fig)
cols = ['d_mm', 'C_mm', 'sqrt_fc', 'rho_percent', 'a_over_d', 'Opening_Size_mm', 'Opening_Dist_mm', 'Pu_kN']
labs = [r'$d$', r'$c$', r'$\sqrt{f^\prime_c}$', r'$\rho$', r'$a/d$', r'$D_{op}$', r'$S_{op}$', r'$V_u$']
C = M[cols].corr().values
fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
im = ax.imshow(C, cmap='coolwarm', vmin=-1, vmax=1)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f'{C[i,j]:.2f}', ha='center', va='center', fontsize=8.5, color='white' if abs(C[i, j]) > 0.6 else 'black')
ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols))); ax.set_xticklabels(labs, fontsize=10); ax.set_yticklabels(labs, fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig('fig3_corr.png', bbox_inches='tight'); plt.close(fig)
print(pd.DataFrame(C, index=cols, columns=cols).round(2))
