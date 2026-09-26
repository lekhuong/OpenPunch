import pandas as pd, numpy as np
SENT = 1000.0
FEATS = ['d_mm', 'C_mm', 'sqrt_fc', 'rho_percent', 'a_over_d', 'Opening_Size_mm', 'Opening_Dist_mm']

class UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, i):
        while s.p[i] != i: s.p[i] = s.p[s.p[i]]; i = s.p[i]
        return i
    def u(s, a, b): s.p[s.f(a)] = s.f(b)

def load(path='../../data/Full_Merged_754_Samples.xlsx'):
    A = pd.read_excel(path, 'All_754_Samples')
    A['row754'] = np.arange(1, len(A) + 1)                                  # 1-based workbook row
    A['source'] = ['opening_programme'] * 232 + ['previous_db'] * 522
    A['has_opening'] = A.Opening_Size_mm > 0
    A['sqrt_fc'] = np.sqrt(A.fc_prime_MPa)
    M = A[~(A.has_opening & A.Opening_Dist_mm.isna())].copy()               # 10 multi-opening specimens without Sop
    M.loc[~M.has_opening, 'Opening_Dist_mm'] = SENT
    M = M[~M.duplicated(FEATS + ['Pu_kN'], keep='first')]                   # exact duplicate records (rows 61 and 374)
    M = M.reset_index(drop=True)
    # test series: consecutive specimens of the same source with the same shear span a = (a/d) d (within 3 %);
    # series sharing an identical set-up (d, c, a) and specimens with identical inputs are merged
    a = M.a_over_d * M.d_mm
    run = np.zeros(len(M), int); r = 0
    for i in range(1, len(M)):
        same = (M.source[i] == M.source[i - 1]) and abs(a[i] - a[i - 1]) <= 0.03 * a[i - 1]
        r += 0 if same else 1; run[i] = r
    uf = UF(r + 1)
    setup = M.d_mm.round(0).astype(str) + '|' + M.C_mm.round(0).astype(str) + '|' + a.round(-1).astype(str)
    for key in (setup, M[FEATS].round(4).astype(str).agg('|'.join, axis=1)):
        first = {}
        for i, k in enumerate(key):
            if k in first: uf.u(run[i], run[first[k]])
            else: first[k] = i
    roots = [uf.f(x) for x in run]
    M['series'] = pd.factorize(pd.Series(roots))[0]
    return A, M

def add_codes(M, cache='codes_cache.csv'):
    """ACI 318-19 and EC2 resistances from the specimen variables (codes_impl.py) and the EC2 perimeter ratio."""
    import os
    from codes_impl import vu_aci, vu_ec2, shadowed_length
    if os.path.exists(cache):
        C = pd.read_csv(cache)
        if len(C) == len(M) and (C.row754.values == M.row754.values).all():
            for k in ('Vu_ACI', 'Vu_EC2', 'u_ratio'): M[k] = C[k].values
            return M
    pos = M.Opening_Pos.astype(str).str.lower().str.replace('paralllel', 'parallel')
    shp = np.where(M.Opening_Shape == 'Circular', 'Circular', 'Square')
    aci, ec2, ur = [], [], []
    for i, r in M.iterrows():
        n = int(r.Num_Openings) if r.has_opening else 0; S = r.Opening_Dist_mm if n else 0
        aci.append(vu_aci(r.fc_prime_MPa, r.C_mm, r.d_mm, r.Opening_Size_mm, S, n, pos[i], shp[i]))
        ec2.append(vu_ec2(r.fc_prime_MPa, r.rho_percent, r.C_mm, r.d_mm, r.Opening_Size_mm, S, n, pos[i], shp[i]))
        u1 = 4 * r.C_mm + 4 * np.pi * r.d_mm
        red = shadowed_length(r.C_mm, 2 * r.d_mm, True, r.Opening_Size_mm, S, n, pos[i], shp[i]) if (n and S <= 6 * r.d_mm) else 0.0
        ur.append((u1 - red) / u1)
    M['Vu_ACI'] = aci; M['Vu_EC2'] = ec2; M['u_ratio'] = ur
    M[['row754', 'Vu_ACI', 'Vu_EC2', 'u_ratio']].to_csv(cache, index=False)
    return M
