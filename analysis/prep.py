import pandas as pd, numpy as np
SENT = 1000.0
def load():
    A = pd.read_excel('../data/Full_Merged_754_Samples.xlsx', 'All_754_Samples')
    A['source'] = ['opening_programme'] * 232 + ['previous_db'] * 522
    A['has_opening'] = A.Opening_Size_mm > 0
    A['sqrt_fc'] = np.sqrt(A.fc_prime_MPa)
    M = A[~(A.has_opening & A.Opening_Dist_mm.isna())].copy()          # 10 multi-opening specimens without Sop
    M.loc[~M.has_opening, 'Opening_Dist_mm'] = SENT                        # all solid/control slabs: Dop=0, Sop=1000 mm
    M = M.reset_index().rename(columns={'index': 'row754'})
    return A, M
FEATS = ['d_mm', 'C_mm', 'sqrt_fc', 'rho_percent', 'a_over_d', 'Opening_Size_mm', 'Opening_Dist_mm']
