"""Executable implementations of ACI 318-19 and Eurocode 2 (EN 1992-1-1:2004) punching resistance,
evaluated with mean material strengths and without partial safety factors.

Geometry conventions (as in the database):
- c is the side of a square column, or the side of the square with the same perimeter for circular and rectangular columns;
- Dop is the opening dimension parallel to the column face (diameter for circular openings);
- Sop is the clear distance from the column face (or corner, for diagonal openings) to the opening.
Openings reduce the critical perimeter by the part enclosed between the radial lines from the column
centre that are tangent to the opening (ACI 318-19 22.6.4.3; EC2 6.4.2(3)). In EC2 the reduction applies only
to openings within 6d of the column face.
"""
import numpy as np

N_PTS = 20000


def perimeter_points(c, dist, rounded):
    """Points along the control perimeter at `dist` from the face of a square column of side c,
    centred at the origin. Returns (x, y, ds)."""
    h = c / 2
    pts = []
    if rounded:   # EC2: straight segments plus quarter circles of radius dist at the corners
        L_str, L_arc = c, np.pi * dist / 2
        total = 4 * (L_str + L_arc)
        s = (np.arange(N_PTS) + 0.5) * total / N_PTS
        seg = L_str + L_arc
        x = np.empty(N_PTS); y = np.empty(N_PTS)
        for q in range(4):
            m = (s >= q * seg) & (s < (q + 1) * seg)
            u = s[m] - q * seg
            # local frame: straight part along +y on the side x = h + dist, then arc around corner (h, h)
            xs = np.where(u < L_str, h + dist, h + dist * np.cos((u - L_str) / dist))
            ys = np.where(u < L_str, -h + u, h + dist * np.sin((u - L_str) / dist))
            ang = q * np.pi / 2
            x[m] = xs * np.cos(ang) - ys * np.sin(ang)
            y[m] = xs * np.sin(ang) + ys * np.cos(ang)
        return x, y, total / N_PTS
    a = h + dist   # ACI: square perimeter of side c + 2*dist
    total = 8 * a
    s = (np.arange(N_PTS) + 0.5) * total / N_PTS
    side = (s // (2 * a)).astype(int); u = s - side * 2 * a - a
    x = np.select([side == 0, side == 1, side == 2, side == 3], [np.full_like(u, a), -u, np.full_like(u, -a), u])
    y = np.select([side == 0, side == 1, side == 2, side == 3], [u, np.full_like(u, a), -u, np.full_like(u, -a)])
    return x, y, total / N_PTS


def opening_intervals(c, Dop, Sop, n, pos, shape):
    """Angular intervals (radians) shadowed by n openings."""
    h = c / 2; out = []
    diagonal = str(pos).lower().startswith('diag')
    for k in range(int(n)):
        rot = k * np.pi / 2
        if diagonal:   # opening near the column corner, nearest point at Sop from the corner along the diagonal
            p0 = np.array([h, h]) + Sop / np.sqrt(2)
            if shape == 'Circular':
                cen = p0 + Dop / 2 / np.sqrt(2); L = np.hypot(*cen); th = np.arcsin(min(Dop / 2 / L, 1)); mid = np.arctan2(cen[1], cen[0])
                lo, hi = mid - th, mid + th
            else:
                corners = [p0, p0 + [Dop, 0], p0 + [0, Dop], p0 + [Dop, Dop]]
                angs = [np.arctan2(q[1], q[0]) for q in corners]; lo, hi = min(angs), max(angs)
        else:          # opening centred on a column axis, near edge at Sop from the face
            if shape == 'Circular':
                L = h + Sop + Dop / 2; th = np.arcsin(min(Dop / 2 / L, 1)); lo, hi = -th, th
            else:
                th = np.arctan2(Dop / 2, h + Sop); lo, hi = -th, th
        out.append((lo + rot, hi + rot))
    return out


def shadowed_length(c, d_crit, rounded, Dop, Sop, n, pos, shape):
    if n == 0 or Dop == 0:
        return 0.0
    x, y, ds = perimeter_points(c, d_crit, rounded)
    a = np.arctan2(y, x); hit = np.zeros_like(a, bool)
    for lo, hi in opening_intervals(c, Dop, Sop, n, pos, shape):
        da_lo = (a - lo + np.pi) % (2 * np.pi) - np.pi
        width = hi - lo
        hit |= (da_lo >= 0) & (da_lo <= width)
    return hit.sum() * ds


def vu_aci(fc, c, d, Dop=0, Sop=0, n=0, pos='parallel', shape='Square', beta=1.0, alpha_s=40):
    b0 = 4 * (c + d) - shadowed_length(c, d / 2, False, Dop, Sop, n, pos, shape)
    lam_s = min(np.sqrt(2 / (1 + 0.004 * d)), 1.0)
    v = min(0.17 * (1 + 2 / beta) * lam_s * np.sqrt(fc), 0.083 * (2 + alpha_s * d / b0) * lam_s * np.sqrt(fc), 0.33 * lam_s * np.sqrt(fc))
    return v * b0 * d / 1e3


def vu_ec2(fc, rho_percent, c, d, Dop=0, Sop=0, n=0, pos='parallel', shape='Square'):
    k = min(1 + np.sqrt(200 / d), 2.0); r = min(rho_percent / 100, 0.02)
    v = max(0.18 * k * (100 * r * fc) ** (1 / 3), 0.035 * k ** 1.5 * np.sqrt(fc))
    u1 = 4 * c + 4 * np.pi * d
    if n and Dop and Sop <= 6 * d:
        u1 -= shadowed_length(c, 2 * d, True, Dop, Sop, n, pos, shape)
    return v * u1 * d / 1e3
