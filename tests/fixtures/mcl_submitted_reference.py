"""Numerical code supplied by the user on 2026-10-17. Unchanged equations.
Retained solely as a reproduction fixture, not as validated physical truth.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

nx, ny = 64, 64
Lx, Ly = 75.4e-3, 72.0e-3
dx, dy = Lx / nx, Ly / ny
area_cell = dx * dy
dz = [0.5e-3, 0.1e-3, 1.0e-3, 1.0e-3]

io_mask = np.zeros((nx, ny), dtype=bool)
io_mask[26:38, 26:38] = True

def make_masks(coords):
    ccd_masks = []
    for (ix, iy) in coords:
        m = np.zeros((nx, ny), dtype=bool)
        m[ix:ix+8, iy:iy+7] = True
        ccd_masks.append(m)
    chip_mask = io_mask.copy()
    for m in ccd_masks:
        chip_mask |= m
    return ccd_masks, chip_mask

def solve_mcl_bp(D_chan, coords, modes, dP=15000.0, q_bp=0.02):
    mu = 8.9e-4
    H_ch = dz[3]
    Kh_max = (H_ch ** 2) / (12.0 * mu)
    alpha_min = 1.0 / Kh_max
    alpha_max = alpha_min * 1e5

    alpha_D = alpha_max + (alpha_min - alpha_max) * (D_chan * (1.0 + q_bp)) / (D_chan + q_bp)
    Kh = 1.0 / alpha_D

    N2 = nx * ny
    grid_idx = np.arange(N2).reshape(nx, ny)
    rows, cols, vals = [], [], []

    kx = 2.0 * Kh[:-1, :] * Kh[1:, :] / (Kh[:-1, :] + Kh[1:, :])
    Gx = (kx * (dy * H_ch) / dx).ravel()
    u, v = grid_idx[:-1, :].ravel(), grid_idx[1:, :].ravel()
    rows.extend([u, v, u, v]); cols.extend([u, v, v, u]); vals.extend([Gx, Gx, -Gx, -Gx])

    ky = 2.0 * Kh[:, :-1] * Kh[:, 1:] / (Kh[:, :-1] + Kh[:, 1:])
    Gy = (ky * (dx * H_ch) / dy).ravel()
    u, v = grid_idx[:, :-1].ravel(), grid_idx[:, 1:].ravel()
    rows.extend([u, v, u, v]); cols.extend([u, v, v, u]); vals.extend([Gy, Gy, -Gy, -Gy])

    inlet_mask = np.zeros(ny, dtype=bool); inlet_mask[24:40] = True
    outlet_mask = np.zeros(ny, dtype=bool); outlet_mask[24:40] = True

    G_in = (2.0 * Kh[0, inlet_mask] * (dy * H_ch) / dx).ravel()
    u_in = grid_idx[0, inlet_mask].ravel()
    rows.append(u_in); cols.append(u_in); vals.append(G_in)

    G_out = (2.0 * Kh[-1, outlet_mask] * (dy * H_ch) / dx).ravel()
    u_out = grid_idx[-1, outlet_mask].ravel()
    rows.append(u_out); cols.append(u_out); vals.append(G_out)

    rhs_p = np.zeros(N2, dtype=np.float64)
    rhs_p[u_in] = G_in * dP
    K_p = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N2, N2))
    P = spla.spsolve(K_p, rhs_p).reshape(nx, ny)

    Qx = kx * (dy * H_ch / dx) * (P[:-1, :] - P[1:, :])
    Qy = ky * (dx * H_ch / dy) * (P[:, :-1] - P[:, 1:])
    Q_in = G_in * (dP - P[0, inlet_mask])
    Q_out = G_out * (P[-1, outlet_mask] - 0.0)

    ccd_masks, chip_mask = make_masks(coords)
    k_bot = np.full((nx, ny), 400.0)
    k_top = 0.6 + (400.0 - 0.6) * ((1.0 - D_chan) ** 3)

    N = 4 * N2
    t_rows, t_cols, t_vals = [], [], []
    for l in range(4):
        if l == 0: k_layer = np.where(chip_mask, 130.0, 0.0)
        elif l == 1: k_layer = np.where(chip_mask, 5.0, 0.0)
        elif l == 2: k_layer = k_bot
        else: k_layer = k_top

        k_x1, k_x2 = k_layer[:-1, :], k_layer[1:, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            k_x = np.where((k_x1 > 0) & (k_x2 > 0), 2.0 * k_x1 * k_x2 / (k_x1 + k_x2), 0.0)
        G_tx = (k_x * (dy * dz[l]) / dx).ravel()
        u, v = l * N2 + grid_idx[:-1, :].ravel(), l * N2 + grid_idx[1:, :].ravel()
        m = G_tx > 0
        t_rows.extend([u[m], v[m], u[m], v[m]]); t_cols.extend([u[m], v[m], v[m], u[m]]); t_vals.extend([G_tx[m], G_tx[m], -G_tx[m], -G_tx[m]])

        k_y1, k_y2 = k_layer[:, :-1], k_layer[:, 1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            k_y = np.where((k_y1 > 0) & (k_y2 > 0), 2.0 * k_y1 * k_y2 / (k_y1 + k_y2), 0.0)
        G_ty = (k_y * (dx * dz[l]) / dy).ravel()
        u, v = l * N2 + grid_idx[:, :-1].ravel(), l * N2 + grid_idx[:, 1:].ravel()
        m = G_ty > 0
        t_rows.extend([u[m], v[m], u[m], v[m]]); t_cols.extend([u[m], v[m], v[m], u[m]]); t_vals.extend([G_ty[m], G_ty[m], -G_ty[m], -G_ty[m]])

    R_01 = (0.5 * dz[0] / 130.0) + 2.0e-6 + (0.5 * dz[1] / 5.0)
    G_01 = np.where(chip_mask, area_cell / R_01, 0.0).ravel()
    u, v = 0 * N2 + grid_idx.ravel(), 1 * N2 + grid_idx.ravel()
    m = G_01 > 0
    t_rows.extend([u[m], v[m], u[m], v[m]]); t_cols.extend([u[m], v[m], v[m], u[m]]); t_vals.extend([G_01[m], G_01[m], -G_01[m], -G_01[m]])

    R_12 = (0.5 * dz[1] / 5.0) + 2.0e-6 + (0.5 * dz[2] / k_bot)
    G_12 = np.where(chip_mask, area_cell / R_12, 0.0).ravel()
    u, v = 1 * N2 + grid_idx.ravel(), 2 * N2 + grid_idx.ravel()
    m = G_12 > 0
    t_rows.extend([u[m], v[m], u[m], v[m]]); t_cols.extend([u[m], v[m], v[m], u[m]]); t_vals.extend([G_12[m], G_12[m], -G_12[m], -G_12[m]])

    h_fluid_local = 15000.0 * (D_chan ** 2) + (1.0 - D_chan ** 2) * (k_top / (0.5 * dz[3]))
    R_23 = (0.5 * dz[2] / k_bot) + (1.0 / h_fluid_local)
    G_23 = (area_cell / R_23).ravel()
    u, v = 2 * N2 + grid_idx.ravel(), 3 * N2 + grid_idx.ravel()
    t_rows.extend([u, v, u, v]); t_cols.extend([u, v, v, u]); t_vals.extend([G_23, G_23, -G_23, -G_23])

    rho_cp = 4.18e6
    u_x, v_x = 3 * N2 + grid_idx[:-1, :].ravel(), 3 * N2 + grid_idx[1:, :].ravel()
    qp_x, qm_x = (rho_cp * np.maximum(Qx, 0.0)).ravel(), (rho_cp * np.maximum(-Qx, 0.0)).ravel()
    t_rows.extend([u_x, u_x, v_x, v_x]); t_cols.extend([u_x, v_x, u_x, v_x]); t_vals.extend([qp_x, -qm_x, -qp_x, qm_x])

    u_y, v_y = 3 * N2 + grid_idx[:, :-1].ravel(), 3 * N2 + grid_idx[:, 1:].ravel()
    qp_y, qm_y = (rho_cp * np.maximum(Qy, 0.0)).ravel(), (rho_cp * np.maximum(-Qy, 0.0)).ravel()
    t_rows.extend([u_y, u_y, v_y, v_y]); t_cols.extend([u_y, v_y, u_y, v_y]); t_vals.extend([qp_y, -qm_y, -qp_y, qm_y])

    u_out_3 = 3 * N2 + grid_idx[-1, outlet_mask].ravel()
    t_rows.append(u_out_3); t_cols.append(u_out_3); t_vals.append(rho_cp * np.maximum(Q_out, 0.0))

    u_all3 = 3 * N2 + grid_idx.ravel()
    t_rows.append(u_all3); t_cols.append(u_all3); t_vals.append(np.full(N2, 20.0 * area_cell))

    inactive = (~chip_mask).ravel()
    for l in [0, 1]:
        u = l * N2 + grid_idx.ravel()[inactive]
        t_rows.append(u); t_cols.append(u); t_vals.append(np.ones_like(u, dtype=np.float64))

    K_t = sp.csc_matrix((np.concatenate(t_vals), (np.concatenate(t_rows), np.concatenate(t_cols))), shape=(N, N))
    lu_t = spla.splu(K_t)
    RHS = np.zeros((N, len(modes)), dtype=np.float64)
    for s, p in enumerate(modes):
        q_map = np.zeros((nx, ny))
        q_map[io_mask] = 50.0 / (np.sum(io_mask) * area_cell)
        for m, pw in zip(ccd_masks, p):
            q_map[m] = pw / (np.sum(m) * area_cell)
        RHS[0:N2, s] = (q_map * area_cell).ravel()
    TH = lu_t.solve(RHS)
    peaks = [float(TH[:, s].reshape(4, nx, ny)[0, chip_mask].max()) for s in range(len(modes))]
    return max(peaks), peaks, float(Q_in.sum() * 1e6), P, TH
