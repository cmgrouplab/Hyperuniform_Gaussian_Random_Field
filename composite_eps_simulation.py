#!/usr/bin/env python3
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from fenics import *

# ---------- 图像配置：全局字体放大 ----------
plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlesize": 18,
    "legend.fontsize": 16,
})

def setup_logging(job_dir):
    log_file = os.path.join(job_dir, "simulation.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ])
    return logging.getLogger()

class PeriodicBoundary(SubDomain):
    def __init__(self, L_temp, tol=1e-10):
        super().__init__()
        self.L_temp = L_temp
        self.tol = tol

    def inside(self, x, on_boundary):
        return bool(on_boundary and (
            (near(x[0], 0.0, self.tol) and x[1] > self.tol and x[1] < self.L_temp - self.tol) or
            (near(x[1], 0.0, self.tol) and x[0] > self.tol and x[0] < self.L_temp - self.tol)
        ))

    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1]
        if near(x[0], self.L_temp, self.tol):
            y[0] = x[0] - self.L_temp
        if near(x[1], self.L_temp, self.tol):
            y[1] = x[1] - self.L_temp

def generate_field_regularized(F_white, dx, alpha, sigma=2.0, k0=0.1,
                               C=1.0, target_std=1.0):
    N = F_white.shape[0]
    kx = np.fft.fftfreq(N, d=dx)*2*np.pi
    ky = np.fft.fftfreq(N, d=dx)*2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    exponent = alpha * np.log(K + k0) - (K**2)/(2*sigma**2)
    S = np.exp(exponent)
    M = np.max(S)
    S_norm = (S / M)*C if M > 0 else S
    A = np.sqrt(S_norm)
    F_field = A * F_white
    field2d = np.fft.ifftn(F_field).real
    field2d -= np.mean(field2d)
    std_val = np.std(field2d)
    if std_val > 0:
        field2d /= std_val
        field2d *= target_std
    return field2d, S_norm

def compute_power_spectrum_2d(field2d):
    F = np.fft.fftn(field2d)
    PS = np.abs(F)**2
    return np.fft.fftshift(PS)

def interpolate_conductivity_to_fenics(k_field, V0, size, L_cond, L_temp):
    k_fenics = Function(V0)
    dof_coords = V0.tabulate_dof_coordinates().reshape(-1, 2)
    xi = (dof_coords[:, 0] * size / L_temp).astype(int)
    yi = (dof_coords[:, 1] * size / L_temp).astype(int)
    xi = np.clip(xi, 0, size - 1)
    yi = np.clip(yi, 0, size - 1)
    values = k_field[yi, xi]
    k_fenics.vector().set_local(values)
    return k_fenics

def define_variational_problem_periodic(V, k_fenics, G):
    T_p = TrialFunction(V)
    v = TestFunction(V)
    a = k_fenics * dot(grad(T_p), grad(v)) * dx
    L_form = - k_fenics * dot(G, grad(v)) * dx
    return a, L_form

def solve_problem_periodic(V, a, L_form):
    T_p_sol = Function(V)
    solve(a == L_form, T_p_sol, solver_parameters={"linear_solver": "mumps"})
    return T_p_sol

def create_periodic_mesh_and_space(size, L_temp):
    from dolfin import Point, RectangleMesh
    mesh = RectangleMesh(Point(0, 0), Point(L_temp, L_temp), size, size)
    pbc = PeriodicBoundary(L_temp=L_temp, tol=1e-10)
    V = FunctionSpace(mesh, 'P', 1, constrained_domain=pbc)
    return mesh, V

def run_simulation_periodic(size, L_cond, L_temp, alpha, baseline, logger,
                            sigma=2.0, k0=0.1, target_std=1.0,
                            macro_gradient=(1.0, 0.0), offset=0.0):
    np.random.seed(42)
    noise = np.random.normal(0, 1, (size, size))
    F_white = np.fft.fftn(noise)
    dx_local = L_cond / size
    field_no_shift, _ = generate_field_regularized(
        F_white, dx_local, alpha, sigma, k0, C=1.0, target_std=target_std
    )
    field_positive = field_no_shift * 10 + baseline
    mesh, V = create_periodic_mesh_and_space(size, L_temp)
    V0 = FunctionSpace(mesh, 'DG', 0)
    k_fenics = interpolate_conductivity_to_fenics(field_positive, V0, size, L_cond, L_temp)
    Gx_val, Gy_val = macro_gradient
    G_const = Constant((Gx_val, Gy_val))
    a, L_form = define_variational_problem_periodic(V, k_fenics, G_const)
    T_p = solve_problem_periodic(V, a, L_form)
    T_data_p = T_p.vector().get_local()
    dof_coords = V.tabulate_dof_coordinates().reshape(-1, 2)
    T_array_p = np.zeros((size, size), dtype=np.float64)
    step = L_temp / size
    for i, (xx, yy) in enumerate(dof_coords):
        col = int(round(xx / step))
        row = int(round(yy / step))
        col = min(col, size - 1)
        row = min(row, size - 1)
        T_array_p[row, col] = T_data_p[i]
    return T_array_p

def main():
    logger = setup_logging(os.getcwd())
    L_cond = 50.0
    L_temp = 1.0
    size = 512
    offset = 0.0
    macro_gradient = (1.0, 0.0)
    alpha = 20
    baseline_eps_pairs = [
        (50, r"$\varepsilon = 0.2$"),
        (500, r"$\varepsilon = 0.02$"),
        (5000, r"$\varepsilon = 0.002$"),
    ]
    spectral_data = []
    for baseline, _ in baseline_eps_pairs:
        T_p = run_simulation_periodic(
            size, L_cond, L_temp, alpha, baseline, logger,
            sigma=2.0, k0=0.1, target_std=1.0,
            macro_gradient=macro_gradient, offset=offset
        )
        log_ps = np.log10(compute_power_spectrum_2d(T_p) + 1e-12)
        spectral_data.append(log_ps)
    global_vmin = min(np.min(ps) for ps in spectral_data)
    global_vmax = max(np.max(ps) for ps in spectral_data)
    n_rows = len(baseline_eps_pairs)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 8), constrained_layout=True)

    for row_idx, (baseline, eps_label) in enumerate(baseline_eps_pairs):
        T_p = run_simulation_periodic(
            size, L_cond, L_temp, alpha, baseline, logger,
            sigma=2.0, k0=0.1, target_std=1.0,
            macro_gradient=macro_gradient, offset=offset
        )
        ax_tp = axes[row_idx, 0]
        im_tp = ax_tp.imshow(T_p, origin='lower', cmap='jet', extent=[0, 1, 0, 1])
        ax_tp.set_xlabel("x")
        ax_tp.set_ylabel("y")
        cb = fig.colorbar(im_tp, ax=ax_tp, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=10)
        ax_tp.tick_params(labelsize=10)
        ax_tp.text(-0.35, 0.5, eps_label, fontsize=14, rotation='vertical',
                   transform=ax_tp.transAxes, va='center', ha='center')

        T_mean = np.mean(T_p)
        binary = (T_p - T_mean > 0).astype(float)
        ax_bin = axes[row_idx, 1]
        im_bin = ax_bin.imshow(binary, origin='lower', cmap='gray', extent=[0, 1, 0, 1])
        ax_bin.set_xlabel("x")
        ax_bin.set_ylabel("y")
        cb = fig.colorbar(im_bin, ax=ax_bin, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=10)
        ax_bin.tick_params(labelsize=10)

        log_ps = np.log10(compute_power_spectrum_2d(T_p) + 1e-12)
        ax_ps = axes[row_idx, 2]
        im_ps = ax_ps.imshow(log_ps, origin='lower', cmap='jet',
                             vmin=global_vmin, vmax=global_vmax)
        ax_ps.set_xlabel(r"$k_x$")
        ax_ps.set_ylabel(r"$k_y$")
        cb = fig.colorbar(im_ps, ax=ax_ps, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=10)
        ax_ps.tick_params(labelsize=10)

    out_file = f"composite_eps_vertical_alpha{alpha}.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"Saved clean figure to: {out_file}")

if __name__ == "__main__":
    main()
