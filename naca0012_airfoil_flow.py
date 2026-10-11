"""
================================================================================
Experiment 1: External Flow over a NACA 0012 Airfoil
                    Finite Difference Method (FDM) - teaching code
================================================================================
Single-file, no notebook, no Colab. Only numpy + matplotlib needed.

AIM
    Study how the angle of attack changes the velocity, pressure and wake
    around a NACA 0012 airfoil, and look at qualitative lift/drag trends.

GOVERNING EQUATIONS (incompressible Navier-Stokes, constant properties)
    du/dt + u du/dx + v du/dy = -dp/dx + nu (d2u/dx2 + d2u/dy2)
    dv/dt + u dv/dx + v dv/dy = -dp/dy + nu (d2v/dx2 + d2v/dy2)
    du/dx + dv/dy = 0

FINITE DIFFERENCE IDEA
    The domain is a uniform Cartesian grid. Every variable lives at grid
    POINTS: u[i,j] means u at (x_i, y_j). Derivatives are replaced by
    algebraic difference formulas:

      central difference :  df/dx ~ (f[i+1,j] - f[i-1,j]) / (2 dx)
      upwind difference  :  df/dx ~ (f[i,j] - f[i-1,j]) / dx   if flow u > 0
                                    (f[i+1,j] - f[i,j]) / dx   if flow u < 0
      Laplacian (5-point):  (f[i+1,j] - 2 f[i,j] + f[i-1,j]) / dx^2
                          + (f[i,j+1] - 2 f[i,j] + f[i,j-1]) / dy^2

TIME INTEGRATION - the projection (fractional step) method
    Step A  Predictor : advance u without pressure  ->  u*
    Step B  Poisson   : solve laplacian(phi) = div(u*)/dt  (Jacobi iterations)
    Step C  Corrector : u = u* - dt grad(phi)  ->  divergence-free velocity

AIRFOIL - immersed boundary (no body-fitted mesh)
    A smooth mask chi(x,y) is 1 inside the solid and 0 outside. A penalty
    term -chi*u/eta in the momentum equation forces u -> 0 inside the
    airfoil. Forces are estimated from that penalty term and are
    QUALITATIVE only (teaching level).

HOW TO RUN
    python naca0012_airfoil_fdm.py

OUTPUTS (folder "airfoil_results/")
    airfoil_final.vtk    - pressure, speed, vorticity, mask, velocity (ParaView)
    airfoil_wall.vtk     - airfoil outline
    history.csv          - residual and Cd/Cl history
    airfoil_summary.txt  - run summary
    results_overview.png - convergence, force history, vorticity contour
================================================================================
"""

import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 0. PARAMETERS - students edit only this block
# ==============================================================================
NX, NY = 120, 60          # number of grid POINTS in x and y
RE = 100.0                # Reynolds number  Re = U_inf * chord / nu
ALPHA_DEG = 5.0           # angle of attack in degrees
ITERATIONS = 300          # pseudo-time steps of the projection loop
POISSON_ITERATIONS = 80   # Jacobi sweeps per pressure solve

LX, LY = 5.0, 2.5         # domain size (chord = 1)
U_INF = 1.0               # free-stream speed
CHORD = 1.0               # airfoil chord
X_LE = 1.35               # leading-edge x position

OUTPUT_DIR = Path("airfoil_results")


# ==============================================================================
# 1. NACA 0012 GEOMETRY
# ==============================================================================
def naca0012_half_thickness(x):
    """Half thickness of a NACA 0012 at chordwise position x in [0, 1].

    Returns -1 outside [0, 1] to mark 'no airfoil here'.
    Standard 4-digit NACA thickness formula (closed trailing edge).
    """
    xc = np.clip(x, 0.0, 1.0)
    yt = 5.0 * 0.12 * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2
                       + 0.2843 * xc**3 - 0.1036 * xc**4)
    return np.where((x >= 0.0) & (x <= 1.0), yt, -1.0)


def airfoil_outline(alpha_deg, n=401):
    """(x, y) coordinates of the airfoil surface, rotated by alpha.

    Cosine spacing puts dense points near the rounded leading edge;
    the outline is rotated about the quarter-chord point.
    """
    beta = np.linspace(0.0, math.pi, n)
    xc = 0.5 * (1.0 - np.cos(beta))          # cosine spacing, 0 -> 1
    yt = naca0012_half_thickness(xc)
    xb = np.concatenate((xc[::-1], xc[1:])) * CHORD   # lower then upper side
    yb = np.concatenate((-yt[::-1], yt[1:])) * CHORD

    angle = math.radians(alpha_deg)
    xq, yq = X_LE + 0.25 * CHORD, 0.0        # rotation centre: quarter chord
    xr = xb - 0.25 * CHORD
    x = xq + math.cos(angle) * xr - math.sin(angle) * yb
    y = yq + math.sin(angle) * xr + math.cos(angle) * yb
    return x, y


def airfoil_mask(x, y, alpha_deg, dx, dy):
    """Smooth solid mask chi in [0, 1]: 1 inside the airfoil, 0 in the fluid.

    Each grid point is rotated into the airfoil frame; if its |y| is below
    the local NACA half-thickness it is inside the solid. A smoothstep over
    ~1 grid cell avoids a jagged staircase boundary.
    """
    angle = math.radians(alpha_deg)
    xq = X_LE + 0.25 * CHORD
    # rotate the grid point into the airfoil (body) frame
    xr = math.cos(angle) * (x - xq) + math.sin(angle) * y + 0.25 * CHORD
    yr = -math.sin(angle) * (x - xq) + math.cos(angle) * y
    yt = naca0012_half_thickness(xr / CHORD) * CHORD

    distance = yt - np.abs(yr)               # > 0 means inside the solid
    eps = 0.75 * max(dx, dy)                 # smoothing width ~ 1 cell
    raw = np.clip(0.5 + 0.5 * distance / eps, 0.0, 1.0)
    chi = raw * raw * (3.0 - 2.0 * raw)      # smoothstep
    return np.where(yt >= 0.0, chi, 0.0)


# ==============================================================================
# 2. FINITE DIFFERENCE OPERATORS on a uniform grid
#    (arrays are indexed [j, i]: j = y index, i = x index)
# ==============================================================================
def ddx_central(f, dx):
    """df/dx ~ (f[i+1] - f[i-1]) / (2 dx)   at interior points."""
    out = np.zeros_like(f)
    out[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / (2.0 * dx)
    return out


def ddy_central(f, dy):
    """df/dy ~ (f[j+1] - f[j-1]) / (2 dy)   at interior points."""
    out = np.zeros_like(f)
    out[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2.0 * dy)
    return out


def laplacian(f, dx, dy):
    """d2f/dx2 + d2f/dy2 with the standard 5-point stencil (interior only)."""
    out = np.zeros_like(f)
    out[1:-1, 1:-1] = ((f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dx**2
                       + (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy**2)
    return out


def ddx_upwind(f, u, dx):
    """df/dx with 1st-order upwinding: use the neighbour the flow comes FROM.

        u > 0:  df/dx ~ (f[i] - f[i-1]) / dx     (backward difference)
        u < 0:  df/dx ~ (f[i+1] - f[i]) / dx     (forward difference)
    Upwinding adds numerical damping and keeps advection stable.
    """
    out = np.zeros_like(f)
    backward = (f[1:-1, 1:-1] - f[1:-1, :-2]) / dx
    forward = (f[1:-1, 2:] - f[1:-1, 1:-1]) / dx
    out[1:-1, 1:-1] = np.where(u[1:-1, 1:-1] >= 0.0, backward, forward)
    return out


def ddy_upwind(f, v, dy):
    """df/dy with 1st-order upwinding (same idea, in the y direction)."""
    out = np.zeros_like(f)
    backward = (f[1:-1, 1:-1] - f[:-2, 1:-1]) / dy
    forward = (f[2:, 1:-1] - f[1:-1, 1:-1]) / dy
    out[1:-1, 1:-1] = np.where(v[1:-1, 1:-1] >= 0.0, backward, forward)
    return out


def poisson_jacobi(rhs, dx, dy, sweeps, solid_mask):
    """Solve laplacian(phi) = rhs by Jacobi iteration.

    Rearranging the 5-point Laplacian stencil gives the update formula

        phi[i,j] = ( (phi[i+1,j] + phi[i-1,j]) / dx^2
                   + (phi[i,j+1] + phi[i,j-1]) / dy^2
                   - rhs[i,j] ) / (2/dx^2 + 2/dy^2)

    Boundary condition: zero normal gradient (Neumann) on inlet/top/bottom,
    phi = 0 at the outlet (fixes the pressure level). phi = 0 inside the solid.
    """
    phi = np.zeros_like(rhs)
    inv_dx2, inv_dy2 = 1.0 / dx**2, 1.0 / dy**2
    denom = 2.0 * inv_dx2 + 2.0 * inv_dy2
    for _ in range(sweeps):
        old = phi.copy()
        phi[1:-1, 1:-1] = ((old[1:-1, 2:] + old[1:-1, :-2]) * inv_dx2
                           + (old[2:, 1:-1] + old[:-2, 1:-1]) * inv_dy2
                           - rhs[1:-1, 1:-1]) / denom
        phi[solid_mask > 0.5] = 0.0
        phi[:, 0] = phi[:, 1]      # inlet:  dphi/dx = 0
        phi[:, -1] = 0.0           # outlet: phi = 0
        phi[0, :] = phi[1, :]      # bottom: dphi/dy = 0
        phi[-1, :] = phi[-2, :]    # top:    dphi/dy = 0
    return phi


# ==============================================================================
# 3. BOUNDARY CONDITIONS
# ==============================================================================
def apply_bc(u, v, p, speed):
    """Uniform inflow, zero-gradient outlet, free-stream top/bottom."""
    u[:, 0], v[:, 0] = speed, 0.0            # inlet: uniform stream
    u[:, -1], v[:, -1] = u[:, -2], v[:, -2]  # outlet: du/dx = 0, dv/dx = 0
    p[:, -1], p[:, 0] = 0.0, p[:, 1]         # p = 0 outlet, dp/dx = 0 inlet
    u[0, :], u[-1, :] = speed, speed         # top/bottom: free stream
    v[0, :], v[-1, :] = 0.0, 0.0             #             (free slip)
    p[0, :], p[-1, :] = p[1, :], p[-2, :]    #             dp/dy = 0


# ==============================================================================
# 4. VTK OUTPUT (structured grid, readable by ParaView / VisIt)
# ==============================================================================
def write_structured_grid_vtk(path, x, y, scalars, vectors, title):
    ny, nx = x.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(title + "\nASCII\nDATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write(f"POINTS {nx * ny} float\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{x[j, i]:.12e} {y[j, i]:.12e} 0\n")
        f.write(f"POINT_DATA {nx * ny}\n")
        for name, values in scalars.items():
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            for value in values.ravel():
                f.write(f"{value:.12e}\n")
        for name, (vx, vy) in vectors.items():
            f.write(f"VECTORS {name} float\n")
            for ax, ay in zip(vx.ravel(), vy.ravel()):
                f.write(f"{ax:.12e} {ay:.12e} 0\n")


def write_polyline_vtk(path, x, y, title):
    """Airfoil outline as a closed polyline."""
    ids = list(range(x.size)) + [0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(title + "\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {x.size} float\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.12e} {yi:.12e} 0\n")
        f.write(f"LINES 1 {len(ids) + 1}\n")
        f.write(f"{len(ids)} " + " ".join(str(i) for i in ids) + "\n")


# ==============================================================================
# 5. MAIN SOLVER
# ==============================================================================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---- uniform Cartesian grid -------------------------------------------
    x1 = np.linspace(0.0, LX, NX)
    y1 = np.linspace(-0.5 * LY, 0.5 * LY, NY)
    x, y = np.meshgrid(x1, y1)           # x[j,i], y[j,i] = coordinates of point
    dx, dy = x1[1] - x1[0], y1[1] - y1[0]
    nu = U_INF * CHORD / RE              # kinematic viscosity from Re

    # ---- airfoil immersed-boundary mask ------------------------------------
    chi = airfoil_mask(x, y, ALPHA_DEG, dx, dy)

    # ---- initial condition: uniform stream, u = 0 inside the solid ---------
    u = U_INF * (1.0 - chi)
    v = np.zeros_like(u)
    p = np.zeros_like(u)

    # ---- stable time step (explicit advective + diffusive limits) ----------
    dt = min(0.10 * min(dx, dy) / U_INF, 0.18 * min(dx * dx, dy * dy) / nu)
    eta = max(0.5 * dt, 1e-5)            # penalization parameter
    apply_bc(u, v, p, U_INF)

    print(f"FDM grid {NX}x{NY}, Re = {RE}, alpha = {ALPHA_DEG} deg, dt = {dt:.3e}")
    print("-" * 78)

    history = []                         # rows: iter, time, res, umax, Cd, Cl
    for step in range(1, ITERATIONS + 1):

        # -- Step A: predictor ----------------------------------------------
        # u* = u + dt ( -u du/dx - v du/dy          (upwind advection)
        #               + nu (d2u/dx2 + d2u/dy2)    (5-point diffusion)
        #               - chi u / eta )             (immersed-boundary penalty)
        us = u + dt * (-(u * ddx_upwind(u, u, dx) + v * ddy_upwind(u, v, dy))
                       + nu * laplacian(u, dx, dy) - chi * u / eta)
        vs = v + dt * (-(u * ddx_upwind(v, u, dx) + v * ddy_upwind(v, v, dy))
                       + nu * laplacian(v, dx, dy) - chi * v / eta)
        apply_bc(us, vs, p, U_INF)

        # -- Step B: pressure Poisson equation --------------------------------
        # laplacian(phi) = div(u*) / dt   (central differences for div(u*))
        rhs = (ddx_central(us, dx) + ddy_central(vs, dy)) / dt
        phi = poisson_jacobi(rhs, dx, dy, POISSON_ITERATIONS, chi)

        # -- Step C: corrector ------------------------------------------------
        # u = u* - dt dphi/dx ,  v = v* - dt dphi/dy  (central differences)
        u = (us - dt * ddx_central(phi, dx)) * (1.0 - chi)
        v = (vs - dt * ddy_central(phi, dy)) * (1.0 - chi)
        p += 0.25 * phi                    # under-relaxed pressure update
        apply_bc(u, v, p, U_INF)

        # -- diagnostics ------------------------------------------------------
        # continuity residual: L2 norm of div(u) = du/dx + dv/dy over fluid
        div = ddx_central(u, dx) + ddy_central(v, dy)
        fluid = chi < 0.5
        continuity_l2 = float(np.sqrt(np.mean(div[fluid]**2)))

        # penalty reaction forces, rotated into wind axes -> Cd, Cl
        force_x = float(np.sum(chi * u / eta) * dx * dy)
        force_y = float(np.sum(chi * v / eta) * dx * dy)
        ang = math.radians(ALPHA_DEG)
        drag = force_x * math.cos(ang) + force_y * math.sin(ang)
        lift = -force_x * math.sin(ang) + force_y * math.cos(ang)
        qref = 0.5 * U_INF**2 * CHORD
        cd, cl = drag / qref, lift / qref
        umax = float(np.max(np.sqrt(u * u + v * v)))
        history.append([step, step * dt, continuity_l2, umax, cd, cl])

        # -- safety checks ----------------------------------------------------
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            raise FloatingPointError(f"NaN detected at iteration {step}")
        if umax > 25.0:
            raise FloatingPointError(f"solution diverged at iteration {step}")

        if step == 1 or step == ITERATIONS or step % max(1, ITERATIONS // 5) == 0:
            print(f"iter {step:5d} | continuity L2 = {continuity_l2:.4e} | "
                  f"max speed = {umax:.3f} | Cd = {cd:+.4f} | Cl = {cl:+.4f}")

    # ---- derived fields -----------------------------------------------------
    speed_field = np.sqrt(u * u + v * v)
    # vorticity omega_z = dv/dx - du/dy  (central differences)
    vorticity = ddx_central(v, dx) - ddy_central(u, dy)

    # ---- write files --------------------------------------------------------
    write_structured_grid_vtk(
        OUTPUT_DIR / "airfoil_final.vtk", x, y,
        {"pressure": p, "speed": speed_field, "vorticity": vorticity,
         "solid_mask": chi},
        {"velocity": (u, v)}, "NACA 0012 immersed-boundary flow (FDM)")
    wall_x, wall_y = airfoil_outline(ALPHA_DEG)
    write_polyline_vtk(OUTPUT_DIR / "airfoil_wall.vtk", wall_x, wall_y,
                       "NACA 0012 wall")

    with open(OUTPUT_DIR / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "pseudo_time", "continuity_l2",
                         "max_speed", "Cd_qualitative", "Cl_qualitative"])
        writer.writerows(history)

    with open(OUTPUT_DIR / "airfoil_summary.txt", "w", encoding="utf-8") as f:
        f.write("Teaching-level FDM immersed-boundary projection solution\n")
        f.write(f"nx {NX}\nny {NY}\nRe {RE}\nalpha_deg {ALPHA_DEG}\n")
        f.write(f"dt {dt:.6e}\niterations {ITERATIONS}\n")
        f.write(f"poisson_iterations {POISSON_ITERATIONS}\n")
        f.write(f"final_continuity_l2 {history[-1][2]:.6e}\n")
        f.write("penalty_force_coefficients_are_qualitative_only true\n")

    print("-" * 78)
    print(f"Final continuity L2 = {history[-1][2]:.4e}")
    print(f"Files written to '{OUTPUT_DIR}/'")

    # ---- plots ---------------------------------------------------------------
    history = np.array(history)
    plot_results(x, y, vorticity, wall_x, wall_y, history)


# ==============================================================================
# 6. PLOTTING
# ==============================================================================
def plot_results(x, y, vorticity, wall_x, wall_y, history):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # (a) convergence history
    axes[0].semilogy(history[:, 0], history[:, 2], "b-")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Continuity L2 residual")
    axes[0].set_title("Convergence")
    axes[0].grid(True)

    # (b) qualitative force history
    axes[1].plot(history[:, 0], history[:, 4], "r-", label="Cd (qualitative)")
    axes[1].plot(history[:, 0], history[:, 5], "g-", label="Cl (qualitative)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Coefficient")
    axes[1].set_title("Penalty force history")
    axes[1].legend()
    axes[1].grid(True)

    # (c) vorticity field with the airfoil outline
    image = axes[2].contourf(x, y, vorticity, levels=30, cmap="RdBu_r")
    axes[2].plot(wall_x, wall_y, "k-", linewidth=1.5)
    fig.colorbar(image, ax=axes[2], label="vorticity")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title(f"Vorticity, alpha = {ALPHA_DEG} deg, Re = {RE:g}")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlim(0.5, 4.0)
    axes[2].set_ylim(-1.0, 1.0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "results_overview.png", dpi=150)
    print(f"Plot saved to '{OUTPUT_DIR}/results_overview.png'")
    plt.show()


if __name__ == "__main__":
    main()
