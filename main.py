import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from runge_kutta import RungeKutta
from galerkin import Galerkin


if __name__ == "__main__":
    # initial values of chemical dynamics
    C0 = 0.5
    N0 = 0.5
    x0 = np.array([N0, C0])

    # time settings
    t0 = 0.0
    dt = 0.01
    t_final = 10.0
    current_time = t0 + dt # the first moment in time we need to find values for
    current_n = 1 # the first step we need to find the new values for

    # space settings
    xs = np.linspace(0, 1, 100)

    # initial function
    v = lambda x: 0.1 + 0.9 * np.sin(np.pi * x)

    # initialize solvers
    rk = RungeKutta(x0=x0, t0=t0, dt=dt, 
                        t_final=t_final, xs=xs)
    
    galerkin = Galerkin(xs=xs, N=len(xs), t0=t0, dt=dt, 
                        t_final=t_final, initial_condition=v)

    # Store chemical substances concentrations
    N_values = np.zeros(int((t_final - t0) / dt) + 1)
    N_values[0] = N0

    C_values = np.zeros(int((t_final - t0) / dt) + 1)
    C_values[0] = C0

    # Store values of u
    u_values = np.zeros((int((t_final - t0) / dt) + 1,
                         len(xs)))
    
    u_values[0, :] = v(xs)

    # Store values of lifting functions
    lifting_function_values = np.zeros((int((t_final - t0) / dt) + 1,
                                        len(xs)))

    # Compute initial lifting function from initial BCs (using C0)
    galerkin.n = 0
    bc_0_init, bc_1_init = galerkin.boundary_conditions(C_values)
    lifting_function_values[0, :] = galerkin.get_lifting_function_values(bc_0_init, bc_1_init)

    # Initial alpha coefficients: v^0 = u^0 - U^0 at interior nodes
    galerkin.alpha_history[0, :] = u_values[0, 1:-1] - lifting_function_values[0, 1:-1]

    while current_time <= t_final:
        # Advance chemical concentrations from step n-1 to step n
        N_new, C_new = rk.single_step(
            t=current_time - dt,
            NC_s=np.array([N_values[current_n - 1], C_values[current_n - 1]]),
            R_vector=rk.u_to_r(u_values[current_n - 1, :])
        )
        N_values[current_n] = N_new
        C_values[current_n] = C_new

        # Compute new boundary conditions from updated C at step n
        galerkin.n = current_n
        bc_0, bc_1 = galerkin.boundary_conditions(C_values=C_values)

        # Lifting function at step n
        lifting_next = galerkin.get_lifting_function_values(bc_0, bc_1)
        lifting_function_values[current_n, :] = lifting_next
        lifting_prev = lifting_function_values[current_n - 1, :]

        # Galerkin step: advance interior alpha coefficients from n-1 to n
        alpha_new = galerkin.crank_nicolson_galerkin_step(
            n=current_n,
            u_history=galerkin.alpha_history[:current_n, :],
            lifting_prev=lifting_prev,
            lifting_next=lifting_next
        )
        galerkin.alpha_history[current_n, :] = alpha_new

        # Reconstruct full solution: u = U + v_h
        u_values[current_n, :] = lifting_next + galerkin.get_u_from_alpha_coeffs(alpha_new)

        current_time += dt
        current_n += 1

    # ------------------------------------------------------------------
    # Post-processing: reconstruct the radius R = sqrt(2 u) and plot
    # ------------------------------------------------------------------
    times = t0 + dt * np.arange(len(N_values))
    R_values = rk.u_to_r(u_values)  # shape (n_steps, n_space)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Chemical concentrations N and C over time
    ax = axes[0]
    ax.plot(times, N_values, label="N (NO)", color="tab:blue")
    ax.plot(times, C_values, label="C (Ca)", color="tab:red")
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1,
               alpha=0.7, label="switching surface C = 1")
    ax.set_xlabel("time")
    ax.set_ylabel("concentration")
    ax.set_title("Chemical dynamics N(t), C(t)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # (2) Radius profiles R(x) at several time snapshots
    ax = axes[1]
    snapshot_idx = np.linspace(0, len(times) - 1, 6, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_idx)))
    for color, idx in zip(colors, snapshot_idx):
        ax.plot(xs, R_values[idx, :], color=color,
                label=f"t = {times[idx]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("R(x, t)")
    ax.set_title("Radius profiles R(x) at snapshots")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3) Space-time evolution of the radius R(x, t)
    ax = axes[2]
    mesh = ax.pcolormesh(xs, times, R_values, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="R(x, t)")
    ax.set_xlabel("x")
    ax.set_ylabel("time")
    ax.set_title("Radius evolution R(x, t)")

    fig.tight_layout()
    plt.show()