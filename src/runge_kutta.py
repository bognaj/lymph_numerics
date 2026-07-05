import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from chemical_dynamics import F1, F2, get_mode, dynamics


class RungeKutta:
    def __init__(self, x0: np.array, t0: float, dt: float, t_final: float, epsilon: float = 1e-9):
        """
        Initialize the Runge-Kutta integrator for the chemical dynamics system.

        Args:
            x0 (np.array): Initial state vector
            t0 (float): Initial time
            dt (float): Time step
            t_final (float): Final time
            epsilon (float, optional): Small perturbation for switching. Defaults to 1e-9.
        """     
        self.x0 = x0
        self.t0 = t0
        self.dt = dt
        self.t_final = t_final
        self.epsilon = epsilon
        self.mode = "F2" if x0[1] >= 1 else "F1"

    def integrate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate the system using the Runge-Kutta method with event detection for switching.

        Returns:
            tuple[np.ndarray, np.ndarray]: Time points and state vectors.
        """
        mode = get_mode(self.x0)
        t = self.t0
        x = self.x0.copy()

        def event_surface(t, x):
            return x[1] - 1.0
        
        event_surface.terminal = True
        event_surface.direction = 0

        T = [self.t0]
        X = [self.x0.copy()]

        while t < self.t_final:
            sol = solve_ivp(
                lambda t, x: dynamics(t, x, mode),
                (t, self.t_final),
                x,
                events=event_surface,
                max_step=0.1,
                rtol=1e-6,
                atol=1e-9
            )

            # store trajectory
            T.extend(sol.t[1:])
            X.extend(sol.y.T[1:])

            t = sol.t[-1]
            x = sol.y[:, -1]

            if sol.status == 1:
                # boundary event occurred, switch mode
                mode = "F2" if mode == "F1" else "F1"

                # small push to avoid re-triggering event
                if mode == "F1":
                    x[1] = 1.0 - self.epsilon
                else:
                    x[1] = 1.0 + self.epsilon

            else:
                break

        return np.array(T), np.array(X)
    
    def plot_solution(self) -> None:
        """
        Plot the solution trajectory along with the vector field and switching surface.
        """
        t, x = self.integrate()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the solution trajectory
        ax.plot(x[:,0], x[:,1], 'b-', linewidth=2, label='Trajectory')
        ax.plot(x[0,0], x[0,1], 'go', markersize=8, label='Start')
        ax.plot(x[-1,0], x[-1,1], 'r*', markersize=15, label='End')
        
        # Create grid for streamplot
        n_range = np.linspace(x[:,0].min() - 0.5, x[:,0].max() + 0.5, 20)
        c_range = np.linspace(max(0, x[:,1].min() - 0.5), x[:,1].max() + 0.5, 20)
        N, C = np.meshgrid(n_range, c_range)
        
        # Compute vector field
        U = np.zeros_like(N)
        V = np.zeros_like(N)
        
        for i in range(N.shape[0]):
            for j in range(N.shape[1]):
                state = np.array([N[i,j], C[i,j]])
                # Use F1 for C < 1, F2 for C >= 1
                if C[i,j] < 1.0:
                    dxdt = F1(0, state)
                else:
                    dxdt = F2(0, state)
                U[i,j] = dxdt[0]
                V[i,j] = dxdt[1]
        
        # Add streamplot
        ax.streamplot(N, C, U, V, color='gray', density=1.5, arrowsize=1.5, linewidth=0.7)
        
        # Mark the switching boundary
        ax.axhline(1.0, linestyle='--', color='red', linewidth=1.5, alpha=0.7, label='Switching surface (C=1)')
        
        ax.set_xlabel("N", fontsize=12)
        ax.set_ylabel("C", fontsize=12)
        ax.set_title("Solution Trajectory with Vector Field", fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    x0 = np.array([0.5, 0.5])
    t0 = 0.0
    dt = 0.01
    t_final = 100.0

    rk = RungeKutta(x0, t0, dt, t_final)
    rk.plot_solution()
        