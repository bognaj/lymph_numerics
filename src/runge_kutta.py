import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from config import *
from chemical_dynamics import F1, F2, get_mode, dynamics


class RungeKutta:
    def __init__(self, x0: np.array, t0: float, dt: float, t_final: float, xs: np.array, epsilon: float = 1e-9):
        """
        Initialize the Runge-Kutta integrator for the chemical dynamics system.

        Args:
            x0 (np.array): Initial state vector
            t0 (float): Initial time
            dt (float): Time step
            t_final (float): Final time
            xs (np.array): spatial grid points for the system (coming from Galerkin method)
            epsilon (float, optional): Small perturbation for switching. Defaults to 1e-9.
        """     
        self.x0 = x0
        self.t0 = t0
        self.dt = dt
        self.t_final = t_final
        self.xs = xs
        self.epsilon = epsilon
        self.mode = "F2" if x0[1] >= 1 else "F1"

    @staticmethod
    def u_to_r(u: np.array) -> np.array:
        """
        Convert the state vector u to the corresponding r vector.

        Args:
            u (np.array): State vector

        Returns:
            np.array: Corresponding r vector
        """
        r = np.sqrt(2*u)
        return r
    
    @staticmethod
    def r_to_u(r: np.array) -> np.array:
        """
        Convert the r vector to the corresponding state vector u.

        Args:
            r (np.array): r vector

        Returns:
            np.array: Corresponding state vector u
        """
        u = 0.5 * r**2
        return u

    @staticmethod
    def phi_prim(R_vector: np.array, gamma: float=20, lambda_: float=0.5, alpha: float=0.1, beta: float=0.04) -> np.array:
        """
        Compute the derivative of the phi function with respect to R.

        Args:
            R_vector (np.array): Vector of radii
            gamma (float, optional): gamma parameter. Defaults to 20.
            lambda_ (float, optional): lambda parameter. Defaults to 0.5.
            alpha (float, optional): alpha parameter. Defaults to 0.1.
            beta (float, optional): beta parameter. Defaults to 0.04.

        Returns:
            np.array: Derivative of the phi function with respect to R
        """
        return gamma/(R0 - lambda_ * R_vector) - alpha * R_vector**(-3) + beta
    
    @staticmethod
    def R_x():
        return 1
    
    def p_x(self, R_vector: np.array) -> np.array:
        """
        Calculates the pressure derivative with respect to x 
        based on the radius R and eta parameter.

        Args:
            R_vector (np.array): Vector of radii

        Returns:
            np.array: Values of p_x
        """
        return eta * self.phi_prim(R_vector) * self.R_x()

    def get_shear_stress(self, R_vector: np.array) -> np.array:
        """
        Calculate the shear stress based on the radius R and pressure derivative p_x.        

        Args:
            R_vector (np.array): Vector of radii

        Returns:
            np.array: Shear stress values
        """
        return (R_vector * self.p_x(R_vector))/(2 * L)

    @staticmethod
    def get_alpha() -> float:
        """
        Calculates the value of alpha parameter.
        Returns:
            float: Alpha value
        """
        return k_Ca_minus/k_NO_minus
    
    @staticmethod
    def get_beta(R: float) -> float:
        """
        Calculates the value of beta parameter based on the averaged radius R.
        Returns:
            float: Beta value
        """
        return (k_Ca_plus * (R/R_crit)**11)/(k_NO_minus * C_shear)
    
    @staticmethod
    def get_gamma() -> float:
        """
        Calculates the value of gamma parameter.

        Returns:
            float: Gamma value
        """
        return (10 * k_Ca_plus)/(k_NO_minus * C_shear)
    
    @staticmethod
    def get_zeta(tau: float) -> float:
        """
        Calculates the value of zeta parameter based on the averagedshear stress tau.

        Args:
            tau (float): averaged shear stress

        Returns:
            float: Zeta value
        """
        return (k_NO_plus * m * tau/tau_ref)/(k_NO_minus)

    def single_step(self, t: float, x: np.array) -> np.array:
        """
        Performs a single Runge-Kutta step for the current mode.
        If the state crosses the switching surface, it switches to the other mode.
        Utilizes event detection and solve_ivp to handle the switching.

        Args:
            t (float): Current time
            x (np.array): Current state vector

        Returns:
            np.array: New state vector after the time step
        """
        def event_surface(t, x):
            return x[1] - 1.0
        
        event_surface.terminal = True
        event_surface.direction = 0

        sol = solve_ivp(
            lambda t, x: dynamics(t, x, self.mode),
            (t, t + self.dt),
            x,
            events=event_surface,
            max_step=self.dt,
            rtol=1e-6,
            atol=1e-9
        )

        if sol.status == 1:
            # boundary event occurred, switch mode
            self.mode = "F2" if self.mode == "F1" else "F1"

            # small push to avoid re-triggering event
            if self.mode == "F1":
                sol.y[1, -1] = 1.0 - self.epsilon
            else:
                sol.y[1, -1] = 1.0 + self.epsilon

        return sol.y[:, -1]

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
    xs = np.linspace(0, 1, 100)

    rk = RungeKutta(x0, t0, dt, t_final, xs)
    rk.plot_solution()
        