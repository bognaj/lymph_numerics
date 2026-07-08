import numpy as np

from collections.abc import Callable

from config import *


class Galerkin:
    def __init__(self, x0: np.array, N: float, t0: float, dt: float, t_final: float, 
                 v: Callable, R_minus: float = 0.1, R_plus: float = 2.0):
         """
         Initialize the Galerkin method for the chemical dynamics system.

         Args:
             x0 (np.array): Initial state vector
             N (float): Number of spatial grid points
             t0 (float): Initial time
             dt (float): Time step
             t_final (float): Final time
             v (Callable): initial condition function
         """
         self.x0 = x0
         self.N = N # number of spatial grid points
         self.h = 1.0 / (N - 1)
         self.t0 = t0
         self.dt = dt
         self.t_final = t_final
         self.v = v
         self.R_minus = R_minus
         self.R_plus = R_plus
         self.n = 0 # count time steps
    
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
    def smooth_heaviside(x: float, x_ref: float, k: float = 10.0) -> float:
        """
        Smooth approximation of the Heaviside step function.

        Args:
            x (float): Input value
            x_ref (float): Reference value for the step
            k (float): Steepness parameter

        Returns:
            float: Value of the smooth Heaviside function at x
        """
        return 1 / (1 + np.exp(-2*k * (x - x_ref)))
    
    def boundary_conditions(self, NC_s: np.array) -> tuple:
        """
        Calculate the boundary conditions R_0 and R_1 based on the current state vector NC_s.

        Args:
            NC_s (np.array): _description_

        Returns:
            tuple: _description_
        """
        R_0 = self.R_minus + (self.R_plus - self.R_minus) * (1 - self.smooth_heaviside(NC_s[1], C_shear))
        R_1 = self.R_minus + (self.R_plus - self.R_minus) * (1 - self.smooth_heaviside(NC_s[1], C_shear))
        return (R_0, R_1)
    
    def extrapolate_nonlinearity(self, n: int, R_history: np.ndarray) -> np.array:
        """
        Extrapolate the nonlinear term based on the history of boundary conditions.

        Args:
            n (int): Current time step
            R_history (np.ndarray): History of boundary conditions

        Returns:
            np.array: Extrapolated nonlinear term
        """
        if n >= 2:
            return 1.5 * R_history[n-1, :] - 0.5 * R_history[n-2, :]
        elif n == 1:
            # TODO
            return R_history[n-1, :]
        elif n == 0:
            return R_history[n, :]

    def basis_function_i(self, i, x) -> float:
        """
        Evaluate the i-th linear hat basis function at x.

        Args:
            i (int): Index of the basis function
            x (float or np.array): Point(s) at which to evaluate

        Returns:
            float: Value of the i-th basis function at x
        """
        xi = i * self.h

        x = np.asarray(x)
        y = np.zeros_like(x, dtype=float)

        # Left side
        left = (x >= xi - self.h) & (x <= xi)
        y[left] = (x[left] - (xi - self.h)) / self.h

        # Right side
        right = (x >= xi) & (x <= xi + self.h)
        y[right] = ((xi + self.h) - x[right]) / self.h

        return y