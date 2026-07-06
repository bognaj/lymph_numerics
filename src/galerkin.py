import numpy as np

from collections.abc import Callable


class Galerkin:
    def __init__(self, x0: np.array, N: float, t0: float, dt: float, t_final: float, v: Callable):
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
         self.N = N
         self.h = 1.0 / (N - 1)
         self.t0 = t0
         self.dt = dt
         self.t_final = t_final
         self.v = v
    
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

    def phi(self, i, x) -> float:
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