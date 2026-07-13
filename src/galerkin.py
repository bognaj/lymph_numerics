import numpy as np

from collections.abc import Callable

from config import *


class Galerkin:
    def __init__(self, x0: np.array, xs: np.array, N: float, t0: float, dt: float, t_final: float, 
                 v: Callable, R_minus: float = 0.1, R_plus: float = 2.0):
         """
         Initialize the Galerkin method for the chemical dynamics system.

         Args:
             x0 (np.array): Initial state vector
             xs (np.array): Spatial grid points
             N (float): Number of spatial grid points
             t0 (float): Initial time
             dt (float): Time step
             t_final (float): Final time
             v (Callable): initial condition function
         """
         self.x0 = x0
         self.xs = xs
         self.N = N # number of spatial grid points
         self.h = (xs[-1] - xs[0]) / (N - 1)
         self.t0 = t0
         self.dt = dt
         self.t_final = t_final
         self.v = v
         self.R_minus = R_minus
         self.R_plus = R_plus
         self.n = 0 # count time steps
         self.coeffs_history = np.zeros((int((t_final - t0) / dt) + 1, N))
    
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

    def get_lifting_function_values(self, NC_s: np.array) -> np.array:
        """
        Calculate the lifting function based on the current state vector NC_s.

        Args:
            NC_s (np.array): Current state vector

        Returns:
            np.array: Lifting function
        """
        R_0, R_1 = self.boundary_conditions(NC_s)
        lifting_function = R_0 + self.xs/(self.xs[-1] - self.xs[0]) * (R_1 - R_0)
        return lifting_function
    
    def nonlinearity_a(self, alpha: np.array) -> np.array:
        """
        Define the nonlinearity function a(u) = u^2.

        Returns:
            Callable: Nonlinearity function
        """
        # TODO: Implement the nonlinearity function a(u) in the correct form
        return np.power(alpha, 2)
    
    def extrapolate_u(self, n: int, u_history: np.ndarray) -> np.array:
        """
        Extrapolate the state vector based on its history.

        Args:
            n (int): Current time step
            u_history (np.ndarray): History of state vectors

        Returns:
            np.array: Extrapolated state vector
        """
        if n >= 2:
            return 1.5 * u_history[n-1, :] - 0.5 * u_history[n-2, :]
        elif n == 1:
            # TODO
            return u_history[n-1, :]
        elif n == 0:
            return u_history[n, :]

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
    
    def scalar_product(self, i: int, j: int, h: float) -> float:
        """
        Compute the scalar product of the i-th and j-th basis functions over the domain.

        Args:
            i (int): Index of the first basis function
            j (int): Index of the second basis function
            h (float): Grid spacing
        
        Returns:
            float: Scalar product of the i-th and j-th basis functions
        """
        if i == j:
            return 2 * h / 3
        elif abs(i - j) == 1:
            return h / 6
        else:
            return 0.0
        
    def mass_matrix(self) -> np.array:
        """
        Construct the mass matrix for the Galerkin method.

        Returns:
            np.array: Mass matrix
        """
        M = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    if i == 0 or i == self.N-1:
                        M[i,j] = self.h/3
                    else:
                        M[i,j] = 2*self.h/3
                elif abs(i-j) == 1:
                    M[i,j] = self.h/6
                else:
                    M[i,j] = 0
        return M
    
    def stiffness_matrix(self, n: int, alpha_history: np.ndarray) -> np.array:
        """
        Construct the stiffness matrix for the Galerkin method.

        Args:
            n (int): Current time step
            alpha_history (np.ndarray): History of coefficients multiplying basis functions

        Returns:
            np.array: Stiffness matrix
        """
        K = np.zeros((self.N, self.N))
        extrapolated_alpha = self.extrapolate_u(n, alpha_history)
        linearization = self.nonlinearity_a(extrapolated_alpha)

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    if i == 0 or i == self.N-1:
                        K[i,j] = 1/self.h
                    else:
                        K[i,j] = 2/self.h
                elif abs(i-j) == 1:
                    K[i,j] = -1/self.h
                else:
                    K[i,j] = 0
        return K
    
    def crank_nicolson_galerkin_step(self, n: int, alpha_history: np.ndarray) -> np.ndarray:
        """
        Perform a single Crank-Nicolson step for the Galerkin method.

        Args:
            n (int): Current time step
            alpha_history (np.ndarray): History of coefficients multiplying basis functions

        Returns:
            np.ndarray: Updated state vector
        """
        M = self.mass_matrix()
        K = self.stiffness_matrix(n, alpha_history)

        A = M + (self.dt/2)*K
        B = M - (self.dt/2)*K

        rhs = B @ alpha_history[-1, :]

        alpha_new = np.linalg.solve(A, rhs)

        return alpha_new
    
    def get_u_from_alpha_coeffs(self, alpha_coeffs: np.ndarray) -> np.ndarray:
        """
        Reconstruct the state vector u from the coefficients of the basis functions.

        Args:
            alpha_coeffs (np.ndarray): Coefficients of the basis functions
        
        Returns:
            np.ndarray: Reconstructed state vector u
        """
        u = np.zeros_like(self.xs)
        for i in range(self.N):
            u += alpha_coeffs[i] * self.basis_function_i(i, self.xs)
        return u
