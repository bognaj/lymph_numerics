import numpy as np

from collections.abc import Callable

from config import *


class Galerkin:
    def __init__(self,  xs: np.array, N: float, t0: float, dt: float, t_final: float, 
                 v: Callable):
         """
         Initialize the Galerkin method for the chemical dynamics system.

         Args:
             xs (np.array): Spatial grid points
             N (float): Number of spatial grid points
             t0 (float): Initial time
             dt (float): Time step
             t_final (float): Final time
             v (Callable): initial condition function
         """
         self.xs = xs
         self.N = N # number of spatial grid points
         self.num_dofs = N - 2
         self.h = (xs[-1] - xs[0]) / (N - 1)
         self.t0 = t0
         self.dt = dt
         self.t_final = t_final
         self.v = v
         self.R_initial_0 = v(0)
         self.R_initial_1 = v(1)
         self.u_initial_0 = 0.5 * np.power(self.R_initial_0, 2)
         self.u_initial_1 = 0.5 * np.power(self.R_initial_1, 2)
         self.R_minus = 0.05
         self.u_minus = 0.5 * np.power(self.R_minus, 2)
         self.R_plus = 1
         self.u_plus = 0.5 * np.power(self.R_plus, 2)
         self.n = 0 # count time steps
         self.alpha_history = np.zeros((int((t_final - t0) / dt) + 1, 
                                        self.num_dofs))
    
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
    
    def boundary_conditions(self, C_values: np.array) -> tuple:
        """
        Calculate the boundary conditions u_0 and u_1 based on the current state vector C_values.

        Args:
            C_values (np.array): _description_

        Returns:
            tuple: _description_
        """
        u_0 = self.u_minus + (self.u_plus - self.u_minus) * (1 - self.smooth_heaviside(C_values[self.n], C_shear))
        u_1 = self.u_minus + (self.u_plus - self.u_minus) * (1 - self.smooth_heaviside(C_values[self.n], C_shear))
        return (u_0, u_1)

    def get_lifting_function_values(self, u_0: float, u_1: float) -> np.array:
        """
        Calculate the lifting function based on the current state vector NC_s.

        Args:
            u_0 (float): Value of u at the left boundary
            u_1 (float): Value of u at the right boundary

        Returns:
            np.array: Lifting function
        """
        lifting_function = u_0 + self.xs/(self.xs[-1] - self.xs[0]) * (u_1 - u_0)
        return lifting_function
    
    @staticmethod
    def phi_prim(u_vector: np.array, gamma: float=20, lambda_: float=0.5, alpha: float=0.1, beta: float=0.04) -> np.array:
        """
        Compute the derivative of the phi function with respect to R.

        Args:
            u_vector (np.array): Vector of state variables
            gamma (float, optional): gamma parameter. Defaults to 20.
            lambda_ (float, optional): lambda parameter. Defaults to 0.5.
            alpha (float, optional): alpha parameter. Defaults to 0.1.
            beta (float, optional): beta parameter. Defaults to 0.04.

        Returns:
            np.array: Derivative of the phi function with respect to u
        """
        return gamma/(R0 - lambda_ * u_vector) - alpha * u_vector**(-3) + beta
    
    def nonlinearity_a(self, u_vector: np.array, eta: float) -> np.array:
        """
        Define the nonlinearity function a(u) = u^2.

        Returns:
            np.array: Nonlinearity function values
        """
        return np.sqrt(2)/8 * eta * self.phi_prim(np.sqrt(2 *u_vector)) * np.power(u_vector, 1.5)
    
    def extrapolate_u(self, u_history: np.ndarray) -> np.array:
        """
        Extrapolate the state vector based on its history.

        Args:
            u_history (np.ndarray): History of state vectors

        Returns:
            np.array: Extrapolated state vector
        """
        if self.n >= 2:
            return 1.5 * u_history[self.n-1, :] - 0.5 * u_history[self.n-2, :]
        elif self.n == 1:
            raise ValueError(
            "First step must be computed using first_step()."
        )
        elif self.n == 0:
            return u_history[self.n, :]

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
        M = np.zeros((self.num_dofs, self.num_dofs))
        for i in range(self.num_dofs):
            for j in range(self.num_dofs):
                if i == j:
                    M[i,j] = 2*self.h/3
                elif abs(i-j) == 1:
                    M[i,j] = self.h/6
                else:
                    M[i,j] = 0
        return M
    
    def stiffness_matrix(self, u_star: np.ndarray) -> np.ndarray:
        """
        Assemble nonlinear stiffness matrix.

        Only interior degrees of freedom are assembled.
        Boundary nodes are eliminated.

        Parameters
        ----------
        u_star : np.ndarray
            Extrapolated interior solution.
            Length = N-2

        Returns
        -------
        K : np.ndarray
            Reduced stiffness matrix.
        """

        K = np.zeros(
            (self.num_dofs, self.num_dofs)
        )

        # Loop over elements
        for e in range(self.N - 1):
            node_left = e
            node_right = e + 1

            if node_left == 0:
                u_left = 0.0
            else:
                u_left = u_star[node_left - 1]

            if node_right == self.N - 1:
                u_right = 0.0
            else:
                u_right = u_star[node_right - 1]

            # Element midpoint value
            u_mid = 0.5 * (u_left + u_right)

            # Nonlinear coefficient
            a_e = self.nonlinearity_a(
                u_mid,
                eta
            )

            # Local stiffness matrix
            K_local = (
                a_e / self.h
            ) * np.array([
                [ 1, -1],
                [-1,  1]
            ])

            # Assembly
            local_nodes = [
                node_left,
                node_right
            ]

            for i_local, i_node in enumerate(local_nodes):
                # Ignore boundary nodes
                if i_node == 0 or i_node == self.N-1:
                    continue
                i_dof = i_node - 1
                for j_local, j_node in enumerate(local_nodes):
                    if j_node == 0 or j_node == self.N-1:
                        continue
                    j_dof = j_node - 1
                    K[i_dof, j_dof] += (
                        K_local[i_local, j_local]
                    )
        return K
    
    def first_step(self, u_history: np.ndarray) -> np.ndarray:
        """
        Compute the first time step using the predictor-corrector
        procedure described by equations (13.26)-(13.27).
        Predictor:
            U1,0 from U0
        Corrector:
            U1 using midpoint (U1,0 + U0)/2

        Parameters
        ----------
        u_history : np.ndarray
            History array containing U0.

        Returns
        -------
        U1 : np.ndarray
            First time step solution.
        """
        U0 = u_history[0, :]
        # -----------------------------
        # Step 1: predictor U1,0
        # -----------------------------
        K0 = self.stiffness_matrix(
            U0
        )
        M = self.mass_matrix()
        A0 = (
            M
            +
            (self.dt/2)*K0
        )
        B0 = (
            M
            -
            (self.dt/2)*K0
        )
        rhs0 = B0 @ U0

        U10 = np.linalg.solve(
            A0,
            rhs0
        )

        # -----------------------------
        # Step 2: corrector
        # -----------------------------
        U_mid = 0.5*(U10 + U0)

        K1 = self.stiffness_matrix(
            U_mid
        )

        A1 = (
            M + (self.dt/2)*K1
        )

        B1 = (
            M - (self.dt/2)*K1
        )
        rhs1 = B1 @ U0

        U1 = np.linalg.solve(
            A1,
            rhs1
        )

        return U1
    
    def crank_nicolson_galerkin_step(self, n: int, u_history: np.ndarray) -> np.ndarray:
        """
        Perform a single Crank-Nicolson step for the Galerkin method.

        Args:
            n (int): Current time step
            u_history (np.ndarray): History of u

        Returns:
            np.ndarray: Updated state vector
        """
        if n == 1:
            return self.first_step(
                u_history
            )
        
        u_star = self.extrapolate_u(u_history)

        M = self.mass_matrix()
        K = self.stiffness_matrix(u_star)

        A = M + (self.dt/2)*K
        B = M - (self.dt/2)*K

        rhs = B @ u_history[-1, :]

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
        for local in range(self.num_dofs):
            physical = local + 1

            u += alpha_coeffs[local] * self.basis_function_i(
                    physical,
                    self.xs
                )
        return u
