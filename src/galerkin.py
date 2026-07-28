import numpy as np

from collections.abc import Callable

from config import *


class Galerkin:
    def __init__(self,  xs: np.array, N: float, t0: float, dt: float, t_final: float, 
                 initial_condition: Callable):
         """
         Initialize the Galerkin method for the chemical dynamics system.

         Args:
             xs (np.array): Spatial grid points
             N (float): Number of spatial grid points
             t0 (float): Initial time
             dt (float): Time step
             t_final (float): Final time
             initial_condition (Callable): initial condition u(., 0) = w as a
                 function of x, returning nodal values of u
         """
         self.xs = xs
         self.N = N # number of spatial grid points
         self.num_dofs = N - 2
         self.h = (xs[-1] - xs[0]) / (N - 1)
         self.t0 = t0
         self.dt = dt
         self.t_final = t_final
         self.initial_condition = initial_condition
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
        return 1 / (1 + np.exp(-k * (x - x_ref)))
    
    def boundary_conditions(self, C_values: np.array) -> tuple:
        """
        Calculate the boundary conditions u_0 and u_1 based on the current state vector C_values.

        Args:
            C_values (np.array): _description_

        Returns:
            tuple: _description_
        """
        H = 1 - self.smooth_heaviside(C_values[self.n], C_shear)
        u_b = (np.sqrt(self.u_minus) + np.sqrt(self.u_plus - self.u_minus) * H) ** 2
        return (u_b, u_b)

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

    def _full_nodes_from_alpha(self, alpha: np.ndarray, lifting: np.ndarray) -> np.ndarray:
        """
        Reconstruct full nodal solution u = U + v at all N nodes.

        The lifting function U satisfies the boundary conditions; the interior
        DOF coefficients alpha represent v = u - U (zero Dirichlet BCs).

        Args:
            alpha (np.ndarray): Interior DOF coefficients (length N-2)
            lifting (np.ndarray): Lifting function values at all nodes (length N)

        Returns:
            np.ndarray: Full solution values u = U + v at all nodes (length N)
        """
        u_full = lifting.copy()
        u_full[1:-1] += alpha
        return u_full

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
    
    def stiffness_matrix(self, u_full_nodes: np.ndarray) -> np.ndarray:
        """
        Assemble nonlinear stiffness matrix.

        Only interior degrees of freedom are assembled.
        Boundary nodes are eliminated.

        Parameters
        ----------
        u_full_nodes : np.ndarray
            Full solution U + v at all N nodes (including boundaries).
            Length = N

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

            # Full solution values at element nodes (lifting + interior DOFs)
            u_left = u_full_nodes[node_left]
            u_right = u_full_nodes[node_right]

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

    def _lifting_stiffness_vector(self, u_full_nodes: np.ndarray,
                                  u_bar_nodes: np.ndarray) -> np.ndarray:
        """
        Interior-row action of the nonlinear stiffness operator on the lifting
        function: (K_full @ U_bar) restricted to interior nodes.

        This assembles the term int a(U+v) U_x phi_j' dx that appears on the
        right-hand side of the lifted weak form
            int v_t phi_j + int a(U+v) v_x phi_j'
                = -int U_t phi_j - int a(U+v) U_x phi_j'.
        It vanishes for symmetric boundary conditions (U constant in x, so
        U_x = 0) but is required whenever u_B^- != u_B^+.

        Args:
            u_full_nodes (np.ndarray): Full solution U + v used to freeze the
                nonlinear coefficient a, values at all N nodes.
            u_bar_nodes (np.ndarray): Lifting function values (all N nodes) to
                which the stiffness operator is applied.

        Returns:
            np.ndarray: Interior-row stiffness action, length num_dofs.
        """
        result = np.zeros(self.num_dofs)
        for e in range(self.N - 1):
            node_left = e
            node_right = e + 1

            u_mid = 0.5 * (u_full_nodes[node_left] + u_full_nodes[node_right])
            a_e = self.nonlinearity_a(u_mid, eta)

            # Local stiffness (a_e/h) * [[1, -1], [-1, 1]] applied to the local
            # lifting values [U_left, U_right].
            local = (a_e / self.h) * np.array([
                u_bar_nodes[node_left] - u_bar_nodes[node_right],
                u_bar_nodes[node_right] - u_bar_nodes[node_left],
            ])

            for i_local, i_node in enumerate([node_left, node_right]):
                if i_node == 0 or i_node == self.N - 1:
                    continue
                result[i_node - 1] += local[i_local]
        return result

    def first_step(self, u_history: np.ndarray, lifting_prev: np.ndarray, lifting_next: np.ndarray) -> np.ndarray:
        """
        Compute the first time step using the predictor-corrector
        procedure described by equations (13.26)-(13.27).
        Predictor:
            alpha_1,0 from alpha_0
        Corrector:
            alpha_1 using midpoint (alpha_1,0 + alpha_0)/2

        Parameters
        ----------
        u_history : np.ndarray
            History array containing alpha coefficients at t=0.
        lifting_prev : np.ndarray
            Lifting function U at t=0, values at all N nodes.
        lifting_next : np.ndarray
            Lifting function U at t=dt, values at all N nodes.

        Returns
        -------
        alpha_1 : np.ndarray
            First time step interior DOF coefficients.
        """
        alpha_0 = u_history[0, :]
        M = self.mass_matrix()

        # Lifting function averaged over the step, used for the U_x forcing.
        u_bar = 0.5 * (lifting_prev + lifting_next)

        # Forcing from -U_t: exact integral of -U_t against hat basis phi_j is
        # -h * U_t(x_j). Approximating U_t via backward difference gives
        # -h*(U^1 - U^0)/dt per node. The dt factor cancels with the
        # Crank-Nicolson dt multiplication, leaving:
        f_ut = -self.h * (lifting_next[1:-1] - lifting_prev[1:-1])

        # -----------------------------
        # Step 1: predictor alpha_1,0
        # -----------------------------
        u_full_0 = self._full_nodes_from_alpha(alpha_0, lifting_prev)
        K0 = self.stiffness_matrix(u_full_0)
        A0 = M + (self.dt/2)*K0
        B0 = M - (self.dt/2)*K0
        # Add the lifting-gradient forcing -dt * (K_full @ U_bar); nonzero only
        # when the boundary conditions are asymmetric (U_x != 0).
        f0 = f_ut - self.dt * self._lifting_stiffness_vector(u_full_0, u_bar)
        rhs0 = B0 @ alpha_0 + f0

        alpha_10 = np.linalg.solve(A0, rhs0)

        # -----------------------------
        # Step 2: corrector
        # -----------------------------
        alpha_mid = 0.5*(alpha_10 + alpha_0)
        lifting_mid = 0.5*(lifting_prev + lifting_next)
        u_full_mid = self._full_nodes_from_alpha(alpha_mid, lifting_mid)

        K1 = self.stiffness_matrix(u_full_mid)
        A1 = M + (self.dt/2)*K1
        B1 = M - (self.dt/2)*K1
        f1 = f_ut - self.dt * self._lifting_stiffness_vector(u_full_mid, u_bar)
        rhs1 = B1 @ alpha_0 + f1

        alpha_1 = np.linalg.solve(A1, rhs1)

        return alpha_1
    
    def crank_nicolson_galerkin_step(self, n: int, u_history: np.ndarray,
                                      lifting_prev: np.ndarray, lifting_next: np.ndarray) -> np.ndarray:
        """
        Perform a single Crank-Nicolson step for the Galerkin method.

        The equation solved is v_t - (a(U+v)(U+v)_x)_x = -U_t, where U is the
        lifting function and v = u - U satisfies zero Dirichlet boundary conditions.

        Args:
            n (int): Current time step index (1-indexed; step being computed)
            u_history (np.ndarray): Interior alpha coefficients at steps 0..n-1,
                                    shape (n, num_dofs)
            lifting_prev (np.ndarray): Lifting function U at step n-1, all N nodes
            lifting_next (np.ndarray): Lifting function U at step n, all N nodes

        Returns:
            np.ndarray: Interior alpha coefficients at step n
        """
        if n == 1:
            return self.first_step(u_history, lifting_prev, lifting_next)

        # Extrapolate alpha to predict the new step (Adams-Bashforth)
        alpha_star = self.extrapolate_u(u_history)
        # Evaluate nonlinearity at full solution U + v*
        u_full_star = self._full_nodes_from_alpha(alpha_star, lifting_next)

        M = self.mass_matrix()
        K = self.stiffness_matrix(u_full_star)

        A = M + (self.dt/2)*K
        B = M - (self.dt/2)*K

        # Forcing from -U_t: -h * (U^n - U^{n-1}) at interior nodes
        # (integral of -U_t against the hat basis).
        f = -self.h * (lifting_next[1:-1] - lifting_prev[1:-1])

        # Lifting-gradient forcing -dt * (K_full @ U_bar), i.e. the contribution
        # of int a(U+v) U_x phi_j'. Vanishes for symmetric BCs (U_x = 0) but is
        # required when u_B^- != u_B^+.
        u_bar = 0.5 * (lifting_prev + lifting_next)
        f = f - self.dt * self._lifting_stiffness_vector(u_full_star, u_bar)

        rhs = B @ u_history[-1, :] + f

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
