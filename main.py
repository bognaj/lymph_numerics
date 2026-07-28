import numpy as np

from src.runge_kutta import RungeKutta
from src.galerkin import Galerkin


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
                        t_final=t_final, v=v)

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
    
    lifting_function_values[0, :] = galerkin.get_lifting_function_values(v(xs)[0],
                                                                         v(xs)[-1])
    
    while current_time < t_final:
        # get the chemical substances concentration for current step
        N_new, C_new = rk.single_step(t=current_time,
                                      NC_s=np.array([N_values[current_n],
                                                    C_values[current_n]]),
                                      R_values=rk.u_to_r(u_values[current_n, :]))
        N_values[current_n + 1] = N_new
        C_values[current_n + 1] = C_new

        # calculate the new boundary conditions
        bc_0, bc_1 = galerkin.boundary_conditions(C_values=C_values)

        # get the lifting function for further u calculations
        lifting_function = galerkin.get_lifting_function_values(bc_0, bc_1)
        lifting_function_values[current_n, :] = lifting_function

        # 

        current_time += dt
        current_n += 1
        galerkin.n += 1


        
    
