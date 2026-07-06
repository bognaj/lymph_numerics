def main():
    print("Hello from lymph-numerics!")


if __name__ == "__main__":
    # 1. Initialize:
    # - C with C(0) = C0 (number)
    # - N with N(0) = N0 (number)
    # - initial condition:u(*, 0) = v (function)
    # - boundary conditions: u(0, t) = 0, u(1, t) = 0 

    # 2. Single step:
    # - calculate the mean value of u at the current time step (spatial average)
    # - calculate the new value of C and N using the ODEs (F1 and F2) 
    #   with the mean value of u as input
    # - calculate the new values of boundary conditions for u 
    #   using the new values of C and N
    # - calculate the new values of u at the next time step using the 
    #   Crank-Nicolson Galerkin method
    main()
