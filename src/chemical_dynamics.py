import numpy as np


def N_minus_dynamics(N):
    return -N

def N_plus_dynamics(N, alpha, beta, gamma, zeta):
    return N + zeta

def N_dynamics(t, x, mode, alpha, beta, gamma, zeta):
    return N_minus_dynamics(x[0]) if mode == "minus" else N_plus_dynamics(x[0], alpha, beta, gamma, zeta)

def C_minus_dynamics(N, C, alpha, beta, gamma, zeta):
    return -alpha * (N + 1) * C  + beta

def C_plus_dynamics(N, C, alpha, beta, gamma, zeta):
    return -alpha * (N + 1) * C + beta + gamma

def C_dynamics(t, x, mode, alpha, beta, gamma, zeta):
    return C_minus_dynamics(x[0], x[1], alpha, beta, gamma, zeta) if mode == "minus" else C_plus_dynamics(x[0], x[1], alpha, beta, gamma, zeta)

def F1(t, x, alpha, beta, gamma, zeta):
    N, C = x
    return np.array([N_minus_dynamics(N), C_minus_dynamics(N, C, alpha, beta, gamma, zeta)])

def F2(t, x, alpha, beta, gamma, zeta):
    N, C = x
    return np.array([N_plus_dynamics(N, alpha, beta, gamma, zeta), C_plus_dynamics(N, C, alpha, beta, gamma, zeta)])

def switching_function(x):
    return x[1] - 1

def get_mode(x):
    return "F1" if x[1] < 1.0 else "F2"

def dynamics(t, x, mode, alpha, beta, gamma, zeta):
    return F1(t, x, alpha, beta, gamma, zeta) if mode == "F1" else F2(t, x, alpha, beta, gamma, zeta)