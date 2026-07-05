import numpy as np


ALPHA = 0.3
BETA = 0.35
GAMMA = 0.5
ZETA = 2

def N_minus_dynamics(N):
    return -N

def N_plus_dynamics(N):
    return N + ZETA

def N_dynamics(t, x, mode):
    return N_minus_dynamics(x[0]) if mode == "minus" else N_plus_dynamics(x[0])

def C_minus_dynamics(N, C):
    return -ALPHA * (N + 1) * C  + BETA

def C_plus_dynamics(N, C):
    return -ALPHA * (N + 1) * C + BETA + GAMMA

def C_dynamics(t, x, mode):
    return C_minus_dynamics(x[0], x[1]) if mode == "minus" else C_plus_dynamics(x[0], x[1])

def F1(t, x):
    N, C = x
    return np.array([N_minus_dynamics(N), C_minus_dynamics(N, C)])

def F2(t, x):
    N, C = x
    return np.array([N_plus_dynamics(N), C_plus_dynamics(N, C)])

def switching_function(x):
    return x[1] - 1

def get_mode(x):
    return "F1" if x[1] < 1.0 else "F2"

def dynamics(t, x, mode):
    return F1(t, x) if mode == "F1" else F2(t, x)