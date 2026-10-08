# VESSEL PARAMETERS
R0 = 1e-3 # typical radius of the vessel
L = 3e-3 # length of the vessel
G = 0.01 # pressure amplitude of stretching
rho = 998 # fluid density
Vx = 5e-3 # typical fluid velocity
nu = 0.001/rho # fluid kinematic viscosity
Re = (L * Vx) / nu # Reynolds number
Re_eps = R0/L
eta = G/(rho * Vx**2) * Re * (Re_eps)**2 # dimensionless pressure amplitude

# CHEMICAL REACTION PARAMETERS
k_NO_minus = 75 # rate of NO removal
k_NO_plus = 75 # rate of NO production
k_Ca_minus = 75 # rate of Ca removal
k_Ca_plus = 40 # rate of Ca production
C_shear = 1.1 # threshold concentration of shear stress for NO production
m = 0.5 # rate constant for NO inhibition of Ca production
R_crit = 0.77 * R0 # critical radius activating stretch channels
tau_ref = 0.1 # reference level of shear stress
R_min = 0.4 # dimensionless lower bound for the contracted vessel radius