# VESSEL PARAMETERS
R0 = 0.5 # typical radius of the vessel
L = 1.0 # length of the vessel
G = 1.0 # pressure amplitude of stretching
rho = 1.0 # fluid density
Vx = 1.0 # typical fluid velocity
nu = 0.01 # fluid viscosity
Re = (L * Vx) / nu # Reynolds number
Re_eps = R0/L
eta = G/(rho * Vx**2) * Re * (Re_eps)**2 # dimensionless pressure amplitude

# CHEMICAL REACTION PARAMETERS
k_NO_minus = 0.1 # rate of NO removal
k_NO_plus = 0.2 # rate of NO production
k_Ca_minus = 0.05 # rate of Ca removal
k_Ca_plus = 0.1 # rate of Ca production
C_shear = 1.0 # threshold concentration of shear stress for NO production
m = 0.5 # rate constant for NO inhibition of Ca production
R_crit = 0.77 * R0 # critical radius activating stretch channels
tau_ref = 0.1 # reference level of shear stress