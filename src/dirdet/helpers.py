import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping, minimize

from dirdet.config.targets import XENON
from dirdet.config.physics import NeutrinoRegistry, SourceConfig
from dirdet.neutrino_rates import dRdE_CEvNS
from dirdet.wimp_rates import dRdE_WIMP


#from constants.plot_parameters import NU_PLOT_COLOURS, WIMP_PARAMS
def latex_float(f):
    float_str = "{:.2e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        # Remove leading zeros and '+' from exponent    
        exponent = int(exponent) 
        return rf"${base} \times 10^{{{exponent}}}$"
    else:
        return rf"${float_str}$"


#====================================================================#
#----------------- Neutrino Relate Helper Functions -----------------#
#====================================================================#
def nu_load_EnergyAndFlux(
    nu_ls: list[SourceConfig] | SourceConfig, 
    dir: str = "input_data/neutrino_fluxes"
) -> tuple[np.ndarray, np.ndarray] | tuple[dict, dict]:
    ''' Loads neutirno flux and energy, based on iterable group'''

    if isinstance(nu_ls, SourceConfig):
        return np.loadtxt(f"{dir}/{nu_ls.name}.txt",unpack=True)

    E_nu_dic, Flux_dic = {}, {}
    for nu in nu_ls:
        E_nu_dic[nu.name], Flux_dic[nu.name] = np.loadtxt(f"{dir}/{nu.name}.txt",unpack=True)
    return E_nu_dic, Flux_dic

def dRdE_neutrino_dict(
        dir: str="input_data/neutrino_fluxes",
        E_r: np.ndarray = np.geomspace(1e-3,1e3,1000),
) -> dict | tuple[np.ndarray, np.ndarray]:  
    """
    Precalculates neutrino recoil rates for all sources in the Registry.
    Returns a dictionary keyed by source name (e.g., '8B', 'Atm').
    """
    
    all_sources = NeutrinoRegistry.all_sources()
    dR_nu_dict = {}

    for source in all_sources:
        # load neutrino data
        E_nu, Flux = np.loadtxt(f"{dir}/{source.name}.txt", unpack=True)

        # calculate data
        dR = dRdE_CEvNS(E_r=E_r, E_nu=E_nu, Flux=Flux, A=XENON.A, Z=XENON.Z)
        dR_nu_dict[source.name] = dR
    
    return dR_nu_dict


def neutrino_angles(
    xpix_num: int = 8,
    num_angles: tuple[int, str] | None = None,  # return subset angles, with specific strat
    q_sun: np.ndarray[3] = np.array([0,1,0]),
    mask: bool=True
    ) -> np.ndarray:

    ''' Returs an array of costheta sun arrays, based on xpix value
    required for diretional nu rate'''

    q = np.loadtxt(f"input_data/xpix/xpix{xpix_num}.txt",unpack=False)
    cosThetaSun_arr = np.dot(q,q_sun)
    if mask:
        cosThetaSun_arr = cosThetaSun_arr[cosThetaSun_arr < 0] # keep only relavant recoil

    if num_angles is None:  return cosThetaSun_arr

    strat = num_angles[1]
    valid_strat = ("first","last","rand")
    if strat.lower() not in valid_strat:
        return ValueError(f"stratergy {strat} is not recognised. Choose from {valid_strat}")

    n = num_angles[0]   
    if strat == "first":    return cosThetaSun_arr[0:n]
    if strat == "last":     return cosThetaSun_arr[-n:]
    return  np.random.choice(cosThetaSun_arr, size=n, replace=False)


#====================================================================#
#----------------- WIMP Recoil Rate Calculation -----------------#
#====================================================================#
def wimp_max_recoil(E_r: np.ndarray, dR: np.ndarray ) -> float:
    ''' Returns last non-zero E_r for wimp'''
    # handle all 0 case
    if all(dR == 0.0):  return 0

    # skip sequence of zero at start
    first_nonzero_idx = np.argmax(dR != 0)
    first_zero_after = first_nonzero_idx + np.argmax(dR[first_nonzero_idx:] == 0)

    return E_r[first_zero_after + 10]


#====================================================================================#
#----------------- WIMP-Neutrino-Overlay WIMP Parameters Derivation -----------------#
#====================================================================================#

def log_objective(log_params, E_r:np.ndarray, A: int, dR_nu: np.ndarray) -> float:
    ''' Objective function for optimiser'''
    # Convert log-parameters back to physical values
    m_chi = 10**log_params[0]
    sigma_p = 10**log_params[1]
    
    # Compute the physics model
    dR_wimp = dRdE_WIMP(E_r, m_chi, sigma_p, A)
    
    # Add a tiny floor (eps) to avoid log(0) if dR is zero
    eps = 1e-60
    
    # Compare the LOGS of the curves
    # This ensures the optimizer cares about the tail of the curve
    log_mse = np.mean((np.log10(dR_wimp + eps) - np.log10(dR_nu + eps))**2)
    
    return log_mse

def overlay_parm(dR_nu: np.ndarray,
                 A: int,
                 E_r: np.ndarray = np.geomspace(1e-3,1e3,1000),
                 log_bounds: list = [(0, 4), (-49, -45)],
                 log_initial_guess: list = [1.0, -47.0],
                 niter: int = 100,
                 temp: float = 1.0,  # temp for accepting higher steps
                 stepsize: float = 0.2,   
                 method: str = "L-BFGS-B",
                 tol: float =1e-6): 
    ''' Returns the m_chi and sigma_p of a WIMP which closest matches the neutrino recoil rate'''

    # minimiser kwards
    minimizer_kwargs = {
        "method": method,
        "args": (E_r, A, dR_nu),
        "bounds": log_bounds,
        "tol": tol
    }

    # Run Basinhopping
    result = basinhopping(
        func=log_objective,
        x0=log_initial_guess,
        niter=niter,           # Total number of basin jumps
        T=temp,               
        stepsize=stepsize,      
        minimizer_kwargs=minimizer_kwargs
    )
    '''
    result = minimize(
        fun=log_objective, 
        args= (E_r,A,dR_nu),
        x0=log_initial_guess, 
        method='L-BFGS-B', 
        bounds=log_bounds,
        tol=1e-9 # Tighten tolerance for better precision
    )
    '''

    m_chi = 10**result.x[0]
    sig_p = 10**result.x[1]

    return m_chi,sig_p