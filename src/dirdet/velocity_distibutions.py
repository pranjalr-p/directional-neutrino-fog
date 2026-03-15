# PACKAGES
import numpy as np
from scipy.special import erf

# import constants
from dirdet.config.physics import GALACTIC

'''
All code in the NON-DIRECTIONAL section has been adapted from O'Hare. 
(see refs)
'''
#===============================================================================# 
#--------------------------------NON-DIRECTIONAL--------------------------------#
#===============================================================================#
def escape_velocity_normalisation(v_esc: float, sig_v: float) -> float:
    '''
    Maxwellian vel dist => vel can go infty
    Need to enforce v < v_esc
    This function ensures normalisation after truncation
    '''
    N_esc = erf(v_esc / (np.sqrt(2) * sig_v)) \
            - np.sqrt(2/np.pi) * v_esc/sig_v * np.exp(-v_esc**2/(2*sig_v**2))
    return N_esc

def mean_inverse_speed(v_min: np.ndarray | float,   # min WIMP speed for to induce recoil
        sig_v: float = GALACTIC.SIG_V,   
        v_esc: float = GALACTIC.V_ESC,   
        v_lab: float = GALACTIC.V_LAB) -> np.ndarray | float:
    '''
    Mean inverse speed of WIMP vel dist with spped greater than the min WIMP speed 
    required to induce recoil
    Event rate of WIMP scattering at a particular energy is proprtional to this value
    '''
    N_esc = escape_velocity_normalisation(v_esc=v_esc, sig_v=sig_v)

    # non-dimensionalisation
    v_0 = sig_v * np.sqrt(2.0)
    x = v_min / v_0     # threshold speed
    y = v_lab / v_0     # lab motion
    z = v_esc / v_0     # escape cut off

    # Set up conditional terms
    g = np.zeros_like(v_min)
    g[(x<abs(y-z))&(z<y)] = (1.0/(v_0*y))
    g2 = (1.0/(2.0*N_esc*v_0*y))*(erf(x+y)-erf(x-y)-(4.0/np.sqrt(np.pi))*y*np.exp(-z**2))
    g3 = (1.0/(2.0*N_esc*v_0*y))*(erf(z)-erf(x-y)-(2.0/np.sqrt(np.pi))*(y+z-x)*np.exp(-z**2))
    
    # Apply conditions
    g[(x<abs(y-z))&(z>y)] = g2[(x<abs(y-z))&(z>y)]
    g[(abs(y-z)<x)&(x<(y+z))] = g3[(abs(y-z)<x)&(x<(y+z))]
    g[(y+z)<x] = 0.0
    
    return g


#===============================================================================# 
#--------------------------------- DIRECTIONAL ---------------------------------#
#===============================================================================#
def radon_transform(
    v_min: np.ndarray,  # shape (N_E,)
    x_pix: np.ndarray,  # shape (3,) or (N_pix, 3)
    v_lab: np.ndarray,  # shape (3,)
    sig_v=GALACTIC.SIG_V,
    v_esc=GALACTIC.V_ESC,
) -> np.ndarray:

    N_esc = escape_velocity_normalisation(v_esc=v_esc, sig_v=sig_v)
    
    v_dot = np.dot(x_pix, v_lab)  # scalar or (N_pix,)
    
    # Reshape for broadcasting
    if np.ndim(v_dot) > 0:
        v_min_2d = v_min[np.newaxis, :]   
        v_dot_2d = v_dot[:, np.newaxis]    # (768, 1)
    else:
        v_min_2d, v_dot_2d = v_min, v_dot

    a = (np.abs(v_min_2d + v_dot_2d)**2) / (2*sig_v**2)
    b = v_esc**2 / (2*sig_v**2)
    num = np.exp(-a) - np.exp(-b)
    denom = N_esc * sig_v * (2*np.pi)**0.5

    cond = v_min_2d + v_dot_2d
    mask = cond < v_esc

    f_hat = np.zeros_like(a)
    f_hat[mask] = num[mask] / denom
    return f_hat  # shape (N_E,) or (N_E, N_pix)