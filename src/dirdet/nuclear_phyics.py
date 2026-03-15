import numpy as np
from dirdet.config.physics import NUCLEAR


'''
All code here has been adapted from O'Hare.
'''

def nucleus_size(A: int) -> float:
    # parameters
    s = NUCLEAR.NUCLEAR_SKIN_FM
    a = NUCLEAR.SID_COUPLING_FM

    c = 1.23 *A**(1/3) - 0.6
    r1 = np.sqrt(c**2 + (7/3) * np.pi**2 * a**2 - 5*s**2)
    return r1

def form_factor_helm(
        E_r: np.ndarray | float, 
        A:int # num of nucleons
) -> np.ndarray | float:
    # parameters
    s = NUCLEAR.NUCLEAR_SKIN_FM
    a = NUCLEAR.SID_COUPLING_FM

    # Momentum transfer in fm^-1
    q = np.sqrt(2*A*931.5*1000*E_r)*1.0e-12/1.97e-7 # q = sqrt(2 m_N E_r)
    
    r_1 = nucleus_size(A)
    F = (3*(np.sin(q*r_1) - q*r_1*np.cos(q*r_1))*np.exp(-q*q*s*s/2.0)/(q*r_1)**3)
    F[q==0.0] = 1.0

    return F
