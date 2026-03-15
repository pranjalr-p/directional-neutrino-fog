import numpy as np

from dirdet.config.conversions import C_KM as c_km, C_CM as c_cm, GEV_TO_KG as GeV_2_kg, SECONDS_PER_YEAR as seconds2year
from dirdet.config.physics import NUCLEAR, GALACTIC

from dirdet.nuclear_phyics import form_factor_helm
from dirdet.velocity_distibutions import mean_inverse_speed, radon_transform

'''
All code in the NON-DIRECTIONAL section has been adapted from O'Hare. 
(see refs)
'''
#===============================================================================# 
#--------------------------------NON-DIRECTIONAL--------------------------------#
#===============================================================================#

def min_wimp_speed(
        E_r: np.ndarray | float,   # recoil energy in keV
        A: int,          # target nucleus mass number
        m_chi: np.ndarray | float,   # WIMP mass in GeV
        delta: int=0    # inelastic scattering parameter
        ) -> np.ndarray | float:
    ''' Calculate min wimp speed needed to induce recoil '''
    # unpack variables
    m_p_keV = NUCLEAR.PROTON_MASS_KEV

    m_N_keV = A * m_p_keV # nucleus mass in keV
    m_chi_keV = m_chi * 1.0e6
    mu_N_keV = (m_chi_keV * m_N_keV) / (m_chi_keV + m_N_keV) # reduced nucleus mass
    v_min = np.sqrt(1.0/(2*m_N_keV*E_r))*(m_N_keV*E_r/mu_N_keV + delta)*c_km
    return v_min


def dRdE_WIMP(E_r,m_chi,sigma_p,A,rho_0=0.3):
    '''
    * Spin independent differentual recoil rate that takes in recoil energy in 
    units of keV and a proton cross section in units of cm^2 and outputs a rate
    in units of (ton year keVr)^-1
    '''
    
    # unpack variables
    m_p_keV = NUCLEAR.PROTON_MASS_KEV

    # DM-proton reduced mass (in units of keV)
    mu_p = 1.0e6*m_chi*m_p_keV/(1.0e6*m_chi + m_p_keV)
    
    # Rate constants (in units cm kg^-1 s^-2)
    R0 = (c_cm**2)*((rho_0*1.0e6*A**2*sigma_p)/(2*m_chi*GeV_2_kg*mu_p**2)) 
    
    # Mean inverse speed
    v_min = min_wimp_speed(E_r,A,m_chi)
    g = mean_inverse_speed(v_min)/(1000.0*100.0) # convert to cm^-1 s

    # Compute rate = (Rate amplitude * gmin * form factor)
    FF = form_factor_helm(E_r,A)**2.0
    dR = R0*g*FF

    # convert to (ton-year-keV)^-1
    dR = dR*seconds2year*1000.0 

    return dR

'''
All Code in the directionl section is my original work
'''
#===============================================================================# 
#--------------------------------- DIRECTIONAL ---------------------------------#
#===============================================================================#

def recoil_vector(theta: float, phi: float) -> np.ndarray:
    ''' Using theta and phi angel, creates the recoil vector x_pix'''
    x = np.array([np.sin(theta)*np.cos(phi),
                  np.sin(theta)*np.sin(phi),
                  np.cos(theta)])
    return x


def dRdEdO_WIMP(
        E_r: np.ndarray,
        x_pix: np.ndarray,
        m_chi: float | int,
        sigma_p: float,
        A: float | int,
        v_lab: np.ndarray,
        rho_0=GALACTIC.RHO_0):
    '''
    * Spin independent differentual recoil rate that takes in recoil energy in 
    units of keVr and a proton cross section in units of cm^2 and outputs a rate
    in units of (ton year keVr)^-1
    
    * gvmin_function should be a function that takes in v_min in (km/s) and outputs
    g(v_min) in units of (km/s)^-1
    '''
    m_p_keV = NUCLEAR.PROTON_MASS_KEV

    # DM-proton reduced mass (in units of keV)
    mu_p = 1.0e6*m_chi*m_p_keV/(1.0e6*m_chi + m_p_keV)
    
    # Rate constants (in units cm kg^-1 s^-2)
    R0 = (c_cm**2)*((rho_0*1.0e6*A**2*sigma_p)/(2*m_chi*GeV_2_kg*mu_p**2)) 

    # Mean inverse speed
    v_min = min_wimp_speed(E_r=E_r,A=A,m_chi=m_chi)
    f = radon_transform(v_min,x_pix=x_pix,v_lab=v_lab)/(1000.0*100.0) # convert to cm^-1 s
    
    # Compute rate = (Rate amplitude * gmin * form factor)
    FF = form_factor_helm(E_r,A)**2.0 
    dR = R0 * 1/(2*np.pi) * f * FF[np.newaxis, :]  
    dR = dR*seconds2year*1000.0 # convert to (ton-year-keV)^-1

    return dR
