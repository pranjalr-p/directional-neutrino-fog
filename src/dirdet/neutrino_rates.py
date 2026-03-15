import numpy as np

from dirdet.config.physics import NUCLEAR, SignalType, SourceConfig
from dirdet.config.conversions import N_A
from dirdet.config.targets import FLUORINE, Atom

from dirdet.nuclear_phyics import form_factor_helm


'''
All code in the NON-DIRECTIONAL section has been adapted from O'Hare. 
'''
#===============================================================================# 
#--------------------------------NON-DIRECTIONAL--------------------------------#
#===============================================================================#

def diff_cross_sec(
        E_r: float,
        E_nu: float | np.ndarray,
        m_N_GeV: float,
        m_N_keV: float,
        Q_W: float,
        A: int,
        Z: int,
        G_F_GeV: float = NUCLEAR.FERMI_G,
) -> float | np.ndarray:
    ''' Calculates the differential cross section, without the form factor'''

    const = (G_F_GeV**2 * Q_W**2 * m_N_GeV) / (4.0 * np.pi)
    conv =  (0.197e-13)**2.0*(1.0e-6)*1000.0/(1.0*A+1.0*Z)*(N_A)
    dsigma = const * (1.0 -(m_N_keV*E_r)/(2.0 * (E_nu*1000.0)**2)) * conv
    return dsigma            

def dRdE_CEvNS(
        E_r: np.ndarray,
        E_nu: np.ndarray | float,
        Flux: np.ndarray | float,
        A: int,
        Z: int
) -> np.ndarray:
    '''
    * Calculates the CEvNS rates for an input array of neutrino energies and the corresponding flux
    * N = number of neutrons in the target
    * Z = number of protons
    * E_r = array of recoil energies to calculate the rate at
    * E_nu and Flux can be an array for continuous fluxes, or just scalars for neutrino lines
    '''
    # Parameters
    sinTheta_Wsq = NUCLEAR.WEINBERG_SQ
    Q_W = 1.0*A-(1-4.0*sinTheta_Wsq)*Z # weak nuclear hypercharge
    m_N_GeV = 0.93141941*(A+Z) # nucleus mass in GeV
    m_N_keV = m_N_GeV*1.0e6 # nucleus mass in keV

    FF = form_factor_helm(E_r,A+Z)**2.0 # Form factor squared

    dRdE = np.zeros_like(E_r)

    # continious neutrino flux
    if np.size(E_nu) > 1:
        for i in range(0,np.size(E_r)):
            dsigma = diff_cross_sec(E_r[i],E_nu, m_N_GeV, m_N_keV, Q_W, A, Z)
            dsigma[dsigma<0.0] = 0.0  
            dRdE[i] = np.trapezoid(dsigma*Flux*FF[i],E_nu)

    # monochromatic neutrino flux (=> discrete lines)
    else:
        for i in range(0,np.size(E_r)):
            dsigma = diff_cross_sec(E_r[i],E_nu, m_N_GeV, m_N_keV, Q_W, A, Z)
            if dsigma <= 0: continue
            dRdE[i] = dsigma*Flux*FF[i]*E_nu

    # Convert into /ton/year/keV
    dRdE = dRdE*1000*365.25*3600*24

    return dRdE


#===============================================================================# 
#----------------------------------DIRECTIONAL----------------------------------#
#===============================================================================#

def monochromatic_recoil(
        E_r: np.ndarray,
        cosThetaSun_arr: np.ndarray,
        E_nu: float,
        Flux: float,
        FF: np.ndarray,
        eta = 0.01,      # thichkness of line
        target: Atom = FLUORINE
) -> np.ndarray:
    ''' Calculates the driectional recoil rate for monochromatic nuclear neutirno CEvNS recoil
    Differs from continuos due to not needing to integrate'''
    
    # parameters
    N, Z = target.N, target.Z
    m_N_GeV = NUCLEAR.NUCLEUS_MASS_PARAM * (N+Z)
    m_N_keV = m_N_GeV * 1e6
    G_F_GeV = NUCLEAR.FERMI_G
    sinTheta_Wsq = NUCLEAR.WEINBERG_SQ
    Q_W = 1.0*N - (1-4.0*sinTheta_Wsq)*Z    # weak nuclear hypercharge

    E_nu_keV = E_nu * 1e3       # MeV -> keV
    
    # cross section
    dsigma = (
        (G_F_GeV**2 / (4 * np.pi)) * Q_W**2 * m_N_GeV
        * (1 - (m_N_keV * E_r) / (2 * E_nu_keV**2))
        * (0.197e-13)**2 * 1e-6 * 1000.0 / (N + 1.0 * Z) * N_A
    )

    # calcualte rate, rate calc is not dep on angle, only allocation of it
    dR =  dsigma * Flux * E_nu * FF * 1/(2*np.pi)

    # kinematic constraint mask
    cos_beta = ((E_nu_keV + m_N_keV)/E_nu_keV) * np.sqrt(E_r/(2*m_N_keV))
    kin_cons = np.abs(-cosThetaSun_arr[:,np.newaxis] - cos_beta) < eta 

    # apply kinematic constraint
    dR_mat = np.zeros((np.size(cosThetaSun_arr),np.size(E_r)))
    cols = np.where(kin_cons)[1]
    dR_mat[kin_cons] = dR[cols]

    return dR_mat


def isotropic_recoil(
    E_r: np.ndarray,   
    E_nu: np.ndarray,
    Flux: np.ndarray,
    FF: np.ndarray,
    target: Atom = FLUORINE
) -> np.ndarray:
    ''' Calculates the driectional recoil rate for isotropic_recoil nuclear neutirno CEvNS recoil
    Differs from continuos due to not needing to no costheta requirement'''
    
    # parameters
    N, Z= target.N, target.Z
    m_N_GeV = NUCLEAR.NUCLEUS_MASS_PARAM * (N+Z)
    m_N_keV = m_N_GeV * 1e6
    G_F_GeV = NUCLEAR.FERMI_G
    Q_W = NUCLEAR.WEINBERG_SQ
    E_nu_keV = E_nu * 1e3  # MeV -> keV
    sinTheta_Wsq = NUCLEAR.WEINBERG_SQ
    Q_W = 1.0*N - (1-4.0*sinTheta_Wsq)*Z    # weak nuclear hypercharge

    # mask for physically valid recoil energies 
    E_max = (2.0 * m_N_keV * np.max(E_nu_keV)**2) / (m_N_keV + np.max(E_nu_keV))**2     # (size(e_nu))
    valid = E_r <= E_max  

    Er_valid = E_r[valid][:, np.newaxis]          # (n_valid, 1)
    E_nu_2d  = E_nu_keV[np.newaxis, :]            # (1, n_Enu)

    dsigma_2d = (
    (G_F_GeV**2 / (4 * np.pi)) * Q_W**2 * m_N_GeV
    * (1 - (m_N_keV * Er_valid) / (2 * E_nu_2d**2))
    * (0.197e-13)**2 * 1e-6 * 1000.0 / (N + 1.0 * Z) * N_A
    )    # shape: (n_valid, n_Enu)

    # zero out negative values 
    dsigma_2d[dsigma_2d < 0] = 0.0

    # --- integrate over E_nu for each valid E_r ---
    # Flux and FF[valid] broadcast over E_nu axis
    integrand = dsigma_2d * Flux[np.newaxis, :] * FF[valid][:, np.newaxis]

    dRdEdO = np.zeros_like(E_r)
    dRdEdO[valid] = np.trapezoid(integrand, E_nu, axis=1)
    dRdEdO /= 4.0 * np.pi

    return dRdEdO


def cts_recoil(
    E_r: np.ndarray,
    cosThetaSun_arr: float | int,
    E_nu: np.ndarray,
    Flux: np.ndarray,
    FF: np.ndarray,
    target: Atom = FLUORINE
) -> np.ndarray:
    
    # parameters
    N, Z = target.N, target.Z
    m_N_GeV = NUCLEAR.NUCLEUS_MASS_PARAM * (N+Z)
    m_N_keV = m_N_GeV * 1e6
    G_F_GeV = NUCLEAR.FERMI_G
    sinTheta_Wsq = NUCLEAR.WEINBERG_SQ
    Q_W = 1.0*N - (1-4.0*sinTheta_Wsq)*Z    # weak nuclear hypercharge

    E_nu_keV = E_nu * 1e3  # MeV -> keV
    
    # mask for physically valid recoil energies 
    E_max = (2.0 * m_N_keV * np.max(E_nu_keV)**2) / (m_N_keV + np.max(E_nu_keV))**2     # (size(e_nu))

    # kinematic constraint
    E_min = np.sqrt(m_N_keV * E_r / 2.0)  # shape (n_Er,)

    # epsilon 
    eps = -(m_N_keV * E_min) / (m_N_keV * cosThetaSun_arr[:, np.newaxis] + E_min)

    # physical validity mask
    valid_energy = (E_r <= E_max) & (np.arange(len(E_r)) != 0)
    valid_angle  = cosThetaSun_arr[:, np.newaxis] < -E_min/ m_N_keV
    valid_eps    = (eps > E_min) & (eps < E_nu_keV[-1])
    valid        = valid_energy & valid_angle & valid_eps

    # cross section
    dsigma = (
        (G_F_GeV**2 / (4*np.pi)) * Q_W**2 * m_N_GeV * \
        (1 - (m_N_keV * E_r) / (2 * eps**2)) * (0.197e-13)**2 \
        * (1.0e-6) * 1000.0 / (N + 1.0*Z) * N_A  # shape (n_valid,)
    )

    # interpolate flux
    Flux_ip = np.interp(eps, E_nu_keV, Flux)

    # calculate recoil, and apply constraint mask
    dR = dsigma * Flux_ip * eps**2 / (1000 * E_min) * FF
    dR[~valid] = 0
    dR /= 2.0 * np.pi

    return dR


def dRdEdO_CEvNS(
    neutrino: SourceConfig,
    E_r: np.ndarray,
    cosThetaSun_arr: np.ndarray,
    E_nu: np.ndarray,
    Flux: np.ndarray,
    FF: np.ndarray,
    eta: float = 0.01,  # param for mono neutrinos
    target: Atom = FLUORINE
) -> np.ndarray:
    ''' Calculates the differential neutrino reocil rate as a function of energy E_r and angle cosThetaSun
    Uses different calcualtion fucntions based on signal type of neutrino'''

    # cts neutrinos
    if neutrino.sig_type == SignalType.CONTINUOUS:
        dR_mat = cts_recoil(
            E_r=E_r, cosThetaSun_arr=cosThetaSun_arr, 
            E_nu=E_nu, Flux=Flux, 
            FF=FF, target=target
        )
    
    # monochromatic neutrinos
    elif neutrino.sig_type == SignalType.MONOCHROMATIC:
        dR_mat = monochromatic_recoil(
            E_r=E_r, cosThetaSun_arr=cosThetaSun_arr, 
            E_nu=E_nu, Flux=Flux, 
            FF=FF, target=target, eta=eta)

    # isotropic neutrinos
    else:
        dR = isotropic_recoil(
            E_r=E_r, 
            E_nu=E_nu, Flux=Flux, 
            FF=FF, target=target)
        dR_mat = np.ones((len(cosThetaSun_arr), 1)) * dR
    #Convert into /ton/year/keV
    dR_mat *= 1000*365.25*3600*24

    return dR_mat