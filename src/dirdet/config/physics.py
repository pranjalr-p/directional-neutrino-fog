from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

#------------------------ CONSTANTS  ------------------------#

@dataclass(frozen=True)
class NuclearParams:
    '''Parameters for CEvNS and Nuclear Form Factors (Section 3.1 of Thesis).'''
    WEINBERG_SQ: float =  0.2387e0          # sin^2(Theta_W) weinberg angle
    FERMI_G: float = 1.16637e-5             # GeV**-2 ! Fermi constant in GeV
    SID_COUPLING_FM: float = 0.52           # spin independent coupling constant in fm
    NUCLEAR_SKIN_FM: float = 0.9            # nuclear skin thickness in fm
    PROTON_MASS_KEV: float =  0.9315*1e6    # proton mass in keV
    NUCLEUS_MASS_PARAM: float = 0.93141941  # multipled to num of nucleons


@dataclass(frozen=True)
class GalacticParams:
    '''Standard Halo Model (SHM) parameters for WIMP velocity distributions.'''
    V_LAB: float = 230.6   # movement of earth through WIMP halo in km/s
    V_ESC: float = 533.0   # max speed of WIMP (leaves galaxy after this) in km/s
    SIG_V: float = 167.0   # width of Gaussian Dist of WIMP speed in km/s
    RHO_0: float = 0.3     # galactic DM density


@dataclass(frozen=True)
class WimpPlotStyles:
    '''Visual standards for GDL and Surface graphs.'''
    WIMP_LINES: tuple[str] = ("dotted", "dashed", "dashdot")


#------------------------ NEUTRINO CONFIG  ------------------------#
class NeutrinoGroup(Enum):
    PP = auto()
    PEP = auto()
    CNO = auto()
    OTHER = auto()

class SignalType(Enum):
    ''' required for directional logic'''
    CONTINUOUS = auto()     # e.g., 8B, pp
    MONOCHROMATIC = auto()  # e.g., 7Be, pep
    ISOTROPIC = auto()      # e.g., Atm, DSNB 

@dataclass(frozen=True)
class SourceConfig:
    name: str               # Internal ID (e.g., "8B") used in filenames
    label: str              # LaTeX/Plot label (e.g., r"$^8$B")
    color: str              # Plotting color
    group: NeutrinoGroup    
    sig_type: SignalType    # Relevant for Directional Recoil rate
    # For WIMP overlay 
    equiv_wimp_mass: float 
    equiv_wimp_sigma: float
    max_fl_recoil: float

    # load neutirno energy and flux
    def get_nu_energy_and_flux(self, path: str):
        return np.loadtxt(f"{path}/{self.name}.txt", unpack=True)
    
class NeutrinoRegistry:
    #--- PP-Chain Group ---#
    PP = SourceConfig(
        name="pp", label="pp", color="pink",
        group=NeutrinoGroup.PP, sig_type=SignalType.CONTINUOUS,
        equiv_wimp_mass=0.14, equiv_wimp_sigma=2.39e-45,
        max_fl_recoil = 0.025
    )
    HEP = SourceConfig(
        name="hep", label="hep", color="coral",
        group=NeutrinoGroup.PP, sig_type=SignalType.CONTINUOUS,
        equiv_wimp_mass=6.52, equiv_wimp_sigma=2.64e-47,
        max_fl_recoil = 45

    )
    B8 = SourceConfig(
        name="8B", label="8B", color="red",
        group=NeutrinoGroup.PP, sig_type=SignalType.CONTINUOUS,
        equiv_wimp_mass=5.64, equiv_wimp_sigma=7.65e-45,
        max_fl_recoil = 35
    )

    # --- PEP-Chain Group (Monochromatic) ---
    PEP = SourceConfig(
        name="pep", label="pep", color="lawngreen",
        group=NeutrinoGroup.PEP, sig_type=SignalType.MONOCHROMATIC,
        equiv_wimp_mass=0.48, equiv_wimp_sigma=4.11e-44,
        max_fl_recoil = 0.3

    )
    BE7_1 = SourceConfig(
        name="7Be1", label="7Be1", color="olive",
        group=NeutrinoGroup.PEP, sig_type=SignalType.MONOCHROMATIC,
        equiv_wimp_mass=0.13, equiv_wimp_sigma=7.16e-45,
        max_fl_recoil = 0.02

    )
    BE7_2 = SourceConfig(
        name="7Be2", label="7Be2", color="green",
        group=NeutrinoGroup.PEP, sig_type=SignalType.MONOCHROMATIC,
        equiv_wimp_mass=0.29, equiv_wimp_sigma=1.00e-43,
        max_fl_recoil = 0.1
    )

    # --- CNO-Cycle Group ---
    N13 = SourceConfig(
        name="13N", label="13N", color="turquoise",
        group=NeutrinoGroup.CNO, sig_type=SignalType.CONTINUOUS,
        equiv_wimp_mass=0.40, equiv_wimp_sigma=1.09e-44,
        max_fl_recoil = 0.2
    )
    O15 = SourceConfig(
        name="15O", label="15O", color="skyblue",
        group=NeutrinoGroup.CNO, sig_type=SignalType.CONTINUOUS,
        equiv_wimp_mass=0.57, equiv_wimp_sigma=1.63e-44,
        max_fl_recoil = 0.4
    )
    F17 = SourceConfig(
        name="17F", label="17F", color="dodgerblue",
        group=NeutrinoGroup.CNO, sig_type=SignalType.CONTINUOUS,
        equiv_wimp_mass=0.58, equiv_wimp_sigma=4.19e-46,
        max_fl_recoil = 0.4
    )

    # --- Other (Isotropic) ---
    ATM = SourceConfig(
        name="Atm", label="Atm", color="magenta",
        group=NeutrinoGroup.OTHER, sig_type=SignalType.ISOTROPIC,
        equiv_wimp_mass=968.09, equiv_wimp_sigma=4.05e-48,
        max_fl_recoil = 1000

    )
    DSNB = SourceConfig(
        name="DSNB", label="DSNB", color="blueviolet",
        group=NeutrinoGroup.OTHER, sig_type=SignalType.ISOTROPIC,
        equiv_wimp_mass=39.74, equiv_wimp_sigma=1.52e-49,
        max_fl_recoil = 1000
    )

    @classmethod
    def all_sources(cls):
        return [v for v in cls.__dict__.values() if isinstance(v, SourceConfig)]
    
    @classmethod
    def by_signal_type(cls, sig_type: SignalType):
        return [v for v in cls.__dict__.values() if isinstance(v, SourceConfig) and v.sig_type == sig_type]
        
    @classmethod
    def by_name(cls, names: list):
        return [v for v in cls.__dict__.values() if isinstance(v, SourceConfig) and v.name in names]
    
#------------------------ GLOBAL INSTANCES FOR IMPORT ------------------------#

NUCLEAR = NuclearParams()
GALACTIC = GalacticParams()
WIMP_PLOT_LINES = WimpPlotStyles()