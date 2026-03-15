'''
Universal physical constants and unit conversion factors.
'''

# PHYSICAL CONSTANTS
N_A: float = 6.02214e23 # Avogadro's constant
SPEED_OF_LIGHT_MS: float = 299792458.0  # m/s

# DERIVED CONSTANTS
C_CM = SPEED_OF_LIGHT_MS * 100.0    # speed of light in cm/s
C_KM = SPEED_OF_LIGHT_MS / 1000.0   # speed of light in km/s

# MASS CONVERSIONS
# 1 GeV/c^2 = 1.78266192e-27 kg
GEV_TO_KG: float = 1.0e6*1.783e-33 # GeV to kg conversion

# Time Conversions
SECONDS_PER_DAY: int = 86400
DAYS_PER_YEAR: float = 365.25
SECONDS_PER_YEAR: float = SECONDS_PER_DAY * DAYS_PER_YEAR