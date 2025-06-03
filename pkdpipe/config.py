"""
Configuration management for pkdpipe.

This module centralizes configuration settings, paths, default parameters,
and simulation/cosmology presets for the pkdpipe package.
"""
import os
from pathlib import Path

# --- Environment Variable Dependent Paths ---
def get_env_variable(var_name: str, default: str | None = None) -> str:
    """Fetches an environment variable, raises error if not found and no default."""
    value = os.getenv(var_name)
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"Environment variable {var_name} not set and no default provided.")
    return value

SCRATCH_PATH = get_env_variable("SCRATCH", "/tmp/scratch")  # Added a default for robustness
CFS_PATH = get_env_variable("CFS", "/tmp/cfs") # Added a default for robustness
USER = get_env_variable("USER", "default_user") # Added a default for robustness
PKDGRAVEMAIL = get_env_variable("pkdgravemail", "user@example.com")

DEFAULT_RUN_DIR_BASE = f"{CFS_PATH}/cosmosim/slac/{USER}/pkdgrav3/runs"
DEFAULT_SCRATCH_DIR_BASE = f"{SCRATCH_PATH}/pkdgrav3/runs"

# --- Package Structure Paths ---
PACKAGE_ROOT = Path(__file__).parent.resolve()
TEMPLATES_DIR = PACKAGE_ROOT.parent / "templates"
SCRIPTS_DIR = PACKAGE_ROOT.parent / "scripts"

# --- Template File Paths ---
TEMPLATE_LIGHTCONE_PAR = TEMPLATES_DIR / "lightcone.par"
TEMPLATE_SLURM_SH = TEMPLATES_DIR / "slurm.sh"
SCRIPT_RUN_SH = SCRIPTS_DIR / "run.sh"

# --- Default Simulation Parameters ---
DEFAULT_JOB_NAME_TEMPLATE = "N{nGrid}-L{dBoxSize}-{gpupern}gpus" # Corrected nodes.gpupern to gpupern based on typical usage
DEFAULT_SIMULATION_NAME = "lcone-small"
DEFAULT_COSMOLOGY_NAME = "desi-dr2-planck-act-mnufree"
DEFAULT_TIME_LIMIT = "48:00:00"
DEFAULT_NODES = 2
DEFAULT_CPUS_PER_TASK = 128
DEFAULT_GPUS_PER_NODE = 4
DEFAULT_NGRID = 1400
DEFAULT_BOXSIZE_MPC_H = 1050
DEFAULT_REDSHIFT_FROM = 12.0
DEFAULT_LPT_ORDER = 3
DEFAULT_OUTPUT_INTERVAL = 1

# --- Default Redshift Lists ---
# Combined and sorted unique redshifts
_DEFAULT_REDSHIFT_LIST_VALUES = sorted(list(set([
    3.0, 2.81388253, 2.63504180, 2.46500347, 2.30360093, 2.14960630, 
    2.00300300, 1.86204923, 1.72851296, 1.60145682, 1.48015873, 1.36406619, 
    1.25377507, 1.14868930, 1.04834084, 0.95274360, 0.86150410, 0.77462289, 
    0.69176112, 0.61290323, 0.53751538, 0.46584579, 0.39742873, 0.33226752, 
    0.27000254, 0.21065375, 0.15420129, 0.10035211, 0.04898773, 0.00000000,
    0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.33, 0.922, 0.955
])), reverse=True)

DEFAULT_REDSHIFT_TARGETS_STR = f'{[f"{z:0.4f}" for z in _DEFAULT_REDSHIFT_LIST_VALUES]}'.replace("'", "")
DEFAULT_NSTEPS_STR = f'{[1 for i in range(len(_DEFAULT_REDSHIFT_LIST_VALUES))]}'.replace("'", "")


# --- Simulation Presets ---
# Structure: 'preset_name': {param_key: value_for_preset}
# These override the general defaults.
SIMULATION_PRESETS = {
    "lcone-small": {
        "dBoxSize": 525,
        "nGrid": 700,
        "nodes": 1,
    },
    "lcone-medium": {
        # Uses general defaults if not specified
        "dBoxSize": DEFAULT_BOXSIZE_MPC_H, # Example: explicitly setting default
        "nGrid": DEFAULT_NGRID,
        "nodes": DEFAULT_NODES,
    },
    "lcone-large": {
        "dBoxSize": 2100,
        "nGrid": 2800,
        "nodes": 16,
    },
    
    # Campaign-specific presets for summer multi-cosmology campaigns
    "summer-validation": {
        "dBoxSize": 1050,
        "nGrid": 1400,
        "nodes": 50,
        "gpupern": 4,
        "tlimit": "12:00:00",
        "comment": "Validation resolution for campaign testing",
    },
    "summer-production": {
        "dBoxSize": 5250,
        "nGrid": 7000,
        "nodes": 250,
        "gpupern": 4,
        "tlimit": "48:00:00",
        "comment": "Full summer resolution for production simulations",
    },
    "summer-scaling-2800": {
        "dBoxSize": 2100,
        "nGrid": 2800,
        "nodes": 64,
        "gpupern": 4,
        "tlimit": "24:00:00",
        "comment": "LCDM scaling test with 2800³ grid",
    },
    "summer-scaling-4200": {
        "dBoxSize": 3150,
        "nGrid": 4200,
        "nodes": 144,
        "gpupern": 4,
        "tlimit": "36:00:00",
        "comment": "LCDM scaling test with 4200³ grid",
    },
}

# --- Cosmology Presets ---
# Structure: 'preset_name': {cosmology_param_key: value}
COSMOLOGY_PRESETS = {
    "desi-dr2-planck-act-mnufree": {
        # Based on original get_cosmology defaults
        "h": 68.906658 / 100,
        "ombh2": 0.022597077,
        "omch2": 0.11788858,
        "As": 2.1302742e-09,
        "ns": 0.97278164,
        "TCMB": 2.72548,
        "omegak": 0.0,
        "sigma8": None, # CAMB will calculate this if None
        "mnu": 0.0, # Explicitly 0 as per original comment
        "nnu": 3.044,
        "tau": 0.0543,
        "numnu": 3, # Assuming 3 massive neutrinos if mnu is non-zero, or 3 total species
        "harchy": "normal",
        "k0phys": 0.05,
    },
    "euclid-flagship": {
        "h": 0.67,
        # ombh2 and omch2 will be derived from omegab and (omegam-omegab) * h^2
        "omegam": 0.32, 
        "omegab": 0.049,
        "sigma8": 0.831,
        "As": 2.1e-9,
        "ns": 0.96,
        "mnu": 0.0587,
        "TCMB": 2.72548, # Assuming standard TCMB
        "omegak": 0.0,   # Assuming flat
        "nnu": 3.044,    # Assuming standard N_eff
        "tau": 0.0543,   # Assuming a common tau value
        "numnu": 1, # Typically for Euclid flagship mnu is one massive eigenstate
        "harchy": "normal", # Or degenerate, depending on interpretation
        "k0phys": 0.05,
    },
    "planck18": {
        "h": 67.36 / 100,
        "ombh2": 0.02237,
        "omch2": 0.1200,
        "As": 2.100e-09,
        "ns": 0.9649,
        "TCMB": 2.72548,
        "omegak": 0.0,
        "sigma8": None, # CAMB will calculate this
        "mnu": 0.06, # Sum of neutrino masses in eV
        "nnu": 3.046,
        "tau": 0.0544,
        "numnu": 3,
        "harchy": "normal",
        "k0phys": 0.05,
    },
    "wCDM_w0wa": {
        # Base LCDM with w0-wa parameterization
        "h": 0.67,
        "omegam": 0.32,
        "omegab": 0.049,
        "sigma8": 0.83,
        "As": 2.1e-9,
        "ns": 0.96,
        "mnu": 0.0,
        "TCMB": 2.72548,
        "omegak": 0.0,
        "nnu": 3.044,
        "tau": 0.0543,
        "numnu": 0,
        "harchy": "normal",
        "k0phys": 0.05,
        # Dark energy equation of state parameters
        "w0": -1.0,
        "wa": 0.0,
    },
    "phiCDM_base": {
        # Base configuration for scalar field dark energy
        "h": 0.67,
        "omegam": 0.32,
        "omegab": 0.049,
        "sigma8": 0.83,
        "As": 2.1e-9,
        "ns": 0.96,
        "mnu": 0.0,
        "TCMB": 2.72548,
        "omegak": 0.0,
        "nnu": 3.044,
        "tau": 0.0543,
        "numnu": 0,
        "harchy": "normal",
        "k0phys": 0.05,
        # Scalar field parameters (to be customized per variant)
        "phi_model": "quintessence",
        "phi_params": {},
    },
}

# Add summer campaign cosmology variants based on desi-dr2-planck-act-mnufree
# These inherit from the base preset and only override specific parameters
_base_summer_cosmology = COSMOLOGY_PRESETS["desi-dr2-planck-act-mnufree"].copy()

COSMOLOGY_PRESETS.update({
    "summer-lcdm": {
        **_base_summer_cosmology,
        "description": "Summer campaign LCDM cosmology based on DESI-DR2-Planck-ACT"
    },
    "summer-wcdm": {
        **_base_summer_cosmology,
        # Dark energy equation of state parameters (only parameters that differ)
        "w0": -0.9,  # Different from LCDM for comparison
        "wa": 0.1,   # Time-varying component
        "description": "Summer campaign wCDM cosmology with evolving dark energy"
    },
    "summer-phicdm": {
        **_base_summer_cosmology,
        # Scalar field parameters (only parameters that differ)
        "phi_model": "quintessence",
        "phi_params": {
            "potential": "exponential",
            "lambda_phi": 0.1
        },
        "description": "Summer campaign phiCDM cosmology with scalar field dark energy"
    },
})

# Add flagship campaign cosmology variants based on Planck 2018
# These inherit from the planck18 preset and only override specific parameters
_base_flagship_cosmology = COSMOLOGY_PRESETS["planck18"].copy()

COSMOLOGY_PRESETS.update({
    "flagship-lcdm": {
        **_base_flagship_cosmology,
        "description": "Flagship LCDM cosmology based on Planck 2018"
    },
    "flagship-wcdm": {
        **_base_flagship_cosmology,
        # Dark energy equation of state parameters (only parameters that differ)
        "w0": -0.9,  # Slightly different from LCDM for flagship comparison
        "wa": 0.1,   # Time-varying component
        "description": "Flagship wCDM cosmology with evolving dark energy"
    },
    "flagship-phicdm": {
        **_base_flagship_cosmology,
        # Scalar field parameters (only parameters that differ)
        "phi_model": "quintessence",
        "phi_params": {
            "potential": "exponential",
            "lambda_phi": 0.1
        },
        "description": "Flagship phiCDM cosmology with scalar field dark energy"
    },
})

# --- Campaign-Specific Presets ---
# Additional preset configurations for campaign management
CAMPAIGN_PRESETS = {
    "validation": {
        # Quick validation runs with small box sizes and low resolution
        "dBoxSize": 100,
        "nGrid": 128,
        "nodes": 1,
        "timelimit": "02:00:00",
        "redshift_targets": "[3.0, 1.0, 0.0]",
        "nSteps": "[1, 1, 1]",
    },
    "flagship": {
        # High-resolution flagship simulation parameters
        "dBoxSize": 2100,
        "nGrid": 4096,
        "nodes": 64,
        "timelimit": "48:00:00",
        "gpupern": 4,
    },
    "S0_campaign": {
        # Specific parameters for the S0 campaign with July 1, 2025 deadline
        "dBoxSize": 1050,
        "nGrid": 2048,
        "nodes": 32,
        "timelimit": "36:00:00",
        "gpupern": 4,
        "email": PKDGRAVEMAIL,
        "account": "cosmos",
    },
}

# To derive ombh2 and omch2 for presets where omegam and omegab are given
for preset_name, params in COSMOLOGY_PRESETS.items():
    if "omegam" in params and "omegab" in params and "h" in params:
        if "ombh2" not in params:
            params["ombh2"] = params["omegab"] * params["h"]**2
        if "omch2" not in params:
            params["omch2"] = (params["omegam"] - params["omegab"]) * params["h"]**2

# --- Custom Exceptions ---
class PkdpipeConfigError(Exception):
    """Base exception for configuration errors in pkdpipe."""
    pass

class EnvironmentVariableError(PkdpipeConfigError):
    """Exception raised for missing or invalid environment variables."""
    pass

class DirectoryError(PkdpipeConfigError):
    """Exception raised for issues with directory creation or access."""
    pass

class TemplateError(PkdpipeConfigError):
    """Exception raised for issues with template file processing."""
    pass

class JobSubmissionError(PkdpipeConfigError):
    """Exception raised for errors during job submission."""
    pass

