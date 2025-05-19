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
DEFAULT_NSTEPS_STR = f'{[f"{i}" for i in range(len(_DEFAULT_REDSHIFT_LIST_VALUES))]}'.replace("'", "")


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

