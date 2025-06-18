"""
Parameter type definitions and utilities for pkdpipe.

This module provides type definitions and utilities for maintaining clear separation
between different parameter categories in the pkdpipe pipeline:
- Cosmological parameters (for Cosmology class)
- SLURM/infrastructure parameters (for job submission)
- Simulation parameters (for pkdgrav3 configuration)
"""

from typing import Dict, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum

class ParameterCategory(Enum):
    """Categories of parameters used in pkdpipe."""
    COSMOLOGICAL = "cosmological"
    SLURM = "slurm"
    SIMULATION = "simulation"
    OTHER = "other"

@dataclass
class SeparatedParameters:
    """Container for separated parameter categories."""
    cosmological: Dict[str, Any]
    slurm: Dict[str, Any]
    simulation: Dict[str, Any]
    other: Dict[str, Any]

# Define which parameters belong to each category
COSMOLOGICAL_PARAMETERS: Set[str] = {
    # Core cosmological parameters
    'cosmology', 'cosmo',  # preset name
    'h', 'H0',
    'omegam', 'omegab', 'omegac', 'omegal', 'omegak',
    'ombh2', 'omch2',
    'sigma8', 'As', 'ns',
    'mnu', 'nnu', 'numnu', 'tau', 'TCMB',
    'harchy', 'k0phys',
    # Dark energy parameters
    'w0', 'wa',
    # Scalar field parameters
    'phi_model', 'phi_params',
    # Derived/effective parameters (set by cosmology)
    'effective_h', 'effective_omegam', 'effective_omegal', 
    'effective_sigma8', 'effective_ns',
    'omegal_calculated', 'omegam_calculated', 'sigma8_final',
    'sigma8_calculated_by_camb_before_norm', 'sigma8_normalization_factor_applied'
}

SLURM_PARAMETERS: Set[str] = {
    # SLURM job submission parameters
    'account', 'partition', 'qos',
    'nodes', 'cpupert', 'gpupern', 'mem',
    'tlimit', 'time_limit', 'timelimit',
    'email', 'mail_type', 'mail_user',
    'job_name', 'jobname', 'jobname_template', 'jobname_actual',
    'output', 'error', 'stdout', 'stderr',
    'constraint', 'reservation',
    'sbatch', 'interact', 'interactive',
    # Directory parameters
    'rundir', 'scrdir', 'scratch'
}

SIMULATION_PARAMETERS: Set[str] = {
    # pkdgrav3 simulation parameters
    'simname', 'sim_preset',
    'N',  # Number of particles per dimension
    'L',  # Box size
    'nGrid', 'dBoxSize', 'boxsize',
    'dRedFrom', 'dRedTo', 'dRedshiftLCP',
    'nSteps', 'nsteps', 'iOutInterval', 'iLPT',
    'dSigma8', 'dSpectral', 'dOmega0', 'dLambda', 'dOmegaDE',
    'w0', 'wa',  # Dark energy parameters in PKDGrav format
    'achOutName', 'achTfFile',
    'bDoHalo', 'bDumpFrame', 'bDoGas', 'bDoStar', 
    'bFeedback', 'bOverwrite',
    # Random seed and output parameters
    'seed', 'random_seed',
    # Output and file parameters
    'output_dir', 'scratch_dir'
}

def categorize_parameter(param_name: str) -> ParameterCategory:
    """
    Categorize a parameter by name.
    
    Args:
        param_name: The parameter name to categorize
        
    Returns:
        ParameterCategory: The category this parameter belongs to
    """
    if param_name in COSMOLOGICAL_PARAMETERS:
        return ParameterCategory.COSMOLOGICAL
    elif param_name in SLURM_PARAMETERS:
        return ParameterCategory.SLURM
    elif param_name in SIMULATION_PARAMETERS:
        return ParameterCategory.SIMULATION
    else:
        return ParameterCategory.OTHER

def separate_parameters(params: Dict[str, Any]) -> SeparatedParameters:
    """
    Separate a mixed parameter dictionary into categorized parameters.
    
    Args:
        params: Dictionary containing mixed parameter types
        
    Returns:
        SeparatedParameters: Object with separated parameter dictionaries
    """
    separated = SeparatedParameters(
        cosmological={},
        slurm={},
        simulation={},
        other={}
    )
    
    for key, value in params.items():
        category = categorize_parameter(key)
        
        if category == ParameterCategory.COSMOLOGICAL:
            separated.cosmological[key] = value
        elif category == ParameterCategory.SLURM:
            separated.slurm[key] = value
        elif category == ParameterCategory.SIMULATION:
            separated.simulation[key] = value
        else:
            separated.other[key] = value
    
    return separated

def extract_cosmological_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only cosmological parameters from a mixed parameter dictionary.
    
    Args:
        params: Dictionary containing mixed parameter types
        
    Returns:
        Dict containing only cosmological parameters
    """
    return {k: v for k, v in params.items() if k in COSMOLOGICAL_PARAMETERS}

def extract_slurm_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only SLURM parameters from a mixed parameter dictionary.
    
    Args:
        params: Dictionary containing mixed parameter types
        
    Returns:
        Dict containing only SLURM parameters
    """
    return {k: v for k, v in params.items() if k in SLURM_PARAMETERS}

def extract_simulation_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only simulation parameters from a mixed parameter dictionary.
    
    Args:
        params: Dictionary containing mixed parameter types
        
    Returns:
        Dict containing only simulation parameters
    """
    return {k: v for k, v in params.items() if k in SIMULATION_PARAMETERS}

def merge_parameters(*param_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple parameter dictionaries, with later dictionaries taking precedence.
    
    Args:
        *param_dicts: Variable number of parameter dictionaries to merge
        
    Returns:
        Dict containing merged parameters
    """
    merged = {}
    for param_dict in param_dicts:
        merged.update(param_dict)
    return merged

def validate_parameter_separation(params: Dict[str, Any]) -> Tuple[bool, Dict[str, int]]:
    """
    Validate parameter separation and return statistics.
    
    Args:
        params: Dictionary to validate
        
    Returns:
        Tuple of (is_valid, category_counts) where category_counts shows 
        the number of parameters in each category
    """
    separated = separate_parameters(params)
    
    category_counts = {
        'cosmological': len(separated.cosmological),
        'slurm': len(separated.slurm),
        'simulation': len(separated.simulation),
        'other': len(separated.other)
    }
    
    # Parameter separation is valid if we have some cosmological and some non-cosmological
    # parameters clearly separated
    is_valid = (category_counts['cosmological'] > 0 and 
                sum(category_counts.values()) > category_counts['cosmological'])
    
    return is_valid, category_counts
