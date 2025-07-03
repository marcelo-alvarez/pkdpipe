"""pkdpipe: pkdgrav3 analysis pipeline"""

__version__ = "0.1.0"

# Ensure submodules are accessible if needed, but avoid circular imports
# For example, if you want to expose specific classes/functions at the package level:
from .simulation import Simulation
from .cosmology import Cosmology
from .campaign import Campaign, CampaignConfig, SimulationVariant, CampaignStatus, SimulationStatus
from .config import PkdpipeConfigError, EnvironmentVariableError, DirectoryError, TemplateError, JobSubmissionError
from .cli import parsecommandline, create
from .campaign_cli import main as campaign_cli_main
from .data import Data
from .analysis import analyze_results, validate_power_spectrum
from .utils import find_simulation_data, setup_environment, generate_synthetic_particle_data

__all__ = [
    'Simulation',
    'Cosmology', 
    'Campaign',
    'CampaignConfig',
    'SimulationVariant',
    'CampaignStatus',
    'SimulationStatus',
    'PkdpipeConfigError',
    'EnvironmentVariableError',
    'DirectoryError', 
    'TemplateError',
    'JobSubmissionError',
    'parsecommandline',
    'create',
    'campaign_cli_main',
    'Data',
    'analyze_results',
    'validate_power_spectrum',
    'find_simulation_data',
    'setup_environment',
    'generate_synthetic_particle_data'
]
