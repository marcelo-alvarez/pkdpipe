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
    'campaign_cli_main'
]
