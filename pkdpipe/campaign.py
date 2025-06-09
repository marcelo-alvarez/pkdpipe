"""
Campaign management for pkdpipe.

This module provides comprehensive campaign orchestration capabilities for managing
multiple related simulations in cosmological simulation campaigns. It supports:

- Multi-cosmology campaigns (LCDM, wCDM, phiCDM, etc.)
- Multiple resolution scales within a campaign
- Priority-based submission scheduling
- Dependency management between simulations
- Lifecycle state tracking and monitoring
- YAML-based campaign configuration
- Integration with existing Simulation and Cosmology classes

Key components:
- SimulationVariant: Individual simulation within a campaign
- CampaignConfig: Campaign-wide configuration and parameters
- Campaign: Main orchestration class with state management
- CampaignStatus/SimulationStatus: Status tracking enums
"""

import os
import json
import yaml
import subprocess
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .simulation import Simulation
from .config import (
    SIMULATION_PRESETS, 
    COSMOLOGY_PRESETS,
    DEFAULT_RUN_DIR_BASE,
    DEFAULT_SCRATCH_DIR_BASE,
    DEFAULT_JOB_NAME_TEMPLATE,
    PkdpipeConfigError
)


class CampaignStatus(Enum):
    """Campaign-level status tracking."""
    PLANNING = "planning"
    INITIALIZING = "initializing"
    SUBMITTING = "submitting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SimulationStatus(Enum):
    """Individual simulation status tracking."""
    NOT_SUBMITTED = "not_submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CampaignError(PkdpipeConfigError):
    """Base exception for campaign-related errors."""
    pass


@dataclass
class SimulationVariant:
    """
    Configuration for a single simulation variant within a campaign.
    
    Attributes:
        name: Unique identifier for this simulation variant
        cosmology: Cosmology preset name from COSMOLOGY_PRESETS
        resolution: Simulation resolution preset from SIMULATION_PRESETS
        priority: Priority for submission (higher = more important)
        dependencies: List of variant names that must complete before this one
        submission_deadline: Optional deadline for job submission
        custom_params: Additional parameters specific to this variant
        comment: Optional description/notes
    """
    name: str
    cosmology: str
    resolution: str
    priority: int = 50
    dependencies: List[str] = field(default_factory=list)
    submission_deadline: Optional[datetime] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)
    comment: Optional[str] = None
    
    def __post_init__(self):
        """Validate variant configuration."""
        if self.cosmology not in COSMOLOGY_PRESETS:
            raise ValueError(f"Unknown cosmology preset: {self.cosmology}")
        if self.resolution not in SIMULATION_PRESETS:
            raise ValueError(f"Unknown simulation preset: {self.resolution}")


@dataclass
class CampaignConfig:
    """
    Configuration for a multi-simulation campaign.
    
    Attributes:
        name: Campaign identifier
        description: Human-readable description
        variants: List of simulation variants to execute
        base_params: Base parameters applied to all simulations
        output_dir: Optional custom output directory
        scratch_dir: Optional custom scratch directory  
        max_concurrent_jobs: Maximum simultaneous submissions
        global_deadline: Optional campaign-wide deadline
    """
    name: str
    description: str
    variants: List[SimulationVariant]
    base_params: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[str] = None
    scratch_dir: Optional[str] = None
    max_concurrent_jobs: int = 10
    global_deadline: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate campaign configuration."""
        if not self.variants:
            raise ValueError("Campaign must have at least one simulation variant")
        
        # Check for duplicate variant names
        names = [v.name for v in self.variants]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate variant names found")
    
    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'CampaignConfig':
        """Load campaign configuration from YAML file."""
        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Campaign config file not found: {yaml_file}")
            
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert variant dictionaries to SimulationVariant objects
        variants = []
        for variant_data in data.pop('variants', []):
            # Handle datetime parsing for deadlines
            if 'submission_deadline' in variant_data and variant_data['submission_deadline']:
                variant_data['submission_deadline'] = datetime.fromisoformat(
                    variant_data['submission_deadline']
                )
            variants.append(SimulationVariant(**variant_data))
        
        # Handle global deadline
        if 'global_deadline' in data and data['global_deadline']:
            data['global_deadline'] = datetime.fromisoformat(data['global_deadline'])
        
        return cls(variants=variants, **data)


class Campaign:
    """
    Main campaign management class for orchestrating multiple simulations.
    
    This class provides comprehensive campaign management including:
    - Simulation variant initialization and configuration
    - Priority-based job submission scheduling  
    - Dependency tracking and resolution
    - State persistence and recovery
    - Status monitoring and reporting
    - Integration with SLURM scheduler
    """
    
    def __init__(self, config: Union[str, CampaignConfig]):
        """
        Initialize campaign from config file or object.
        
        Args:
            config: Path to YAML config file or CampaignConfig object
        """
        if isinstance(config, str):
            self.config = CampaignConfig.from_yaml(config)
            self.config_file = config
        else:
            self.config = config
            self.config_file = None
        
        # Initialize state tracking
        self.status = CampaignStatus.PLANNING
        self.simulations: Dict[str, Simulation] = {}
        self.job_ids: Dict[str, str] = {}  # variant_name -> SLURM job ID
        self.simulation_status: Dict[str, SimulationStatus] = {}
        
        # Setup campaign directories
        self._setup_directories()
        
        # Initialize simulations for each variant
        self._initialize_simulations()
        
        # Save initial state
        self._save_state()
    
    def _setup_directories(self):
        """Setup campaign directory structure and load existing state."""
        if self.config.output_dir:
            self.output_dir = Path(self.config.output_dir)
        else:
            self.output_dir = Path(DEFAULT_RUN_DIR_BASE) / f"campaign-{self.config.name}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output_dir / "campaign_state.json"
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Create runs directory for individual simulation directories
        self.runs_dir = self.output_dir / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        
        # Copy the original config file to the campaign directory if we have one
        if self.config_file and Path(self.config_file).exists():
            import shutil
            config_dest = self.output_dir / Path(self.config_file).name
            if not config_dest.exists():
                shutil.copy2(self.config_file, config_dest)
        
        # Load existing state if available
        self._load_state()
    
    def _initialize_simulations(self):
        """Initialize Simulation objects for each variant."""
        for variant in self.config.variants:
            # Start with base campaign parameters
            params = {**self.config.base_params}
            
            # Add resolution-specific parameters from presets
            if variant.resolution in SIMULATION_PRESETS:
                params.update(SIMULATION_PRESETS[variant.resolution])
            
            # Apply variant-specific custom parameters (highest priority)
            params.update(variant.custom_params)
            
            # Set cosmology and simulation name
            params['cosmo'] = variant.cosmology
            params['simname'] = variant.resolution  # Use resolution preset name for proper preset lookup
            
            # Set required simulation parameters
            # For campaigns, use the variant name as the job name instead of the default template
            params['jobname_template'] = variant.name
            
            # Configure output directories
            if self.config.output_dir:
                params['rundir'] = str(self.output_dir / "runs")
            if self.config.scratch_dir:
                params['scrdir'] = str(Path(self.config.scratch_dir))
                params['scratch'] = True
            
            # Create simulation instance
            try:
                sim = Simulation(params=params)
                self.simulations[variant.name] = sim
                
                # Initialize status if not already set
                if variant.name not in self.simulation_status:
                    self.simulation_status[variant.name] = SimulationStatus.NOT_SUBMITTED
                    
            except Exception as e:
                raise CampaignError(f"Failed to initialize simulation '{variant.name}': {e}")
    
    def _load_state(self):
        """Load campaign state from persistent storage."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.status = CampaignStatus(state.get('status', 'planning'))
                self.job_ids = state.get('job_ids', {})
                
                # Convert status strings back to enums
                status_data = state.get('simulation_status', {})
                self.simulation_status = {
                    name: SimulationStatus(status) 
                    for name, status in status_data.items()
                }
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Warning: Could not load campaign state: {e}")
                # Continue with fresh state
    
    def _save_state(self):
        """Save campaign state to persistent storage."""
        state = {
            'status': self.status.value,
            'job_ids': self.job_ids,
            'simulation_status': {
                name: status.value 
                for name, status in self.simulation_status.items()
            },
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save campaign state: {e}")
    
    def get_submittable_variants(self) -> List[SimulationVariant]:
        """
        Get list of variants ready for submission based on dependencies and status.
        
        Returns:
            List of variants that can be submitted now
        """
        submittable = []
        
        for variant in self.config.variants:
            # Skip if already submitted
            current_status = self.simulation_status.get(variant.name, SimulationStatus.NOT_SUBMITTED)
            if current_status != SimulationStatus.NOT_SUBMITTED:
                continue
            
            # Check if deadline has passed
            if variant.submission_deadline and datetime.now() > variant.submission_deadline:
                print(f"Warning: Submission deadline passed for {variant.name}")
                continue
            
            # Check dependencies
            dependencies_met = True
            for dep_name in variant.dependencies:
                dep_status = self.simulation_status.get(dep_name, SimulationStatus.NOT_SUBMITTED)
                if dep_status != SimulationStatus.COMPLETED:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                submittable.append(variant)
        
        # Sort by priority (higher priority first)
        return sorted(submittable, key=lambda v: -v.priority)
    
    def submit_variant(self, variant_name: str, dry_run: bool = False, no_submit: bool = False) -> bool:
        """
        Submit a specific simulation variant.
        
        Args:
            variant_name: Name of variant to submit
            dry_run: If True, show what would be submitted without doing anything
            no_submit: If True, create files and directories but do not submit to SLURM
        
        Returns:
            True if submission successful
        """
        if variant_name not in self.simulations:
            raise ValueError(f"Unknown variant: {variant_name}")
        
        sim = self.simulations[variant_name]
        
        try:
            if dry_run:
                print(f"DRY RUN: Would submit variant '{variant_name}'")
                return True
            elif no_submit:
                # Create files and directories but don't submit to SLURM
                # Set sbatch to False so sim.create() doesn't submit
                sim.params['sbatch'] = False
                sim.create()
                
                # Update status to indicate files are prepared
                self.simulation_status[variant_name] = SimulationStatus.NOT_SUBMITTED
                
                print(f"✓ Files created for variant '{variant_name}' (not submitted to SLURM)")
                self._save_state()
                return True
            else:
                # Set sbatch to True for automatic submission
                sim.params['sbatch'] = True
                job_id = sim.create()
                
                # Store job ID if submission was successful
                if job_id:
                    self.job_ids[variant_name] = job_id
                
                # Update status
                self.simulation_status[variant_name] = SimulationStatus.QUEUED
                
                print(f"Successfully submitted variant '{variant_name}'")
                self._save_state()
                return True
                
        except Exception as e:
            print(f"Failed to submit variant '{variant_name}': {e}")
            self.simulation_status[variant_name] = SimulationStatus.FAILED
            self._save_state()
            return False
    
    def submit_ready_variants(self, max_submissions: Optional[int] = None, dry_run: bool = False, no_submit: bool = False) -> int:
        """
        Submit all variants that are ready for submission.
        
        Args:
            max_submissions: Maximum number of variants to submit (None for unlimited)
            dry_run: If True, show what would be submitted without doing anything
            no_submit: If True, create files and directories but do not submit to SLURM
        
        Returns:
            Number of variants successfully submitted
        """
        submittable = self.get_submittable_variants()
        if max_submissions:
            submittable = submittable[:max_submissions]
        
        submitted_count = 0
        for variant in submittable:
            if self.submit_variant(variant.name, dry_run=dry_run, no_submit=no_submit):
                submitted_count += 1
            
            # Check concurrent job limit (only applies to actual submissions)
            if not dry_run and not no_submit and self._get_running_job_count() >= self.config.max_concurrent_jobs:
                print(f"Reached maximum concurrent job limit ({self.config.max_concurrent_jobs})")
                break
        
        return submitted_count
    
    def _get_running_job_count(self) -> int:
        """Get count of currently running/queued jobs."""
        running_statuses = {SimulationStatus.QUEUED, SimulationStatus.RUNNING}
        return sum(1 for status in self.simulation_status.values() if status in running_statuses)
    
    def update_status(self):
        """Update simulation status by querying SLURM."""
        for variant_name, job_id in self.job_ids.items():
            if not job_id or job_id == "N/A":
                continue
            
            try:
                # Query SLURM for job status
                result = subprocess.run(
                    ["squeue", "-j", job_id, "-h", "-o", "%T"],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    slurm_status = result.stdout.strip()
                    
                    # Map SLURM status to our enum
                    status_mapping = {
                        'PENDING': SimulationStatus.QUEUED,
                        'PD': SimulationStatus.QUEUED,
                        'RUNNING': SimulationStatus.RUNNING,
                        'R': SimulationStatus.RUNNING,
                        'COMPLETED': SimulationStatus.COMPLETED,
                        'CD': SimulationStatus.COMPLETED,
                        'FAILED': SimulationStatus.FAILED,
                        'F': SimulationStatus.FAILED,
                        'TIMEOUT': SimulationStatus.FAILED,
                        'TO': SimulationStatus.FAILED,
                        'CANCELLED': SimulationStatus.CANCELLED,
                        'CA': SimulationStatus.CANCELLED,
                    }
                    
                    new_status = status_mapping.get(slurm_status)
                    if new_status and new_status != self.simulation_status[variant_name]:
                        self.simulation_status[variant_name] = new_status
            
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                print(f"Warning: Could not query status for job {job_id}: {e}")
        
        self._save_state()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get campaign summary statistics."""
        status_counts = {}
        for status in SimulationStatus:
            status_counts[status.value] = sum(
                1 for s in self.simulation_status.values() if s == status
            )
        
        total = len(self.config.variants)
        completed = status_counts.get('completed', 0)
        
        return {
            'name': self.config.name,
            'status': self.status.value,
            'total_variants': total,
            'completed': completed,
            'running': status_counts.get('running', 0),
            'queued': status_counts.get('queued', 0),
            'failed': status_counts.get('failed', 0),
            'not_submitted': status_counts.get('not_submitted', 0),
            'cancelled': status_counts.get('cancelled', 0),
            'completion_rate': completed / total if total > 0 else 0.0,
            'status_counts': status_counts
        }
    
    def generate_report(self) -> str:
        """Generate a detailed text report of campaign status."""
        summary = self.get_summary()
        
        report = []
        report.append(f"Campaign: {self.config.name}")
        report.append("=" * (len(self.config.name) + 10))
        report.append(f"Description: {self.config.description}")
        report.append(f"Status: {summary['status']}")
        report.append(f"Progress: {summary['completed']}/{summary['total_variants']} "
                     f"({summary['completion_rate']:.1%})")
        report.append("")
        
        # Status breakdown
        report.append("Status Breakdown:")
        report.append(f"  Completed:     {summary['completed']:3d}")
        report.append(f"  Running:       {summary['running']:3d}")
        report.append(f"  Queued:        {summary['queued']:3d}")
        report.append(f"  Not Submitted: {summary['not_submitted']:3d}")
        report.append(f"  Failed:        {summary['failed']:3d}")
        report.append(f"  Cancelled:     {summary['cancelled']:3d}")
        report.append("")
        
        # Individual variant status
        report.append("Simulation Variants:")
        report.append("-" * 90)
        header = f"{'Name':<20} {'Cosmology':<15} {'Resolution':<15} {'Status':<12} {'Priority':<8} {'Job ID':<12}"
        report.append(header)
        report.append("-" * 90)
        
        for variant in sorted(self.config.variants, key=lambda v: -v.priority):
            status = self.simulation_status.get(variant.name, SimulationStatus.NOT_SUBMITTED)
            job_id = self.job_ids.get(variant.name, "N/A")
            
            row = f"{variant.name:<20} {variant.cosmology:<15} {variant.resolution:<15} " \
                  f"{status.value:<12} {variant.priority:<8} {job_id:<12}"
            report.append(row)
        
        return "\n".join(report)
    
    def list_variants(self, status_filter: Optional[SimulationStatus] = None) -> List[SimulationVariant]:
        """
        List campaign variants, optionally filtered by status.
        
        Args:
            status_filter: Optional status to filter by
        
        Returns:
            List of variants matching filter
        """
        variants = []
        for variant in self.config.variants:
            if status_filter is None:
                variants.append(variant)
            else:
                current_status = self.simulation_status.get(variant.name, SimulationStatus.NOT_SUBMITTED)
                if current_status == status_filter:
                    variants.append(variant)
        
        return sorted(variants, key=lambda v: -v.priority)
    
    def cancel_variant(self, variant_name: str) -> bool:
        """
        Cancel a submitted variant.
        
        Args:
            variant_name: Name of variant to cancel
        
        Returns:
            True if cancellation successful
        """
        if variant_name not in self.job_ids:
            print(f"No job ID found for variant '{variant_name}'")
            return False
        
        job_id = self.job_ids[variant_name]
        if not job_id or job_id == "N/A":
            print(f"Invalid job ID for variant '{variant_name}'")
            return False
        
        try:
            result = subprocess.run(
                ["scancel", job_id],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                self.simulation_status[variant_name] = SimulationStatus.CANCELLED
                print(f"Successfully cancelled variant '{variant_name}' (job {job_id})")
                self._save_state()
                return True
            else:
                print(f"Failed to cancel job {job_id}: {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            print(f"Error cancelling job {job_id}: {e}")
            return False
