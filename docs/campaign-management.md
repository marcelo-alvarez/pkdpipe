# Campaign Management Documentation

## Overview

The pkdpipe campaign management system provides comprehensive orchestration capabilities for managing multiple related simulations in cosmological simulation campaigns. This system is designed to support multi-cosmology campaigns (such as LCDM, wCDM, phiCDM) with multiple resolution scales, priority-based submission scheduling, and dependency management.

## Key Features

- **Multi-cosmology support**: LCDM, wCDM, phiCDM, and custom cosmologies
- **Multiple resolution scales**: Validation, scaling, and production runs within a campaign
- **Priority-based scheduling**: Higher priority simulations submitted first
- **Dependency management**: Automatic dependency resolution between simulations
- **Lifecycle state tracking**: Track simulation progress through defined stages
- **YAML-based configuration**: Human-readable campaign definitions
- **CLI interface**: Command-line tools for campaign management
- **State persistence**: Automatic state saving and recovery

## Campaign Configuration

### YAML Configuration Format

```yaml
name: "my-campaign"
description: "Description of the campaign"

# Global settings
max_concurrent_jobs: 6
global_deadline: "2025-07-01T23:59:59"

# Base parameters applied to all simulations
base_params:
  bDoHalo: 1
  account: "cosmosim"
  email: "user@example.com"

# Output directories
output_dir: "/path/to/output"
scratch_dir: "/path/to/scratch"

# Simulation variants
variants:
  - name: "lcdm-validation"
    cosmology: "lcdm"
    resolution: "S0-validation"
    priority: 100
    dependencies: []
    submission_deadline: "2025-06-10T23:59:59"
    custom_params:
      dRedTo: "[3.0, 1.0, 0.0]"
      nSteps: "[1, 1, 1]"
```

### Available Cosmology Presets

- **lcdm**: Standard LCDM cosmology based on DESI-DR2-Planck-ACT
- **wcdm**: wCDM with evolving dark energy (w0=-0.9, wa=0.1)
- **phicdm**: phiCDM with scalar field dark energy
- **flagship-lcdm**: LCDM cosmology based on Planck 2018
- **flagship-wcdm**: wCDM with evolving dark energy (w0=-0.9, wa=0.1)
- **flagship-phicdm**: phiCDM with scalar field dark energy
- **desi-dr2-planck-act-mnufree**: DESI DR2 cosmology
- **euclid-flagship**: Euclid flagship cosmology
- **planck18**: Planck 2018 cosmology

### Available Simulation Presets

#### S0 Campaign Presets
- **S0-validation**: 1050 Mpc/h box, 1400³ grid, 2 nodes, 12h
- **S0-scaling**: 2100 Mpc/h box, 2800³ grid, 16 nodes, 24h
- **S0-highres**: 3150 Mpc/h box, 4200³ grid, 54 nodes, 36h
- **S0-production**: 5250 Mpc/h box, 7000³ grid, 250 nodes, 48h

#### Legacy Presets
- **validation**: 1050 Mpc/h box, 1400³ grid, 2 nodes, 12h (deprecated, use S0-validation)
- **scaling**: 2100 Mpc/h box, 2800³ grid, 16 nodes, 24h (deprecated, use S0-scaling)
- **highres**: 3150 Mpc/h box, 4200³ grid, 54 nodes, 36h (deprecated, use S0-highres)
- **production**: 5250 Mpc/h box, 7000³ grid, 250 nodes, 48h (deprecated, use S0-production)

#### Light Cone Presets
- **lcone-small**: 525 Mpc/h box, 700³ grid, 1 node
- **lcone-medium**: 1050 Mpc/h box, 1400³ grid, 2 nodes
- **lcone-large**: 2100 Mpc/h box, 2800³ grid, 16 nodes

## Command Line Interface

### Installation

After installing pkdpipe, the campaign CLI is available as `pkdpipe-campaign`:

```bash
pip install .
```

### Basic Usage

#### Create a campaign

```bash
# Validate configuration only
pkdpipe-campaign create campaigns/my-campaign.yaml --validate-only

# Create the campaign
pkdpipe-campaign create campaigns/my-campaign.yaml
```

#### Submit simulations

```bash
# Submit all ready variants
pkdpipe-campaign submit /path/to/campaign/dir

# Submit a specific variant
pkdpipe-campaign submit /path/to/campaign/dir --variant lcdm-S0-validation

# Dry run (show what would be submitted without doing anything)
pkdpipe-campaign submit /path/to/campaign/dir --variant lcdm-S0-validation --dry-run

# No submit (create files and directories but don't submit to SLURM)
pkdpipe-campaign submit /path/to/campaign/dir --variant lcdm-S0-validation --no-submit

# Limit number of submissions
pkdpipe-campaign submit /path/to/campaign/dir --max-submissions 2
```

#### Check status

```bash
# Get current status
pkdpipe-campaign status /path/to/campaign/dir

# Status without updating from SLURM
pkdpipe-campaign status /path/to/campaign/dir --no-update

# Watch mode (update every 300 seconds)
pkdpipe-campaign status /path/to/campaign/dir --watch 300
```

#### List campaigns

```bash
# List campaigns in current directory
pkdpipe-campaign list

# List campaigns in specific directory
pkdpipe-campaign list --search-dir /path/to/campaigns
```

## Python API

### Basic Usage

```python
from pkdpipe.campaign import Campaign, CampaignConfig, SimulationVariant

# Load campaign from YAML
campaign = Campaign("path/to/campaign.yaml")

# Get campaign status
summary = campaign.get_summary()
print(f"Completed: {summary['completed']}/{summary['total_variants']}")

# Submit ready variants
submitted = campaign.submit_ready_variants(max_submissions=3)
print(f"Submitted {submitted} variants")

# Update status from SLURM
campaign.update_status()

# Generate detailed report
report = campaign.generate_report()
print(report)
```

### Creating Campaigns Programmatically

```python
from pkdpipe.campaign import CampaignConfig, SimulationVariant

# Define variants
variants = [
    SimulationVariant(
        name="lcdm-test",
        cosmology="flagship-lcdm",
        resolution="flagship-validation",
        priority=100
    ),
    SimulationVariant(
        name="wcdm-test",
        cosmology="flagship-wcdm", 
        resolution="flagship-validation",
        priority=90,
        dependencies=["lcdm-test"]
    )
]

# Create configuration
config = CampaignConfig(
    name="test-campaign",
    description="Test campaign",
    variants=variants,
    max_concurrent_jobs=5
)

# Create campaign
campaign = Campaign(config)
```

## Directory Structure

When a campaign is created, the following directory structure is generated:

```
campaign-output-dir/
├── campaign_state.json          # Persistent campaign state
├── logs/                        # Campaign logs
├── runs/                        # Individual simulation directories
│   ├── lcdm-validation/
│   ├── wcdm-validation/
│   └── ...
└── my-campaign.yaml            # Original configuration (copied)
```

## Campaign State Management

The campaign system automatically tracks and persists state in `campaign_state.json`:

```json
{
  "status": "running",
  "job_ids": {
    "lcdm-validation": "12345678",
    "wcdm-validation": "12345679"
  },
  "simulation_status": {
    "lcdm-validation": "completed",
    "wcdm-validation": "running"
  },
  "last_updated": "2025-06-03T14:30:00"
}
```

## Status Tracking

### Campaign Status
- **planning**: Campaign created but no submissions yet
- **initializing**: Campaign being set up
- **submitting**: Simulations being submitted
- **running**: Simulations are executing
- **completed**: All simulations finished successfully
- **failed**: Campaign failed due to critical errors
- **cancelled**: Campaign was cancelled

### Simulation Status
- **not_submitted**: Simulation not yet submitted
- **queued**: Submitted to SLURM and waiting in queue
- **running**: Currently executing
- **completed**: Finished successfully
- **failed**: Failed during execution
- **cancelled**: Cancelled by user or system

## Dependency Management

Simulations can depend on others completing successfully:

```yaml
variants:
  - name: "base-simulation"
    # ... other parameters
    dependencies: []  # No dependencies
    
  - name: "dependent-simulation"
    # ... other parameters
    dependencies: ["base-simulation"]  # Waits for base-simulation
```

The campaign system automatically:
- Determines which simulations are ready to submit
- Submits simulations in dependency order
- Only submits dependent simulations after dependencies complete

## Priority Scheduling

Higher priority simulations are submitted first:

```yaml
variants:
  - name: "critical-simulation"
    priority: 100  # High priority
    
  - name: "optional-simulation"
    priority: 50   # Lower priority
```

## Best Practices

### Campaign Design
1. **Start with validation runs**: Use small-scale validation runs to test parameters
2. **Use scaling tests**: Test resource requirements before production runs
3. **Set realistic deadlines**: Allow buffer time for failures and reruns
4. **Organize by priority**: Critical simulations should have higher priority

### Resource Management
1. **Set appropriate max_concurrent_jobs**: Don't overwhelm the scheduler
2. **Use scratch directories**: Large simulations benefit from fast scratch storage
3. **Monitor queue limits**: Be aware of user and group queue limits

### Error Handling
1. **Regular status updates**: Check campaign status frequently
2. **Monitor logs**: Watch for early warning signs of problems
3. **Have fallback plans**: Prepare alternative strategies for critical deadlines

## Troubleshooting

### Common Issues

**Campaign creation fails**
- Check YAML syntax
- Verify cosmology and resolution presets exist
- Ensure output directories are writable

**Submissions fail**
- Check SLURM configuration
- Verify account and partition settings
- Ensure sufficient queue limits

**Dependencies not resolving**
- Check for circular dependencies
- Verify dependency names match variant names exactly
- Check simulation completion status

**Status updates not working**
- Ensure SLURM commands (squeue, scancel) are available
- Check network connectivity to scheduler
- Verify job IDs are correct

### Getting Help

1. **Validate configuration**: Use `--validate-only` flag
2. **Check logs**: Look in the campaign `logs/` directory  
3. **Use dry run**: Test submissions with `--dry-run`
4. **Check status**: Use `--no-update` to see cached status

## Example: S0 Campaign

The included `cosmosim-mocks-2025.yaml` demonstrates a complete multi-cosmology campaign with:

- **3 cosmologies**: LCDM, wCDM, phiCDM
- **Multiple scales**: Validation, scaling, and production runs
- **Strategic scheduling**: Critical July 1, 2025 deadline management
- **Resource optimization**: Conservative concurrent job limits
- **Dependency chains**: Validation → Scaling → Production workflow

This serves as a template for similar large-scale cosmological simulation campaigns.
