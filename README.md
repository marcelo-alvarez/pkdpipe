# pkdpipe

**pkdpipe** is a Python library designed to streamline the process of working with N-body cosmological simulation data, particularly from PKDGrav3. It provides robust tools for data handling, analysis, and visualization with a focus on performance and reliability.

## Production Status: READY

**pkdpipe v2.0** is **production-ready** for large-scale cosmological simulation analysis on HPC systems like Perlmutter. The complete pipeline has been validated end-to-end with real simulation data.

## Features

*   **Production-Ready Pipeline:** Complete end-to-end validation with 92GB real simulation data (2.744B particles)
*   **Enhanced Data Interface:** Refactored data module with improved error handling, modular architecture, and comprehensive validation
*   **Robust Parameter Parsing:** Advanced parsing of PKDGrav3 parameter files with support for complex formats and validation
*   **Distributed Processing:** MPI-coordinated processing across multiple nodes with SLURM integration
*   **Lightcone and Snapshot Modes:** Support for both lightcone particle data and snapshot-based analysis
*   **Multiple File Formats:** Support for various PKDGrav3 output formats (LCP, TPS, FOF)
*   **Cosmology Calculations:** Built-in functions for common cosmological calculations
*   **Production Power Spectrum Analysis:** Memory-optimized, GPU-accelerated power spectrum calculation with distributed FFT
*   **Multi-GPU Support:** Distributed FFT computations and power spectrum calculations across multiple GPUs
*   **Memory Optimizations:** Advanced memory management achieving 50%+ memory savings through optimized MPI redistribution
*   **Parameter Type System:** Comprehensive categorization of cosmological, SLURM, and simulation parameters
*   **JAX Accelerated FFT:** Efficient Fast Fourier Transforms using JAX with distributed computing support and clean multiprocessing architecture
*   **MPI-Aware Testing:** Comprehensive test infrastructure supporting both serial and distributed functionality
*   **Command-Line Interface:** Access to functionalities via a CLI
*   **HPC Optimization:** Designed and validated for Perlmutter GPU nodes with 256GB RAM and 4x A100 GPUs

## Installation

Currently, `pkdpipe` can be installed from the local directory:

```bash
pip install .
```

## Quick Start

### Campaign Management

```python
from pkdpipe.campaign import Campaign, SimulationVariant, CampaignConfig

# Create a new campaign
config = CampaignConfig(
    name="cosmosim-mocks-2025",
    base_dir="/path/to/campaigns",
    description="Cosmological mock catalog generation for 2025"
)

campaign = Campaign(config)

# Add simulation variants
variants = [
    SimulationVariant("lcdm-validation", {"omegam": 0.315, "sigma8": 0.811}),
    SimulationVariant("wcdm-validation", {"omegam": 0.315, "w0": -0.9}),
    SimulationVariant("neff-validation", {"omegam": 0.315, "neff": 3.2})
]

for variant in variants:
    campaign.add_variant(variant)

# Initialize campaign structure
campaign.initialize()

# Submit simulations to SLURM
campaign.submit_all()

# Monitor campaign progress
status = campaign.get_status()
print(f"Campaign: {status.completed}/{status.total} simulations completed")
```

### Individual Simulation Setup

```python
from pkdpipe.simulation import Simulation
from pkdpipe.config import get_preset_params

# Create simulation with preset parameters
sim_params = get_preset_params("S0-validation")
sim_params.update({
    "omegam": 0.315,
    "sigma8": 0.811,
    "ngrid": 512,
    "boxsize": 1000.0  # Mpc/h
})

simulation = Simulation(
    name="test-simulation",
    campaign_dir="/path/to/campaign",
    parameters=sim_params
)

# Setup directories and configuration files
simulation.setup()

# Generate SLURM submission script
simulation.generate_slurm_script(
    partition="gpu",
    time="24:00:00",
    nodes=4,
    email="user@example.com"
)

# Submit to queue
job_id = simulation.submit()
print(f"Submitted job {job_id}")
```

### Command Line Interface

```bash
# Create a new simulation
pkdpipe-create --name my-simulation --preset S0-validation --omegam 0.315

# Create and manage campaigns
pkdpipe-campaign create --name cosmosim-2025 --description "Mock catalogs"
pkdpipe-campaign add-variant --campaign cosmosim-2025 --name lcdm-base
pkdpipe-campaign submit --campaign cosmosim-2025
pkdpipe-campaign status --campaign cosmosim-2025

# Submit individual simulation
pkdpipe-create --name test-run --submit --partition gpu --time 12:00:00
```

### Basic Data Reading

```python
from pkdpipe.data import Data

# Initialize data interface
# Note: For large files, use nproc=1 for multiprocessing-safe operation
pkdata = Data(param_file="simulation.par", nproc=8, verbose=True)

# Define bounding box [xmin,xmax], [ymin,ymax], [zmin,zmax]
bbox = [[-1, 1], [-1, 1], [-0.5, 0.5]]

# Fetch particle data (lightcone mode)
particles = pkdata.fetch_data(
    bbox=bbox,
    dataset='xvp',      # x, velocity, phi (gravitational potential)
    filetype='lcp',     # lightcone particle format
    lightcone=True
)

# Access particle properties
x = particles[0]['x']  # positions
vx = particles[0]['vx']  # velocities
phi = particles[0]['phi']  # gravitational potential

print(f"Found {len(x)} particles")
```

### Halo Analysis

```python
from pkdpipe.data import Data

# Initialize with halo data
pkdata = Data(param_file="simulation.par", nproc=16)

# Read halo catalog
bbox = [[-1, 1], [-1, 1], [-0.001, 0.001]]
halos = pkdata.fetch_data(
    bbox=bbox,
    dataset='xvh',      # x, velocity, halo properties
    filetype='fof',     # friends-of-friends halo format
    lightcone=False,
    redshifts=[0.0]     # snapshot at z=0
)

# Calculate halo masses
boxsize = pkdata.params.boxsize / 1e3  # Gpc/h
pmass = 2.775e20 * pkdata.params.omegam * boxsize**3 / pkdata.params.ngrid**3
halo_masses = halos['box0']['npart'] * pmass  # Msun/h

print(f"Found {len(halo_masses)} halos")
print(f"Most massive halo: {halo_masses.max():.2e} Msun/h")
```

### Snapshot Mode Analysis

```python
# Fetch data at multiple redshifts
redshifts = [0.0, 0.5, 1.0, 2.0]
snapshot_data = pkdata.fetch_data(
    bbox=bbox,
    dataset='xvp',
    filetype='tps',     # tipsy format
    lightcone=False,
    redshifts=redshifts
)

# Access data for each redshift
for i, z in enumerate(redshifts):
    box_data = snapshot_data[f'box{i}']
    print(f"z={z}: {len(box_data)} particles")
```

### Power Spectrum Analysis

```python
from pkdpipe.data import Data
from pkdpipe.power_spectrum import PowerSpectrumCalculator

# Initialize data interface
# Important: JAX is automatically imported only when FFT operations are needed
pkdata = Data(param_file="simulation.par", nproc=8)

# Fetch particle data
bbox = [[-500, 500], [-500, 500], [-500, 500]]  # Mpc/h
particles = pkdata.fetch_data(
    bbox=bbox,
    dataset='xv',
    filetype='tps',
    lightcone=False,
    redshifts=[0.0]
)

# Create particle dictionary for power spectrum calculation
particle_data = {
    'x': particles['box0']['x'],
    'y': particles['box0']['y'], 
    'z': particles['box0']['z'],
    'mass': np.ones(len(particles['box0']['x']))  # Unit mass
}

# Calculate power spectrum
# JAX initialization happens automatically during FFT operations
calculator = PowerSpectrumCalculator(ngrid=256, box_size=1000.0)
k_bins, power, n_modes = calculator.calculate_power_spectrum(
    particle_data, 
    subtract_shot_noise=True,
    assignment='cic'  # Cloud-in-cell assignment
)

print(f"Power spectrum calculated with {len(k_bins)} k-bins")
print(f"k range: {k_bins[0]:.3f} to {k_bins[-1]:.3f} h/Mpc")
```

### Multi-GPU Power Spectrum Calculation

```python
# For large simulations, use multi-GPU acceleration
# JAX distributed mode is automatically configured from SLURM environment
calculator = PowerSpectrumCalculator(
    ngrid=512, 
    box_size=2000.0, 
    n_devices=4  # Use 4 GPUs
)

# Custom k-binning for high-resolution analysis
k_bins_custom = np.logspace(-2, 1, 41)  # 40 logarithmic bins
calculator = PowerSpectrumCalculator(
    ngrid=512,
    box_size=2000.0,
    k_bins=k_bins_custom,
    n_devices=4
)

k_bins, power, n_modes = calculator.calculate_power_spectrum(
    particle_data,  # Dictionary format with x, y, z, mass
    subtract_shot_noise=True,
    assignment='cic'
)
```

### Distributed Processing with SLURM

```python
# For distributed computing on HPC systems
# Example SLURM command:
# srun -n 4 -c 32 python power_spectrum_analysis.py --ngrid 1024

# The pipeline automatically detects distributed mode from SLURM environment
# JAX distributed initialization happens only during FFT operations
calculator = PowerSpectrumCalculator(ngrid=1024, box_size=2000.0)
k_bins, power, n_modes = calculator.calculate_power_spectrum(
    particle_data,
    assignment='cic'
)
```

## Data Module Architecture

The refactored data module features a modular architecture for better maintainability:

### Core Classes

- **`Data`**: Main interface for PKDGrav3 data access
- **`ParameterParser`**: Handles parameter file parsing and validation
- **`DataProcessor`**: Manages data filtering, spatial culling, and transformations
- **`FileReader`**: Handles file I/O operations with error handling
- **`SimulationParams`**: Type-safe container for simulation parameters
- **`PowerSpectrumCalculator`**: FFT-based power spectrum analysis with multi-GPU support
- **`ParticleGridder`**: Efficient particle-to-grid assignment with multiple schemes
- **`SeparatedParameters`**: Type-safe parameter categorization system

### Key Improvements

- **Enhanced Error Handling**: Comprehensive validation and informative error messages
- **Robust Parameter Parsing**: Support for complex parameter formats including arrays
- **Structured Logging**: Replace print statements with proper logging levels
- **Type Safety**: Full type hints and dataclasses for better code reliability
- **Performance**: Optimized data processing with better memory management
- **Multi-GPU Computing**: Distributed FFT and power spectrum calculations across multiple devices
- **Clean Architecture**: JAX initialization separated from multiprocessing to prevent deadlocks
- **Parameter Organization**: Clear separation of cosmological, SLURM, and simulation parameters
- **Advanced Analytics**: Built-in power spectrum analysis with shot noise correction and k-binning
- **Distributed Processing**: SLURM-aware coordination for HPC environments
- **Backward Compatibility**: Maintains API compatibility with legacy code

## Architecture and Execution Flow

pkdpipe uses a clean three-phase architecture designed to prevent multiprocessing/threading conflicts:

### Phase 1: Data Loading (CPU, Multiprocessing)
- Pure NumPy and Python multiprocessing operations
- **No JAX imports** - prevents CUDA initialization conflicts
- Efficient I/O with configurable worker processes
- SLURM-aware process coordination

### Phase 2: Particle Gridding (CPU, Memory-Bound)
- CPU-based particle-to-grid assignment (NGP, CIC, TSC)
- Optimized for large memory (256GB RAM) on HPC systems
- **No JAX dependencies** - maintains multiprocessing safety
- Produces NumPy density grids ready for FFT transfer

### Phase 3: FFT Operations (GPU, Compute-Intensive)
- **Single point of JAX initialization** in the `fft()` function
- NumPy → JAX array conversion for GPU processing
- Distributed JAX FFT across multiple GPUs
- Automatic SLURM environment detection and coordination

This architecture ensures:
- **No deadlocks**: JAX initialization happens after all multiprocessing completes
- **Memory efficiency**: Keep particles on CPU (256GB) vs GPU (40GB per device)
- **Clean separation**: Each phase uses optimal hardware and libraries
- **HPC compatibility**: Designed for SLURM-managed distributed computing

## Workflow Management

pkdpipe provides comprehensive tools for managing large-scale simulation campaigns:

### Campaign Lifecycle

1. **Planning Phase**
   ```python
   # Define campaign with multiple variants
   campaign = Campaign(config)
   campaign.add_variant(SimulationVariant("lcdm-base", cosmo_params))
   campaign.add_variant(SimulationVariant("wcdm-test", modified_params))
   ```

2. **Setup Phase**
   ```python
   # Initialize directory structure and configuration files
   campaign.initialize()
   # Creates: runs/, scratch/, logs/, results/ directories
   # Generates: .par files, SLURM scripts, run.sh scripts
   ```

3. **Execution Phase**
   ```python
   # Submit all simulations to SLURM queue
   job_ids = campaign.submit_all()
   # Automatic dependency management and resource allocation
   ```

4. **Monitoring Phase**
   ```python
   # Real-time status monitoring
   status = campaign.get_status()
   print(f"Running: {status.running}, Completed: {status.completed}")
   
   # Check individual simulation status
   for variant in campaign.variants:
       print(f"{variant.name}: {variant.status}")
   ```

5. **Analysis Phase**
   ```python
   # Collect and analyze results
   results = campaign.collect_results()
   campaign.generate_summary_report()
   ```

### Status Tracking

Campaign and simulation states are automatically tracked:

- **Campaign States**: `PLANNED`, `INITIALIZED`, `SUBMITTED`, `RUNNING`, `COMPLETED`, `FAILED`
- **Simulation States**: `CONFIGURED`, `SUBMITTED`, `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`

### Resource Management

```python
# Automatic resource allocation based on simulation size
simulation.estimate_resources()  # Returns time, memory, node estimates

# Custom resource specification
simulation.set_resources(
    nodes=8, 
    time="48:00:00", 
    partition="gpu",
    constraint="a100"
)
```

## Supported Data Types

| Dataset | Description | Variables |
|---------|-------------|-----------|
| `xvp` | Particles with positions, velocities, potential | x, y, z, vx, vy, vz, phi |
| `xv` | Particles with positions and velocities | x, y, z, vx, vy, vz |
| `xvh` | Halos with positions, velocities, properties | x, y, z, vx, vy, vz, npart, ... |

## Supported File Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `lcp` | Lightcone particles | Time-evolution analysis |
| `tps` | Tipsy format | Snapshot analysis |
| `fof` | Friends-of-friends | Halo catalogs |

## Parameter Type System

pkdpipe provides a comprehensive parameter categorization system that separates different types of parameters used throughout the pipeline:

### Parameter Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **Cosmological** | Physical cosmology parameters | `h`, `omegam`, `sigma8`, `ns`, `w0`, `wa` |
| **SLURM** | Job submission and resource parameters | `time`, `nodes`, `ntasks`, `partition` |
| **Simulation** | PKDGrav3 simulation settings | `ngrid`, `timesteps`, `output_format` |
| **Other** | Miscellaneous configuration parameters | Custom settings and derived values |

### Usage

```python
from pkdpipe.parameter_types import separate_parameters, ParameterCategory

# Separate mixed parameters into categories
all_params = {
    'h': 0.7, 'omegam': 0.3, 'sigma8': 0.8,  # cosmological
    'time': '24:00:00', 'nodes': 4,  # SLURM
    'ngrid': 512, 'timesteps': 100   # simulation
}

separated = separate_parameters(all_params)

# Access categorized parameters
cosmo_params = separated.cosmological
slurm_params = separated.slurm
sim_params = separated.simulation
```

This system ensures clean separation of concerns and prevents parameter conflicts between different pipeline components.

## Configuration and Presets

pkdpipe includes a comprehensive configuration system with built-in presets for common simulation types:

### Available Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `S0-test` | Quick test runs | Development and debugging |
| `S0-validation` | Medium-scale validation | Parameter validation and testing |
| `S0-scaling` | Large-scale performance testing | Scaling studies and optimization |
| `S0-production` | Full production runs | Scientific production simulations |

### Configuration Management

```python
from pkdpipe.config import Config, get_preset_params, get_cosmology_preset

# Load simulation preset
params = get_preset_params("S0-validation")
print(f"Grid size: {params['ngrid']}")
print(f"Box size: {params['boxsize']} Mpc/h")

# Load cosmological parameters
cosmo = get_cosmology_preset("planck2018")
print(f"Omega_m: {cosmo['omegam']}")
print(f"sigma_8: {cosmo['sigma8']}")

# Environment-specific paths
config = Config()
print(f"Scratch directory: {config.SCRATCH}")
print(f"CFS directory: {config.CFS}")

# Customize parameters for specific runs
custom_params = get_preset_params("S0-production")
custom_params.update({
    "omegam": 0.31,  # Custom cosmology
    "sigma8": 0.82,
    "ngrid": 1024,   # Higher resolution
    "nsteps": 100    # More timesteps
})
```

### Environment Setup

pkdpipe automatically detects and configures paths based on your environment:

```python
# Required environment variables
export SCRATCH="/path/to/scratch"    # High-speed scratch storage
export CFS="/path/to/cfs"           # Persistent storage
export USER="username"              # Username for directory creation
export pkdgravemail="user@domain"   # Email for job notifications

# Optional: Customize JAX/GPU settings
export JAX_PLATFORMS="cuda,cpu"     # Prefer CUDA over CPU
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Control GPU visibility
```

## Examples

The `/examples` directory contains comprehensive examples:

### Campaign and Simulation Management
*   **`create_campaign.py`**: Complete campaign setup and management workflow
*   **`submit_simulation.py`**: Individual simulation creation and submission
*   **`monitor_campaign.py`**: Campaign status monitoring and result collection

### Data Analysis
*   **`readparticles.py`**: Particle data reading and comparison between implementations
*   **`lightcone_particles.py`**: Lightcone particle analysis and visualization
*   **`halos.py`**: Halo catalog analysis and mass function calculation
*   **`power_spectrum_real_data.py`**: **Production-ready** power spectrum analysis of real simulation data
*   **`matterpower.py`**: Matter power spectrum calculations with shot noise correction
*   **`transfer.py`**: Transfer function analysis and comparison with theory

#### Featured: Production Power Spectrum Pipeline

The `power_spectrum_real_data.py` example demonstrates the complete production pipeline:

```bash
# Real data power spectrum analysis (production ready)
srun -n 4 -c 32 --qos=interactive -N 1 --time=60 -C gpu -A cosmosim --gpus-per-node=4 --exclusive \
  python examples/power_spectrum_real_data.py --assignment ngp

# Key features:
# ✅ 92GB simulation files (2.744B particles)
# ✅ Distributed MPI processing across 4 processes
# ✅ GPU-accelerated JAX FFT computation  
# ✅ Memory-optimized (25-30GB per process)
# ✅ Complete end-to-end pipeline validation
```

### Validation and Testing
*   **`compare_*.py`**: Validation scripts comparing legacy vs refactored implementations
*   **`benchmark_performance.py`**: Performance benchmarking and scaling tests
*   **`validate_cosmology.py`**: Cosmological parameter validation against known results

### Advanced Usage
*   **`custom_parameters.py`**: Advanced parameter customization and preset modification
*   **`distributed_processing.py`**: Multi-GPU and distributed computing examples
*   **`data_streaming.py`**: Large dataset processing with streaming mode

## Performance

The refactored data module provides:
- **Parallel Processing**: Efficient multiprocessing with configurable worker count
- **Memory Optimization**: Better memory management for large datasets
- **I/O Efficiency**: Optimized file reading with error recovery
- **Scalability**: Tested on large-scale simulations with billions of particles

## Testing and Validation

**For comprehensive test documentation, see [`./tests/README.md`](./tests/README.md)**

pkdpipe features a comprehensive unit test suite in the `./tests/` directory that validates all core functionality:

### Quick Test Execution

#### Serial Testing (Single Process)
```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_data_io.py -v        # I/O operations (12 tests)
python -m pytest tests/test_power_spectrum.py -v # Power spectrum analysis
python -m pytest tests/test_pkdpipe.py -v       # Simulation creation (4 tests)  
python -m pytest tests/test_campaign.py -v      # Campaign management (20+ tests)
```

#### Comprehensive Testing (RECOMMENDED)
```bash
# Automated comprehensive testing - handles environment setup automatically
./run_tests.sh

# Serial mode for debugging
./run_tests.sh --serial

# Enable verbose DEBUG output for troubleshooting
./run_tests.sh --debug

# Custom parameters
./run_tests.sh --time=30 --ntasks=8
```

#### Manual SLURM Testing (For Debugging)
```bash
# Manual MPI testing - requires environment setup
module load python && mamba activate pkdgrav
srun -n 4 -c 32 --qos=interactive -N 1 --time=60 -C gpu -A cosmosim --gpus-per-node=4 --exclusive python -m pytest tests/ -v
```

### Test Coverage

The unit test suite provides comprehensive validation of:

- **Data I/O Operations**: File format compatibility, spatial culling, multiprocessing (`test_data_io.py`)
- **Power Spectrum Analysis**: JAX/GPU acceleration, statistical validation, shot noise correction (`test_power_spectrum.py`)
- **Simulation Creation**: Parameter validation, directory structure, SLURM integration (`test_pkdpipe.py`)
- **Campaign Management**: Multi-simulation orchestration, dependency resolution, CLI (`test_campaign.py`)

### Test Requirements

```bash
# Install test dependencies
pip install pytest pytest-cov

# Environment setup (handled automatically by run_comprehensive_tests.sh)
# For manual testing only:
module load python && mamba activate pkdgrav
```

### Test Status

**Current Status**: **All tests passing** (43/43 tests, 100% pass rate)

**Recent Fixes Applied**:
- **Distributed FFT validation**: Fixed return value compatibility and test consolidation
- **Memory optimization**: Resolved production data processing issues
- **Test infrastructure**: Enhanced debug controls and clean output
- **JAX distributed**: Eliminated reinitialization conflicts through test restructuring

**Production Validation**: Real data examples (`examples/power_spectrum_real_data.py`) successfully process large datasets without issues.

**For detailed test descriptions, debugging tips, and advanced usage, see [`./tests/README.md`](./tests/README.md)**

## Troubleshooting

### JAX/Multiprocessing Issues

**Problem**: `RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded`

**Solution**: pkdpipe's architecture prevents this by ensuring JAX is only imported during FFT operations, after all multiprocessing completes. If you encounter this warning, ensure you're using the latest version.

### Large File Processing

**Problem**: Memory issues or slow performance with large simulation files

**Solution**: 
- Use `nproc=1` for very large files to avoid multiprocessing overhead
- The pipeline automatically optimizes for CPU-based processing of large particle datasets
- Gridding operations are designed to be memory-bound on CPU rather than GPU

### Distributed Computing on HPC

**Problem**: Jobs hang or fail to initialize in distributed mode

**Solution**:
- Ensure SLURM environment variables are properly set
- JAX distributed mode initialization is handled automatically
- Use recommended SLURM command: `srun -n 4 -c 32 python script.py`

### JAX Backend Issues

**Problem**: JAX tries to use incorrect backend (e.g., 'rocm' instead of 'cuda')

**Solution**: 
- Set `JAX_PLATFORMS='cuda,cpu'` environment variable
- The pipeline automatically configures JAX to use GPU platform
- Check that CUDA is available in your environment
