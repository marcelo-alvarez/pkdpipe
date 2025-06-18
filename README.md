# pkdpipe

**pkdpipe** is a Python library designed to streamline the process of working with N-body cosmological simulation data, particularly from PKDGrav3. It provides robust tools for data handling, analysis, and visualization with a focus on performance and reliability.

## Features

*   **Enhanced Data Interface:** Refactored data module with improved error handling, modular architecture, and comprehensive validation
*   **Robust Parameter Parsing:** Advanced parsing of PKDGrav3 parameter files with support for complex formats and validation
*   **Parallel Data Processing:** Efficient multiprocessing support for large-scale data operations
*   **Lightcone and Snapshot Modes:** Support for both lightcone particle data and snapshot-based analysis
*   **Multiple File Formats:** Support for various PKDGrav3 output formats (LCP, TPS, FOF)
*   **Cosmology Calculations:** Built-in functions for common cosmological calculations
*   **Power Spectrum Analysis:** Complete power spectrum calculation with FFT, shot noise correction, and k-binning
*   **Multi-GPU Support:** Distributed FFT computations and power spectrum calculations across multiple GPUs
*   **Parameter Type System:** Comprehensive categorization of cosmological, SLURM, and simulation parameters
*   **JAX Accelerated FFT:** Efficient Fast Fourier Transforms using JAX with distributed computing support
*   **Command-Line Interface:** Access to functionalities via a CLI
*   **Comprehensive Testing:** Validated against legacy implementations with extensive test coverage

## Installation

Currently, `pkdpipe` can be installed from the local directory:

```bash
pip install .
```

## Quick Start

### Basic Data Reading

```python
from pkdpipe.data import Data

# Initialize data interface
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

# Extract positions for power spectrum calculation
positions = np.column_stack([
    particles['box0']['x'],
    particles['box0']['y'], 
    particles['box0']['z']
])

# Calculate power spectrum
calculator = PowerSpectrumCalculator(ngrid=256, box_size=1000.0)
k_bins, power, n_modes = calculator.calculate_power_spectrum(
    positions, 
    subtract_shot_noise=True,
    assignment='cic'  # Cloud-in-cell assignment
)

print(f"Power spectrum calculated with {len(k_bins)} k-bins")
print(f"k range: {k_bins[0]:.3f} to {k_bins[-1]:.3f} h/Mpc")
```

### Multi-GPU Power Spectrum Calculation

```python
# For large simulations, use multi-GPU acceleration
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
    positions,
    subtract_shot_noise=True,
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
- **Parameter Organization**: Clear separation of cosmological, SLURM, and simulation parameters
- **Advanced Analytics**: Built-in power spectrum analysis with shot noise correction and k-binning
- **Backward Compatibility**: Maintains API compatibility with legacy code

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

## Examples

The `/examples` directory contains comprehensive examples:

*   **`readparticles.py`**: Particle data reading and comparison between implementations
*   **`lightcone_particles.py`**: Lightcone particle analysis and visualization
*   **`halos.py`**: Halo catalog analysis
*   **`matterpower.py`**: Matter power spectrum calculations
*   **`transfer.py`**: Transfer function analysis
*   **`compare_*.py`**: Validation scripts comparing legacy vs refactored implementations

## Performance

The refactored data module provides:
- **Parallel Processing**: Efficient multiprocessing with configurable worker count
- **Memory Optimization**: Better memory management for large datasets
- **I/O Efficiency**: Optimized file reading with error recovery
- **Scalability**: Tested on large-scale simulations with billions of particles

## Testing and Validation

All functionality has been thoroughly validated:
- **Unit Tests**: Comprehensive test suite covering all major functions
- **Integration Tests**: Full workflow testing with real simulation data
- **Comparison Tests**: Validated against legacy implementation for identical results
- **Performance Tests**: Benchmarked for memory usage and execution time

Run tests with:
```bash
python -m pytest tests/
```
