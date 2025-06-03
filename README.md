# pkdpipe

**pkdpipe** is a Python library designed to streamline the process of working with N-body cosmological simulation data, particularly from PKDGrav3. It provides robust tools for data handling, analysis, and visualization with a focus on performance and reliability.

## Features

*   **Enhanced Data Interface:** Refactored data module with improved error handling, modular architecture, and comprehensive validation
*   **Robust Parameter Parsing:** Advanced parsing of PKDGrav3 parameter files with support for complex formats and validation
*   **Parallel Data Processing:** Efficient multiprocessing support for large-scale data operations
*   **Lightcone and Snapshot Modes:** Support for both lightcone particle data and snapshot-based analysis
*   **Multiple File Formats:** Support for various PKDGrav3 output formats (LCP, TPS, FOF)
*   **Cosmology Calculations:** Built-in functions for common cosmological calculations
*   **JAX Accelerated FFT:** Efficient Fast Fourier Transforms using JAX
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

## Data Module Architecture

The refactored data module features a modular architecture for better maintainability:

### Core Classes

- **`Data`**: Main interface for PKDGrav3 data access
- **`ParameterParser`**: Handles parameter file parsing and validation
- **`DataProcessor`**: Manages data filtering, spatial culling, and transformations
- **`FileReader`**: Handles file I/O operations with error handling
- **`SimulationParams`**: Type-safe container for simulation parameters

### Key Improvements

- **Enhanced Error Handling**: Comprehensive validation and informative error messages
- **Robust Parameter Parsing**: Support for complex parameter formats including arrays
- **Structured Logging**: Replace print statements with proper logging levels
- **Type Safety**: Full type hints and dataclasses for better code reliability
- **Performance**: Optimized data processing with better memory management
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
