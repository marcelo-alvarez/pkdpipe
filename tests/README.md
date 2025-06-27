# pkdpipe Test Suite

This directory contains comprehensive tests for the pkdpipe library, covering all major functionality including I/O operations, power spectrum analysis, simulation creation, and campaign management.

## Test Overview

### Core Test Files

| Test File | Coverage | Key Features |
|-----------|----------|--------------|
| `test_data_io.py` | I/O operations and data handling | File format compatibility, spatial culling, multiprocessing |
| `test_power_spectrum.py` | Power spectrum analysis | JAX/GPU acceleration, statistical validation, shot noise |
| `test_pkdpipe.py` | Simulation creation pipeline | Parameter validation, directory structure, SLURM integration |
| `test_campaign.py` | Campaign management system | Multi-simulation orchestration, dependency resolution, CLI |

### Test Data

- **Location**: `tests/test_data/output/`
- **Size**: ~8.5KB total (git-friendly)
- **Formats**: Lightcone (.lcp), Tipsy (.tps), Friends-of-Friends (.fof)
- **Strategy**: Strategically positioned particles for comprehensive spatial testing

## Running Tests

### Serial Tests (Recommended)
```bash
# From project root
python -m pytest tests/

# Verbose output
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=pkdpipe
```

### 🚨 SLURM Distributed Tests (Currently Broken)
```bash
# Comprehensive testing with MPI/GPU support - CURRENTLY FAILING
srun -n 4 -c 32 --qos=interactive -N 1 --time=60 -C gpu -A cosmosim --gpus-per-node=4 --exclusive python -m pytest tests/ -v

# Using test script - ALSO BROKEN
srun -n 4 -c 32 --qos=interactive -N 1 --time=60 -C gpu -A cosmosim --gpus-per-node=4 --exclusive ./run_comprehensive_tests.sh
```

**⚠️ Known Issues**:
- MPI tests crash with exit code 255
- SLURM jobs terminated before completion
- Multi-process execution broken
- See `../testlog` for detailed failure logs

### Individual Test Suites
```bash
# I/O operations (12 tests)
python -m pytest tests/test_data_io.py -v

# Power spectrum analysis (GPU-dependent)
python -m pytest tests/test_power_spectrum.py -v

# Simulation creation (4 tests)
python -m pytest tests/test_pkdpipe.py -v

# Campaign management (20+ tests)
python -m pytest tests/test_campaign.py -v
```

### Specific Test Cases
```bash
# Test specific functionality
python -m pytest tests/test_data_io.py::TestDataIO::test_lightcone_data_loading -v
python -m pytest tests/test_power_spectrum.py::TestPowerSpectrumCalculator::test_shot_noise_power_spectrum -v
```

## Test Details

### I/O Testing (`test_data_io.py`)

**Coverage**: 12 comprehensive tests

**Key Test Cases**:
- **Format Support**: Lightcone (.lcp), Tipsy snapshots (.tps), FOF halos (.fof)
- **Spatial Operations**: Bounding box filtering, edge case handling
- **Multiprocessing**: Consistency across multiple worker processes
- **Error Handling**: Missing files, invalid parameters, corrupted data
- **Data Validation**: Particle counts, coordinate ranges, file size verification

**Test Data Structure**:
```
tests/test_data/output/
├── test_sim.00001           # Unified tipsy snapshot (45 particles)
├── test_sim.00001.lcp.0     # Lightcone file 0 (16 particles)
├── test_sim.00001.lcp.1     # Lightcone file 1 (16 particles)
├── test_sim.00001.lcp.2     # Lightcone file 2 (16 particles)
├── test_sim.00001.lcp.3     # Lightcone file 3 (16 particles)
├── test_sim.00001.fofstats.0 # Halo file 0 (6 halos)
├── test_sim.00001.fofstats.1 # Halo file 1 (6 halos)
├── test_sim.00001.fofstats.2 # Halo file 2 (6 halos)
├── test_sim.log             # Redshift information
└── test_sim.par             # Parameter file
```

### Power Spectrum Testing (`test_power_spectrum.py`)

**Coverage**: JAX-based FFT calculations with multi-GPU support

**Key Features**:
- **GPU Detection**: Automatic fallback to CPU if GPUs unavailable
- **Statistical Validation**: Chi-squared analysis of shot noise power spectrum
- **Multi-Device Support**: Tests distributed computing across available devices
- **Performance Metrics**: Timing and memory usage validation

**Statistical Theory**:
- Tests random particle distributions where P(k) = V/N (shot noise)
- Validates χ² statistics: 90%+ of k-bins within 3σ bounds
- Reduced χ² should be ~1.0 for good statistical consistency

### Simulation Testing (`test_pkdpipe.py`)

**Coverage**: Full simulation creation workflow

**Key Validations**:
- **Directory Structure**: Run directories, scratch symlinks, output paths
- **Parameter Files**: Correct .par file generation with grid/box parameters
- **SLURM Scripts**: Proper .sbatch file creation with resource allocation
- **File Permissions**: Executable scripts, proper ownership
- **Integration**: Matches `create-test.sh` reference implementation

### Campaign Testing (`test_campaign.py`)

**Coverage**: Multi-simulation orchestration system

**Key Components**:
- **Configuration Loading**: YAML parsing, validation, preset resolution
- **Dependency Management**: Simulation ordering, prerequisite checking
- **Status Tracking**: Job states, completion monitoring, persistence
- **CLI Integration**: Command-line interface validation
- **Preset Validation**: Cosmology (LCDM, wCDM, φCDM) and simulation presets

## Test Requirements

### Environment Setup
```bash
# Activate pkdpipe environment
source /global/cfs/cdirs/cosmosim/slac/malvarez/pkdgrav3/env/loadenv.sh

# Install test dependencies
pip install pytest pytest-cov
```

### GPU Testing
- **JAX Tests**: Automatically detect and use available GPUs
- **CPU Fallback**: Tests run on CPU if no GPUs available
- **Multi-GPU**: Tests scale across multiple devices when available

### Expected Results
- **I/O Tests**: 12/12 passing (100% success rate)
- **Power Spectrum**: GPU-dependent, statistical validation
- **Simulation**: 4/4 passing (directory structure, files, permissions)
- **Campaign**: 20+ tests covering all orchestration features

## Test Data Management

### Regenerating Test Data
```bash
# Only needed if test data is corrupted or modified
cd tests/
python utils/generate_test_data.py
```

**When to Regenerate**:
- Test data files are missing or corrupted
- Need different spatial distributions for new test cases
- Binary format changes in PKDGrav3
- Adding support for new file formats

### Test Data Design
- **Strategic Positioning**: Particles placed for comprehensive spatial testing
- **Multi-File Support**: Multiple files test multiprocessing I/O
- **Format Compliance**: Exact PKDGrav3 binary format matching
- **Size Optimization**: Minimal file sizes suitable for git repository

## Debugging Tests

### Common Issues

**File Not Found Errors**:
```bash
# Ensure you're in the correct directory
cd /global/cfs/cdirs/cosmosim/slac/malvarez/pkdgrav3/pipeline/pkdpipe
python -m pytest tests/test_data_io.py -v
```

**GPU Tests Failing**:
```bash
# Check GPU availability
python -c "import jax; print(f'Devices: {jax.devices()}')"

# Force CPU mode if needed
JAX_PLATFORM_NAME=cpu python -m pytest tests/test_power_spectrum.py -v
```

**Permission Errors**:
```bash
# Check file system permissions
ls -la tests/test_data/output/
```

### Verbose Debugging
```bash
# Maximum verbosity
python -m pytest tests/test_data_io.py -vvv --tb=long

# Print debug output
python -m pytest tests/test_power_spectrum.py -v -s
```

## Coverage Metrics

Expected test coverage by component:
- **Data I/O**: 100% of core fetch_data() functionality
- **Power Spectrum**: 95%+ of calculation pipeline
- **Simulation Creation**: 100% of creation workflow
- **Campaign Management**: 90%+ of orchestration features

## Contributing

When adding new tests:
1. Follow existing naming conventions (`test_*.py`)
2. Include comprehensive docstrings
3. Add both positive and negative test cases
4. Validate against real PKDGrav3 data when possible
5. Update this documentation for new test suites

## Performance Expectations

Typical test run times:
- **I/O Tests**: ~10-30 seconds (depends on file system)
- **Power Spectrum**: ~30-120 seconds (depends on GPU availability)
- **Simulation Tests**: ~5-15 seconds
- **Campaign Tests**: ~10-30 seconds

Total test suite: ~1-3 minutes for full validation.