# pkdpipe

**pkdpipe** is a Python library designed to streamline the process of working with N-body cosmological simulation data, particularly from pkdgrav3. It provides tools for data handling, analysis, and visualization.

## Features

*   **Configuration Management:** Easy handling of simulation parameters and configurations.
*   **Cosmology Calculations:** Built-in functions for common cosmological calculations.
*   **Data Handling:** Tools for reading and processing simulation outputs.
*   **Simulation Interface:** Abstractions for interacting with simulation data.
*   **JAX Accelerated FFT:** Efficient Fast Fourier Transforms using JAX.
*   **Command-Line Interface:** Access to functionalities via a CLI.

## Installation

Currently, `pkdpipe` can be installed from the local directory:

```bash
pip install .
```

## Usage

Here's a basic example of how you might use `pkdpipe`:

```python
# Example of importing and using a module from pkdpipe
from pkdpipe import cosmology

# Get a default cosmology
cosmo = cosmology.default_cosmology()
print(cosmo)

# Further examples can be found in the /examples directory.
```

## Examples

The `/examples` directory contains scripts and notebooks demonstrating various functionalities of `pkdpipe`:

*   `halos.py`: Example script for halo analysis.
*   `matterpower.py`: Example script for calculating matter power spectra.
*   `transfer.py`: Example script for working with transfer functions.
