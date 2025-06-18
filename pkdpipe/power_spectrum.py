"""
Power spectrum calculation using JAX FFT with multi-GPU support.

This module provides the main PowerSpectrumCalculator class for computing
power spectra from particle distributions with proper shot noise handling,
k-binning, and window function corrections.

Example usage:
    # Basic power spectrum calculation
    calculator = PowerSpectrumCalculator(ngrid=256, box_size=1000.0)
    k_bins, power, n_modes = calculator.calculate_power_spectrum(particles)
    
    # Multi-GPU calculation with 4 devices
    calculator = PowerSpectrumCalculator(ngrid=512, box_size=2000.0, n_devices=4)
    k_bins, power, n_modes = calculator.calculate_power_spectrum(
        particles, subtract_shot_noise=True, assignment='cic'
    )
    
    # Custom k-binning
    k_bins_custom = np.logspace(-2, 1, 21)  # 20 bins from 0.01 to 10 h/Mpc
    calculator = PowerSpectrumCalculator(
        ngrid=256, box_size=1000.0, k_bins=k_bins_custom
    )
    k_bins, power, n_modes = calculator.calculate_power_spectrum(particles)
"""

import numpy as np
from typing import Dict, Tuple, Optional

try:
    import jax
    import jax.numpy as jnp
    # Enable 64-bit precision in JAX
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from .jax_fft import fft
from .particle_gridder import ParticleGridder
from .multi_gpu_utils import (
    is_distributed_mode, create_local_k_grid, create_full_k_grid,
    bin_power_spectrum_distributed, bin_power_spectrum_single, default_k_bins
)


class PowerSpectrumCalculator:
    """
    Multi-GPU power spectrum calculator using JAX FFT.
    
    Provides clean API for computing power spectra from particle distributions
    with proper shot noise handling, k-binning, and window function corrections.
    
    This calculator supports three execution modes:
    1. Single device (GPU/CPU) - for small simulations
    2. Multi-GPU within single process - for medium simulations
    3. Distributed multi-process - for large simulations with srun/mpirun
    """
    
    def __init__(self, ngrid: int, box_size: float, n_devices: int = 1,
                 k_bins: Optional[np.ndarray] = None):
        """
        Initialize power spectrum calculator.
        
        Args:
            ngrid: Grid resolution for FFT
            box_size: Simulation box size in Mpc/h
            n_devices: Number of GPUs to use (single-process mode)
            k_bins: Custom k-binning array (default: logarithmic)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if ngrid <= 0:
            raise ValueError("Grid size must be positive")
        if box_size <= 0:
            raise ValueError("Box size must be positive")
        if n_devices < 1:
            raise ValueError("Number of devices must be at least 1")
            
        self.ngrid = ngrid
        self.box_size = box_size  
        self.n_devices = n_devices
        self.fundamental_mode = 2 * np.pi / box_size
        self.volume = box_size**3
        self.cell_volume = self.volume / ngrid**3
        
        # Set up k-binning
        if k_bins is None:
            self.k_bins = default_k_bins(ngrid, box_size)
        else:
            self.k_bins = k_bins
    
    def calculate_power_spectrum(self, particles: Dict[str, np.ndarray],
                               subtract_shot_noise: bool = False,
                               assignment: str = 'cic') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate power spectrum from particle distribution.
        
        For random particles, the power spectrum should equal the shot noise P_shot = V/N.
        
        Args:
            particles: Dictionary with particle data ('x', 'y', 'z', 'mass')
            subtract_shot_noise: Whether to subtract shot noise
            assignment: Mass assignment scheme ('cic' or 'ngp')
            
        Returns:
            Tuple of (k_bins, power_spectrum, n_modes_per_bin)
            
        Raises:
            ValueError: If particle data is invalid
        """
        # Validate input
        self._validate_particles(particles)
        
        # Convert particles to density grid(s) using extracted ParticleGridder
        gridder = ParticleGridder(self.ngrid, self.box_size, assignment)
        
        # Check execution mode and calculate power spectrum accordingly
        if is_distributed_mode():
            return self._calculate_distributed(particles, gridder, subtract_shot_noise, assignment)
        elif self.n_devices > 1:
            return self._calculate_multi_gpu(particles, gridder, subtract_shot_noise, assignment)
        else:
            return self._calculate_single_device(particles, gridder, subtract_shot_noise, assignment)
    
    def _validate_particles(self, particles: Dict[str, np.ndarray]) -> None:
        """Validate particle input data."""
        if len(particles['x']) == 0:
            raise ValueError("No particles provided")
        
        # Check that all arrays have same length
        n_particles = len(particles['x'])
        for key in ['y', 'z', 'mass']:
            if key not in particles:
                raise ValueError(f"Missing required key: {key}")
            if len(particles[key]) != n_particles:
                raise ValueError(f"Inconsistent array lengths: {key} has {len(particles[key])}, expected {n_particles}")
        
        # Check bounds
        for coord in ['x', 'y', 'z']:
            if np.any(particles[coord] < 0) or np.any(particles[coord] >= self.box_size):
                raise ValueError("Particles outside simulation box")
    
    def _calculate_distributed(self, particles: Dict[str, np.ndarray], 
                             gridder: ParticleGridder, subtract_shot_noise: bool,
                             assignment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate power spectrum in distributed multi-process mode."""
        # Each process handles its local particles
        density_grid = gridder.particles_to_grid(particles, 1)
        mean_density = density_grid.mean()
        
        if mean_density <= 0:
            raise ValueError("Mean density is zero - check particle data")
        
        delta_grid = (density_grid - mean_density) / mean_density
        
        # Store diagnostics
        self._store_density_diagnostics(density_grid, delta_grid, len(particles['x']))
        
        # Distributed FFT (returns only local k-space slice)
        delta_k_local = fft(delta_grid, direction='r2c')
        
        # Calculate local power spectrum
        power_3d_local = jnp.abs(delta_k_local)**2 * (self.volume / self.ngrid**6)
        power_3d_np = np.array(power_3d_local)
        
        # Create local k-grid and apply corrections
        k_grid_local = create_local_k_grid(self.ngrid, self.box_size)
        power_3d_corrected = self._apply_window_correction(power_3d_np, k_grid_local, assignment)
        
        # Bin and reduce across processes
        k_binned, power_binned, n_modes = bin_power_spectrum_distributed(
            k_grid_local, power_3d_corrected, self.k_bins
        )
        
        return self._finalize_power_spectrum(k_binned, power_binned, n_modes, 
                                           subtract_shot_noise, len(particles['x']))
    
    def _calculate_multi_gpu(self, particles: Dict[str, np.ndarray],
                           gridder: ParticleGridder, subtract_shot_noise: bool,
                           assignment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate power spectrum in single-process multi-GPU mode."""
        # Use domain decomposition within process
        device_grids = gridder.particles_to_grid(particles, self.n_devices)
        
        # Calculate global mean density for normalization
        total_mass = np.sum([grid.sum() for grid in device_grids])
        mean_density = total_mass / self.ngrid**3
        
        if mean_density <= 0:
            raise ValueError("Mean density is zero - check particle data")
        
        # Calculate density contrast for each device grid
        delta_grids = [(grid - mean_density) / mean_density for grid in device_grids]
        
        # Store diagnostics
        all_density = np.concatenate([grid.flatten() for grid in device_grids])
        all_delta = np.concatenate([delta.flatten() for delta in delta_grids])
        self._store_density_diagnostics(all_density, all_delta, len(particles['x']))
        
        # Combine grids and use multi-GPU FFT  
        full_delta_grid = np.concatenate(delta_grids, axis=1)  # Concatenate along Y-axis
        delta_k = fft(full_delta_grid, direction='r2c')
        
        # Calculate power spectrum
        power_3d = jnp.abs(delta_k)**2 * (self.volume / self.ngrid**6)
        power_3d_np = np.array(power_3d)
        
        # Create k-grid and apply corrections
        k_grid = create_full_k_grid(self.ngrid, self.box_size)
        power_3d_corrected = self._apply_window_correction(power_3d_np, k_grid, assignment)
        
        # Bin power spectrum
        k_binned, power_binned, n_modes = bin_power_spectrum_single(
            k_grid, power_3d_corrected, self.k_bins
        )
        
        return self._finalize_power_spectrum(k_binned, power_binned, n_modes,
                                           subtract_shot_noise, len(particles['x']))
    
    def _calculate_single_device(self, particles: Dict[str, np.ndarray],
                               gridder: ParticleGridder, subtract_shot_noise: bool,
                               assignment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate power spectrum on single GPU/CPU."""
        # Single device gridding
        density_grid = gridder.particles_to_grid(particles, 1)
        mean_density = density_grid.mean()
        
        if mean_density <= 0:
            raise ValueError("Mean density is zero - check particle data")
        
        delta_grid = (density_grid - mean_density) / mean_density
        
        # Store diagnostics
        self._store_density_diagnostics(density_grid, delta_grid, len(particles['x']))
        
        # Single GPU/CPU FFT
        if JAX_AVAILABLE:
            delta_k = jnp.fft.rfftn(delta_grid)
            power_3d = jnp.abs(delta_k)**2 * (self.volume / self.ngrid**6)
            power_3d_np = np.array(power_3d)
        else:
            # Fallback to numpy if JAX not available
            delta_k = np.fft.rfftn(delta_grid)
            power_3d_np = np.abs(delta_k)**2 * (self.volume / self.ngrid**6)
        
        # Create k-grid and apply corrections
        k_grid = create_full_k_grid(self.ngrid, self.box_size)
        power_3d_corrected = self._apply_window_correction(power_3d_np, k_grid, assignment)
        
        # Bin power spectrum
        k_binned, power_binned, n_modes = bin_power_spectrum_single(
            k_grid, power_3d_corrected, self.k_bins
        )
        
        return self._finalize_power_spectrum(k_binned, power_binned, n_modes,
                                           subtract_shot_noise, len(particles['x']))
    
    def _store_density_diagnostics(self, density_data: np.ndarray, 
                                 delta_data: np.ndarray, n_particles: int) -> None:
        """Store density field diagnostics for analysis."""
        if hasattr(density_data, 'flatten'):
            density_flat = density_data.flatten()
        else:
            density_flat = density_data
            
        if hasattr(delta_data, 'flatten'):
            delta_flat = delta_data.flatten()
        else:
            delta_flat = delta_data
        
        mean_density = float(np.mean(density_flat))
        
        self._last_density_stats = {
            'mean_density': mean_density,
            'density_variance': float(np.var(density_flat)),
            'delta_mean': float(np.mean(delta_flat)),
            'delta_variance': float(np.var(delta_flat)),
            'theoretical_shot_noise_variance': float(n_particles / (mean_density * self.ngrid**3))
        }
    
    def _apply_window_correction(self, power_3d: np.ndarray, k_grid: np.ndarray, 
                               assignment: str) -> np.ndarray:
        """
        Apply window function correction to the power spectrum.
        
        Args:
            power_3d: 3D power spectrum array
            k_grid: 3D k-magnitude grid
            assignment: Mass assignment scheme ('cic' or 'ngp')
            
        Returns:
            Corrected 3D power spectrum
        """
        if assignment.lower() == 'cic':
            window = self._cic_window_function(k_grid)
        elif assignment.lower() == 'ngp':
            window = self._ngp_window_function(k_grid)
        else:
            # Unknown assignment scheme, no correction
            return power_3d
        
        # Avoid division by zero
        window_safe = np.where(window > 1e-10, window, 1.0)
        
        # Apply correction: P_corrected = P_measured / W²
        power_corrected = power_3d / (window_safe**2)
        
        return power_corrected
    
    def _cic_window_function(self, k_grid: np.ndarray) -> np.ndarray:
        """
        Calculate the Cloud-in-Cell window function correction.
        
        The CIC window function in k-space is:
        W_CIC(k) = ∏[sinc(k_i * dx/2)]² for i=x,y,z
        where sinc(x) = sin(x)/x and dx is the cell size.
        """
        dx = self.box_size / self.ngrid
        
        # Create k-component grids
        kx = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        kz = 2 * np.pi * np.fft.rfftfreq(self.ngrid, dx)  # Real FFT
        
        kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate sinc functions for each component
        def safe_sinc(x):
            return np.where(x == 0, 1.0, np.sin(x) / x)
        
        sinc_x = safe_sinc(kx_3d * dx / 2.0)
        sinc_y = safe_sinc(ky_3d * dx / 2.0)
        sinc_z = safe_sinc(kz_3d * dx / 2.0)
        
        # CIC window function is product of squared sinc functions
        return sinc_x**2 * sinc_y**2 * sinc_z**2
    
    def _ngp_window_function(self, k_grid: np.ndarray) -> np.ndarray:
        """
        Calculate the Nearest Grid Point window function correction.
        
        The NGP window function in k-space is:
        W_NGP(k) = ∏[sinc(k_i * dx/2)] for i=x,y,z
        """
        dx = self.box_size / self.ngrid
        
        # Create k-component grids
        kx = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        kz = 2 * np.pi * np.fft.rfftfreq(self.ngrid, dx)  # Real FFT
        
        kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate sinc functions for each component
        def safe_sinc(x):
            return np.where(x == 0, 1.0, np.sin(x) / x)
        
        sinc_x = safe_sinc(kx_3d * dx / 2.0)
        sinc_y = safe_sinc(ky_3d * dx / 2.0)
        sinc_z = safe_sinc(kz_3d * dx / 2.0)
        
        # NGP window function is product of sinc functions
        return sinc_x * sinc_y * sinc_z
    
    def _finalize_power_spectrum(self, k_binned: np.ndarray, power_binned: np.ndarray,
                               n_modes: np.ndarray, subtract_shot_noise: bool,
                               n_particles: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply final corrections and return results."""
        # Subtract shot noise if requested
        if subtract_shot_noise and len(power_binned) > 0:
            shot_noise = self.volume / n_particles
            power_binned = power_binned - shot_noise
        
        return k_binned, power_binned, n_modes
    
    def get_density_diagnostics(self) -> Dict[str, float]:
        """
        Get diagnostic information about the last density field calculation.
        
        Returns:
            Dictionary with density field statistics including:
            - mean_density: Mean density of the field
            - density_variance: Variance of density field
            - delta_mean: Mean of density contrast field (should be ~0)
            - delta_variance: Variance of density contrast field
            - theoretical_shot_noise_variance: Expected shot noise variance
        """
        if hasattr(self, '_last_density_stats'):
            return self._last_density_stats.copy()
        else:
            return {}