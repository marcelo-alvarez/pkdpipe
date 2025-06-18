"""
Power spectrum calculation using JAX FFT with multi-GPU support.

This module provides clean APIs for:
1. Particle-to-grid mass assignment
2. Multi-GPU FFT-based power spectrum calculation  
3. Integration with existing Data interface
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from pathlib import Path

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

from .jax_fft import fft


class ParticleGridder:
    """
    Handles particle-to-grid mass assignment with various schemes.
    
    Supports Cloud-in-Cell (CIC) and Nearest Grid Point (NGP) assignment
    with proper periodic boundary condition handling.
    """
    
    def __init__(self, ngrid: int, box_size: float, assignment: str = 'cic'):
        """
        Initialize particle gridder.
        
        Args:
            ngrid: Number of grid cells per dimension
            box_size: Size of simulation box in Mpc/h
            assignment: Mass assignment scheme ('cic' or 'ngp')
        """
        self.ngrid = ngrid
        self.box_size = box_size
        self.assignment = assignment.lower()
        self.grid_spacing = box_size / ngrid
        
        if self.assignment not in ['cic', 'ngp']:
            raise ValueError(f"Unknown assignment scheme: {assignment}")
    
    def particles_to_grid(self, particles: Dict[str, np.ndarray], n_devices: int = 1) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Convert particle positions and masses to density grid(s).
        
        Args:
            particles: Dictionary with 'x', 'y', 'z' positions and 'mass'
            n_devices: Number of GPUs/devices for domain decomposition
            
        Returns:
            Single 3D density grid if n_devices=1, or list of per-device grids if n_devices>1
            
        Raises:
            ValueError: If particles are outside simulation box or invalid format
        """
        # Validate input
        if len(particles['x']) == 0:
            raise ValueError("No particles provided")
        
        # Check that all arrays have same length
        n_particles = len(particles['x'])
        for key in ['y', 'z', 'mass']:
            if len(particles[key]) != n_particles:
                raise ValueError(f"Inconsistent array lengths: {key} has {len(particles[key])}, expected {n_particles}")
        
        # Extract positions and masses
        positions = np.column_stack([particles['x'], particles['y'], particles['z']])
        masses = particles['mass']
        
        # Check bounds
        if np.any(positions < 0) or np.any(positions >= self.box_size):
            raise ValueError("Particles outside simulation box")
        
        # Convert to grid coordinates
        grid_coords = positions / self.grid_spacing
        
        if n_devices == 1:
            # Single device - original behavior
            if self.assignment == 'cic':
                return self._cic_assignment(grid_coords, masses)
            elif self.assignment == 'ngp':
                return self._ngp_assignment(grid_coords, masses)
            else:
                raise ValueError(f"Unknown assignment scheme: {self.assignment}")
        else:
            # Multi-device domain decomposition
            return self._multi_device_assignment(grid_coords, masses, n_devices)
    
    def _cic_assignment(self, grid_coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Cloud-in-Cell mass assignment."""
        # Initialize density grid
        density_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)
        
        # Get integer grid coordinates (lower-left corner of cell)
        i_coords = np.floor(grid_coords).astype(int)
        
        # Get fractional offsets within cells
        dx = grid_coords - i_coords
        
        # Apply periodic boundary conditions
        i_coords = i_coords % self.ngrid
        
        # CIC weights (trilinear interpolation)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Weights for this corner
                    weight = ((1-i) * (1-dx[:, 0]) + i * dx[:, 0]) * \
                            ((1-j) * (1-dx[:, 1]) + j * dx[:, 1]) * \
                            ((1-k) * (1-dx[:, 2]) + k * dx[:, 2])
                    
                    # Grid indices with periodic wrapping
                    gi = (i_coords[:, 0] + i) % self.ngrid
                    gj = (i_coords[:, 1] + j) % self.ngrid
                    gk = (i_coords[:, 2] + k) % self.ngrid
                    
                    # Add weighted mass to grid
                    np.add.at(density_grid, (gi, gj, gk), masses * weight)
        
        return density_grid
    
    def _ngp_assignment(self, grid_coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Nearest Grid Point mass assignment."""
        # Initialize density grid
        density_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)
        
        # Round to nearest grid point
        i_coords = np.round(grid_coords).astype(int)
        
        # Apply periodic boundary conditions
        i_coords = i_coords % self.ngrid
        
        # Add masses to nearest grid points
        np.add.at(density_grid, (i_coords[:, 0], i_coords[:, 1], i_coords[:, 2]), masses)
        
        return density_grid

    def _multi_device_assignment(self, grid_coords: np.ndarray, masses: np.ndarray, n_devices: int) -> List[np.ndarray]:
        """
        Multi-device domain decomposition with buffer zones for mass assignment.
        
        Decomposes along Y-axis into n_devices domains. For CIC assignment, includes
        buffer zones to handle particles near domain boundaries.
        
        Args:
            grid_coords: Particle positions in grid coordinates
            masses: Particle masses
            n_devices: Number of devices/domains
            
        Returns:
            List of density grids, one per device
        """
        # Domain decomposition along Y-axis
        cells_per_device = self.ngrid // n_devices
        if self.ngrid % n_devices != 0:
            raise ValueError(f"Grid size {self.ngrid} must be divisible by number of devices {n_devices}")
        
        # Buffer size for CIC assignment (1 cell on each side)
        buffer_size = 1 if self.assignment == 'cic' else 0
        
        device_grids = []
        
        for device_id in range(n_devices):
            # Domain boundaries for this device
            y_start = device_id * cells_per_device
            y_end = (device_id + 1) * cells_per_device
            
            # Extended boundaries including buffer zones
            y_start_buf = max(0, y_start - buffer_size)
            y_end_buf = min(self.ngrid, y_end + buffer_size)
            
            # Find particles that contribute to this device's domain
            y_coords = grid_coords[:, 1]
            
            if self.assignment == 'cic':
                # For CIC, particles can affect neighboring cells
                particle_mask = ((y_coords >= y_start_buf - 1) & (y_coords < y_end_buf + 1))
            else:
                # For NGP, only particles within the domain
                particle_mask = ((y_coords >= y_start_buf) & (y_coords < y_end_buf))
            
            if not np.any(particle_mask):
                # No particles in this domain - create empty grid
                device_grid = np.zeros((self.ngrid, cells_per_device, self.ngrid), dtype=np.float32)
                device_grids.append(device_grid)
                continue
            
            # Get particles for this domain
            domain_coords = grid_coords[particle_mask].copy()
            domain_masses = masses[particle_mask]
            
            # Translate Y coordinates to local domain space
            domain_coords[:, 1] -= y_start
            
            # Initialize device grid (no buffer in the output grid)
            device_grid = np.zeros((self.ngrid, cells_per_device, self.ngrid), dtype=np.float32)
            
            # Perform mass assignment on device grid
            if self.assignment == 'cic':
                self._cic_assignment_device(domain_coords, domain_masses, device_grid, y_start, cells_per_device)
            else:  # NGP
                self._ngp_assignment_device(domain_coords, domain_masses, device_grid, y_start, cells_per_device)
            
            device_grids.append(device_grid)
        
        return device_grids
    
    def _cic_assignment_device(self, grid_coords: np.ndarray, masses: np.ndarray, 
                              device_grid: np.ndarray, y_offset: int, y_size: int) -> None:
        """CIC assignment for a single device domain."""
        # Get integer grid coordinates
        i_coords = np.floor(grid_coords).astype(int)
        dx = grid_coords - i_coords
        
        # CIC weights (trilinear interpolation)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Calculate weights
                    weight = ((1-i) * (1-dx[:, 0]) + i * dx[:, 0]) * \
                            ((1-j) * (1-dx[:, 1]) + j * dx[:, 1]) * \
                            ((1-k) * (1-dx[:, 2]) + k * dx[:, 2])
                    
                    # Target grid coordinates
                    ix = (i_coords[:, 0] + i) % self.ngrid
                    iy = i_coords[:, 1] + j
                    iz = (i_coords[:, 2] + k) % self.ngrid
                    
                    # Only assign to cells within this device's domain
                    valid_mask = (iy >= 0) & (iy < y_size)
                    
                    if np.any(valid_mask):
                        np.add.at(device_grid, 
                                (ix[valid_mask], iy[valid_mask], iz[valid_mask]), 
                                weight[valid_mask] * masses[valid_mask])
    
    def _ngp_assignment_device(self, grid_coords: np.ndarray, masses: np.ndarray,
                              device_grid: np.ndarray, y_offset: int, y_size: int) -> None:
        """NGP assignment for a single device domain."""
        # Round to nearest grid point
        i_coords = np.round(grid_coords).astype(int)
        
        # Apply periodic boundary conditions for X and Z
        i_coords[:, 0] = i_coords[:, 0] % self.ngrid
        i_coords[:, 2] = i_coords[:, 2] % self.ngrid
        
        # Only assign to cells within this device's Y domain
        valid_mask = (i_coords[:, 1] >= 0) & (i_coords[:, 1] < y_size)
        
        if np.any(valid_mask):
            valid_coords = i_coords[valid_mask]
            valid_masses = masses[valid_mask]
            np.add.at(device_grid, 
                    (valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]), 
                    valid_masses)

    def _cic_window_function(self, k_grid: np.ndarray) -> np.ndarray:
        """
        Calculate the Cloud-in-Cell window function correction.
        
        The CIC window function in k-space is:
        W_CIC(k) = ∏[sinc(k_i * dx/2)]^2 for i=x,y,z
        where sinc(x) = sin(x)/x and dx is the cell size.
        
        Args:
            k_grid: 3D array of k-magnitudes
            
        Returns:
            3D array of window function values
        """
        dx = self.box_size / self.ngrid
        
        # Create k-component grids
        kx = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        kz = 2 * np.pi * np.fft.rfftfreq(self.ngrid, dx)  # Real FFT
        
        kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate sinc functions for each component
        # sinc(x) = sin(x)/x, with sinc(0) = 1
        def safe_sinc(x):
            """Safe sinc function that handles x=0 correctly."""
            return np.where(x == 0, 1.0, np.sin(x) / x)
        
        sinc_x = safe_sinc(kx_3d * dx / 2.0)
        sinc_y = safe_sinc(ky_3d * dx / 2.0)
        sinc_z = safe_sinc(kz_3d * dx / 2.0)
        
        # CIC window function is product of squared sinc functions
        window = sinc_x**2 * sinc_y**2 * sinc_z**2
        
        return window
    
    def _ngp_window_function(self, k_grid: np.ndarray) -> np.ndarray:
        """
        Calculate the Nearest Grid Point window function correction.
        
        The NGP window function in k-space is:
        W_NGP(k) = ∏[sinc(k_i * dx/2)] for i=x,y,z
        
        Args:
            k_grid: 3D array of k-magnitudes
            
        Returns:
            3D array of window function values
        """
        dx = self.box_size / self.ngrid
        
        # Create k-component grids
        kx = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        kz = 2 * np.pi * np.fft.rfftfreq(self.ngrid, dx)  # Real FFT
        
        kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate sinc functions for each component
        def safe_sinc(x):
            """Safe sinc function that handles x=0 correctly."""
            return np.where(x == 0, 1.0, np.sin(x) / x)
        
        sinc_x = safe_sinc(kx_3d * dx / 2.0)
        sinc_y = safe_sinc(ky_3d * dx / 2.0)
        sinc_z = safe_sinc(kz_3d * dx / 2.0)
        
        # NGP window function is product of sinc functions
        window = sinc_x * sinc_y * sinc_z
        
        return window

    def _apply_window_correction(self, power_3d: np.ndarray, k_grid: np.ndarray, assignment: str) -> np.ndarray:
        """
        Apply window function correction for mass assignment scheme.
        
        For CIC assignment, the window function is W(k) = sinc(k_x*dx/2) * sinc(k_y*dy/2) * sinc(k_z*dz/2)
        where dx = dy = dz = box_size/ngrid is the grid spacing.
        
        Args:
            power_3d: 3D power spectrum array
            k_grid: 3D k-magnitude grid
            assignment: Mass assignment scheme ('cic' or 'ngp')
            
        Returns:
            Window function corrected power spectrum
        """
        if assignment.lower() == 'ngp':
            # NGP has no significant window function correction needed
            return power_3d
            
        elif assignment.lower() == 'cic':
            # CIC window function correction
            # Grid spacing
            dx = self.box_size / self.ngrid
            
            # Create individual k-component grids
            kx = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
            ky = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)  
            kz = 2 * np.pi * np.fft.rfftfreq(self.ngrid, dx)  # Real FFT
            
            kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
            
            # CIC window function: W(k) = sinc³(k*dx/2) where sinc(x) = sin(x)/x
            # Avoid division by zero
            def safe_sinc(x):
                """Compute sinc(x) = sin(x)/x safely."""
                return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / x)
            
            w_x = safe_sinc(kx_3d * dx / 2)
            w_y = safe_sinc(ky_3d * dx / 2) 
            w_z = safe_sinc(kz_3d * dx / 2)
            
            # Combined window function
            window = w_x * w_y * w_z
            
            # Avoid division by very small window values
            window_corrected = np.where(np.abs(window) > 1e-12, 1.0 / window**2, 1.0)
            
            return power_3d * window_corrected
            
        else:
            raise ValueError(f"Unknown assignment scheme: {assignment}")

    def get_density_diagnostics(self) -> Dict[str, float]:
        """
        Get diagnostic information about the last density field calculation.
        
        Returns:
            Dictionary with density field statistics
        """
        if hasattr(self, '_last_density_stats'):
            return self._last_density_stats.copy()
        else:
            return {}
    
    def _default_k_bins(self) -> np.ndarray:
        """Create default logarithmic k-binning based on theoretical FFT frequencies."""
        # Theoretical k-range for FFT grid
        k_fund = 2 * np.pi / self.box_size  # Fundamental mode
        k_nyquist = np.pi * self.ngrid / self.box_size  # Nyquist frequency
        
        # Use the theoretical range
        k_min = k_fund
        k_max = k_nyquist
        
        # Number of bins - use fewer bins for small grids
        n_bins = min(20, self.ngrid // 2)  
        
        # Ensure we have at least a few bins
        if n_bins < 5:
            n_bins = 5
        
        # Create logarithmic bins
        log_k_min = np.log10(k_min)
        log_k_max = np.log10(k_max)
        
        return np.logspace(log_k_min, log_k_max, n_bins + 1)  # n_bins+1 bin edges
    
    def _create_k_grid(self) -> np.ndarray:
        """Create 3D k-grid for FFT output."""
        # Grid spacing
        dx = self.box_size / self.ngrid
        
        # Create 1D k arrays with correct spacing
        kx = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        kz = 2 * np.pi * np.fft.rfftfreq(self.ngrid, dx)  # Real FFT
        
        # Create 3D grid
        kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate |k|
        k_mag = np.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)
        
        return k_mag
    
    def _bin_power_spectrum(self, k_grid: np.ndarray, power_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bin 3D power spectrum into 1D radial bins."""
        # Flatten arrays
        k_flat = k_grid.flatten()
        power_flat = power_grid.flatten()
        
        # Remove zero mode
        nonzero = k_flat > 0
        k_flat = k_flat[nonzero]
        power_flat = power_flat[nonzero]
        
        # Check if we have any valid k values
        if len(k_flat) == 0:
            return np.array([]), np.array([]), np.array([], dtype=int)
        
        # Find bin indices (digitize returns 0 for values < bins[0], 
        # len(bins) for values >= bins[-1])
        bin_indices = np.digitize(k_flat, self.k_bins)
        
        # Calculate binned quantities
        n_bins = len(self.k_bins) - 1
        k_binned = []
        power_binned = []
        n_modes = []
        
        for i in range(1, n_bins + 1):  # digitize returns 1-based indices
            mask = bin_indices == i
            if np.any(mask):
                k_binned.append(np.mean(k_flat[mask]))
                power_binned.append(np.mean(power_flat[mask]))
                n_modes.append(np.sum(mask))
        
        return np.array(k_binned), np.array(power_binned), np.array(n_modes, dtype=int)


class PowerSpectrumCalculator:
    """
    Multi-GPU power spectrum calculator using JAX FFT.
    
    Provides clean API for computing power spectra from particle distributions
    with proper shot noise handling and k-binning.
    """
    
    def __init__(self, ngrid: int, box_size: float, n_devices: int = 1,
                 k_bins: Optional[np.ndarray] = None):
        """
        Initialize power spectrum calculator.
        
        Args:
            ngrid: Grid resolution for FFT
            box_size: Simulation box size in Mpc/h
            n_devices: Number of GPUs to use
            k_bins: Custom k-binning (default: logarithmic)
        """
        self.ngrid = ngrid
        self.box_size = box_size  
        self.n_devices = n_devices
        self.fundamental_mode = 2 * np.pi / box_size
        self.volume = box_size**3
        self.cell_volume = self.volume / ngrid**3
        
        # Set up k-binning
        if k_bins is None:
            self.k_bins = self._default_k_bins()
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
        if len(particles['x']) == 0:
            raise ValueError("No particles provided")
        
        # Check bounds
        for coord in ['x', 'y', 'z']:
            if np.any(particles[coord] < 0) or np.any(particles[coord] >= self.box_size):
                raise ValueError("Particles outside simulation box")
        
        # Convert particles to density grid(s)
        gridder = ParticleGridder(self.ngrid, self.box_size, assignment)
        
        # Check if we're in multi-process mode (one GPU per process)
        try:
            import jax.distributed
            is_distributed = jax.process_count() > 1
        except:
            is_distributed = False
        
        if is_distributed:
            # Multi-process mode: each process handles its own domain
            # Domain decomposition should be handled by the launcher (srun)
            # Each process only sees its local particles and creates its local grid
            process_id = jax.process_index()
            total_processes = jax.process_count()
            
            # For multi-process, we need domain-decomposed particles
            # For now, use single-process approach and let srun handle the parallelism
            density_grid = gridder.particles_to_grid(particles, 1)
            mean_density = density_grid.mean()
            if mean_density > 0:
                delta_grid = (density_grid - mean_density) / mean_density
            else:
                raise ValueError("Mean density is zero - check particle data")
        elif self.n_devices > 1:
            # Single-process multi-GPU: use domain decomposition within process
            device_grids = gridder.particles_to_grid(particles, self.n_devices)
            
            # Calculate global mean density for normalization
            total_mass = np.sum([grid.sum() for grid in device_grids])
            total_volume = self.ngrid**3
            mean_density = total_mass / total_volume
            
            if mean_density <= 0:
                raise ValueError("Mean density is zero - check particle data")
            
            # Calculate density contrast for each device grid
            delta_grids = []
            for grid in device_grids:
                delta_grid = (grid - mean_density) / mean_density
                delta_grids.append(delta_grid)
        else:
            # Single device: original behavior
            density_grid = gridder.particles_to_grid(particles, 1)
            mean_density = density_grid.mean()
            if mean_density > 0:
                delta_grid = (density_grid - mean_density) / mean_density
            else:
                raise ValueError("Mean density is zero - check particle data")
        
        # Store density statistics for diagnostics
        if self.n_devices > 1:
            # Multi-device: calculate statistics from all grids
            all_density = np.concatenate([grid.flatten() for grid in device_grids])
            all_delta = np.concatenate([delta.flatten() for delta in delta_grids])
            density_variance = float(np.var(all_density))
            delta_mean = float(np.mean(all_delta))
            delta_variance = float(np.var(all_delta))
        else:
            # Single device: original calculation
            density_variance = float(np.var(density_grid))
            delta_mean = float(np.mean(delta_grid))
            delta_variance = float(np.var(delta_grid))
        
        self._last_density_stats = {
            'mean_density': float(mean_density),
            'density_variance': density_variance,
            'delta_mean': delta_mean,
            'delta_variance': delta_variance,
            'theoretical_shot_noise_variance': float(len(particles['x']) / (mean_density * self.ngrid**3))
        }
        
        # Perform FFT and calculate power spectrum with distributed binning
        if is_distributed:
            # Multi-process mode: use distributed FFT (returns only local k-space slice)
            delta_k_local = fft(delta_grid, direction='r2c')
            
            # Calculate local power spectrum
            power_3d_local = jnp.abs(delta_k_local)**2
            
            # Apply normalization
            volume_f64 = float(self.volume)
            ngrid_six = float(self.ngrid**6)
            power_3d_local = power_3d_local * (volume_f64 / ngrid_six)
            
            # Convert to numpy for k-binning
            power_3d_np = np.array(power_3d_local)
            
            # Create local k-grid for this process's domain
            k_grid_local = self._create_local_k_grid()
            
            # Apply window function correction for mass assignment scheme
            power_3d_corrected = self._apply_window_correction(power_3d_np, k_grid_local, assignment)
            
            # Bin local power spectrum (each process bins its own k-modes)
            k_binned, power_binned, n_modes = self._bin_power_spectrum_distributed(k_grid_local, power_3d_corrected)
            
        elif self.n_devices > 1:
            # Single-process multi-GPU: combine grids and use multi-GPU FFT  
            full_delta_grid = np.concatenate(delta_grids, axis=1)  # Concatenate along Y-axis
            delta_k = fft(full_delta_grid, direction='r2c')
            
            # Calculate power spectrum
            power_3d = jnp.abs(delta_k)**2
            
            # Apply normalization
            volume_f64 = float(self.volume)
            ngrid_six = float(self.ngrid**6)
            power_3d = power_3d * (volume_f64 / ngrid_six)
            
            # Convert to numpy for k-binning
            power_3d_np = np.array(power_3d)
            
            # Create k-grid
            k_grid = self._create_k_grid()
            
            # Apply window function correction for mass assignment scheme
            power_3d_corrected = self._apply_window_correction(power_3d_np, k_grid, assignment)
            
            # Bin power spectrum (single-process case)
            k_binned, power_binned, n_modes = self._bin_power_spectrum(k_grid, power_3d_corrected)
            
        else:
            # Single GPU/CPU FFT
            delta_k = jnp.fft.rfftn(delta_grid)
            
            # Calculate power spectrum
            power_3d = jnp.abs(delta_k)**2
            
            # Apply normalization
            volume_f64 = float(self.volume)
            ngrid_six = float(self.ngrid**6)
            power_3d = power_3d * (volume_f64 / ngrid_six)
            
            # Convert to numpy for k-binning
            power_3d_np = np.array(power_3d)
            
            # Create k-grid
            k_grid = self._create_k_grid()
            
            # Apply window function correction for mass assignment scheme
            power_3d_corrected = self._apply_window_correction(power_3d_np, k_grid, assignment)
            
            # Bin power spectrum (single-process case)
            k_binned, power_binned, n_modes = self._bin_power_spectrum(k_grid, power_3d_corrected)
        
        # Subtract shot noise if requested
        if subtract_shot_noise:
            shot_noise = self.volume / len(particles['x'])
            power_binned = power_binned - shot_noise
        
        return k_binned, power_binned, n_modes
    
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
        W_CIC(k) = ∏[sinc(k_i * dx/2)]^2 for i=x,y,z
        where sinc(x) = sin(x)/x and dx is the cell size.
        
        Args:
            k_grid: 3D array of k-magnitudes
            
        Returns:
            3D array of window function values
        """
        dx = self.box_size / self.ngrid
        
        # Create k-component grids
        kx = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        kz = 2 * np.pi * np.fft.rfftfreq(self.ngrid, dx)  # Real FFT
        
        kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate sinc functions for each component
        # sinc(x) = sin(x)/x, with sinc(0) = 1
        def safe_sinc(x):
            """Safe sinc function that handles x=0 correctly."""
            return np.where(x == 0, 1.0, np.sin(x) / x)
        
        sinc_x = safe_sinc(kx_3d * dx / 2.0)
        sinc_y = safe_sinc(ky_3d * dx / 2.0)
        sinc_z = safe_sinc(kz_3d * dx / 2.0)
        
        # CIC window function is product of squared sinc functions
        window = sinc_x**2 * sinc_y**2 * sinc_z**2
        
        return window
    
    def _ngp_window_function(self, k_grid: np.ndarray) -> np.ndarray:
        """
        Calculate the Nearest Grid Point window function correction.
        
        The NGP window function in k-space is:
        W_NGP(k) = ∏[sinc(k_i * dx/2)] for i=x,y,z
        
        Args:
            k_grid: 3D array of k-magnitudes
            
        Returns:
            3D array of window function values
        """
        dx = self.box_size / self.ngrid
        
        # Create k-component grids
        kx = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        kz = 2 * np.pi * np.fft.rfftfreq(self.ngrid, dx)  # Real FFT
        
        kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate sinc functions for each component
        def safe_sinc(x):
            """Safe sinc function that handles x=0 correctly."""
            return np.where(x == 0, 1.0, np.sin(x) / x)
        
        sinc_x = safe_sinc(kx_3d * dx / 2.0)
        sinc_y = safe_sinc(ky_3d * dx / 2.0)
        sinc_z = safe_sinc(kz_3d * dx / 2.0)
        
        # NGP window function is product of sinc functions
        window = sinc_x * sinc_y * sinc_z
        
        return window

    def get_density_diagnostics(self) -> Dict[str, float]:
        """
        Get diagnostic information about the last density field calculation.
        
        Returns:
            Dictionary with density field statistics
        """
        if hasattr(self, '_last_density_stats'):
            return self._last_density_stats.copy()
        else:
            return {}

    def _default_k_bins(self) -> np.ndarray:
        """Create default logarithmic k-binning based on theoretical FFT frequencies."""
        # Theoretical k-range for FFT grid
        k_fund = 2 * np.pi / self.box_size  # Fundamental mode
        k_nyquist = np.pi * self.ngrid / self.box_size  # Nyquist frequency
        
        # Use the theoretical range
        k_min = k_fund
        k_max = k_nyquist
        
        # Number of bins - use fewer bins for small grids
        n_bins = min(20, self.ngrid // 2)  
        
        # Ensure we have at least a few bins
        if n_bins < 5:
            n_bins = 5
        
        # Create logarithmic bins
        log_k_min = np.log10(k_min)
        log_k_max = np.log10(k_max)
        
        return np.logspace(log_k_min, log_k_max, n_bins + 1)  # n_bins+1 bin edges
    
    def _create_k_grid(self) -> np.ndarray:
        """Create 3D k-grid for FFT output."""
        # Grid spacing
        dx = self.box_size / self.ngrid
        
        # Create 1D k arrays with correct spacing
        kx = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        kz = 2 * np.pi * np.fft.rfftfreq(self.ngrid, dx)  # Real FFT
        
        # Create 3D grid
        kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate |k|
        k_mag = np.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)
        
        return k_mag
    
    def _bin_power_spectrum(self, k_grid: np.ndarray, power_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bin 3D power spectrum into 1D radial bins."""
        # Flatten arrays
        k_flat = k_grid.flatten()
        power_flat = power_grid.flatten()
        
        # Remove zero mode
        nonzero = k_flat > 0
        k_flat = k_flat[nonzero]
        power_flat = power_flat[nonzero]
        
        # Check if we have any valid k values
        if len(k_flat) == 0:
            return np.array([]), np.array([]), np.array([], dtype=int)
        
        # Find bin indices (digitize returns 0 for values < bins[0], 
        # len(bins) for values >= bins[-1])
        bin_indices = np.digitize(k_flat, self.k_bins)
        
        # Calculate binned quantities
        n_bins = len(self.k_bins) - 1
        k_binned = []
        power_binned = []
        n_modes = []
        
        for i in range(1, n_bins + 1):  # digitize returns 1-based indices
            mask = bin_indices == i
            if np.any(mask):
                k_binned.append(np.mean(k_flat[mask]))
                power_binned.append(np.mean(power_flat[mask]))
                n_modes.append(np.sum(mask))
        
        return np.array(k_binned), np.array(power_binned), np.array(n_modes, dtype=int)
    
    def _create_local_k_grid(self) -> np.ndarray:
        """Create local k-grid for this process's FFT domain slice."""
        try:
            import jax.distributed
            process_id = jax.process_index()
            total_processes = jax.process_count()
        except:
            # Fallback to full grid if not in distributed mode
            return self._create_k_grid()
        
        # Grid spacing
        dx = self.box_size / self.ngrid
        
        # Create 1D k arrays - kx and kz are the same on all processes
        kx = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        kz = 2 * np.pi * np.fft.rfftfreq(self.ngrid, dx)  # Real FFT
        
        # ky is domain-decomposed along Y-axis
        # Each process gets a slice of the full ky range
        y_slice_size = self.ngrid // total_processes
        y_start = process_id * y_slice_size
        y_end = y_start + y_slice_size
        
        # Handle case where ngrid is not evenly divisible by number of processes
        if process_id == total_processes - 1:
            y_end = self.ngrid
        
        ky_full = 2 * np.pi * np.fft.fftfreq(self.ngrid, dx)
        ky_local = ky_full[y_start:y_end]
        
        # Create 3D grid for local domain
        kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky_local, kz, indexing='ij')
        
        # Calculate |k|
        k_mag = np.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)
        
        return k_mag
    
    def _bin_power_spectrum_distributed(self, k_grid: np.ndarray, power_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bin local power spectrum and reduce across processes."""
        # First, bin the local power spectrum
        k_flat = k_grid.flatten()
        power_flat = power_grid.flatten()
        
        # Remove zero mode
        nonzero = k_flat > 0
        k_flat = k_flat[nonzero]
        power_flat = power_flat[nonzero]
        
        # Initialize local bins
        n_bins = len(self.k_bins) - 1
        local_power_sums = np.zeros(n_bins)
        local_k_sums = np.zeros(n_bins)
        local_mode_counts = np.zeros(n_bins, dtype=int)
        
        if len(k_flat) > 0:
            # Find bin indices
            bin_indices = np.digitize(k_flat, self.k_bins)
            
            # Accumulate local contributions to each bin
            for i in range(1, n_bins + 1):  # digitize returns 1-based indices
                mask = bin_indices == i
                if np.any(mask):
                    local_power_sums[i-1] = np.sum(power_flat[mask])
                    local_k_sums[i-1] = np.sum(k_flat[mask])
                    local_mode_counts[i-1] = np.sum(mask)
        
        # Reduce across all processes using MPI-like operations
        try:
            import jax.distributed
            
            # Convert to JAX arrays for reduction
            local_power_sums_jax = jnp.array(local_power_sums)
            local_k_sums_jax = jnp.array(local_k_sums)
            local_mode_counts_jax = jnp.array(local_mode_counts)
            
            # All-reduce to sum across processes
            global_power_sums = jax.lax.psum(local_power_sums_jax, axis_name=None)
            global_k_sums = jax.lax.psum(local_k_sums_jax, axis_name=None)
            global_mode_counts = jax.lax.psum(local_mode_counts_jax, axis_name=None)
            
            # Convert back to numpy
            global_power_sums = np.array(global_power_sums)
            global_k_sums = np.array(global_k_sums)
            global_mode_counts = np.array(global_mode_counts, dtype=int)
            
        except:
            # Fallback if distributed operations not available
            global_power_sums = local_power_sums
            global_k_sums = local_k_sums
            global_mode_counts = local_mode_counts
        
        # Calculate final binned values
        k_binned = []
        power_binned = []
        n_modes = []
        
        for i in range(n_bins):
            if global_mode_counts[i] > 0:
                k_binned.append(global_k_sums[i] / global_mode_counts[i])  # Mean k
                power_binned.append(global_power_sums[i] / global_mode_counts[i])  # Mean power
                n_modes.append(global_mode_counts[i])
        
        return np.array(k_binned), np.array(power_binned), np.array(n_modes, dtype=int)

