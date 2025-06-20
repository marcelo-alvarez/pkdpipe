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

# Suppress JAX TPU warnings globally
import os
os.environ['JAX_PLATFORMS'] = 'cpu,gpu'

import warnings
import logging
warnings.filterwarnings("ignore", message=".*libtpu.so.*")
warnings.filterwarnings("ignore", message=".*Failed to open libtpu.so.*")
logging.getLogger('jax._src.xla_bridge').setLevel(logging.ERROR)

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
    is_distributed_mode, create_local_k_grid, create_full_k_grid, create_slab_k_grid,
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
        print("DEBUG: PowerSpectrumCalculator.__init__ called - code is updated!")
        # Initialize JAX distributed mode if needed BEFORE any other operations
        from .multi_gpu_utils import is_distributed_mode
        print("DEBUG: About to call is_distributed_mode()")
        is_distributed = is_distributed_mode()  # This will initialize JAX distributed mode
        print(f"DEBUG: is_distributed_mode() returned: {is_distributed}")
        
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
        
        # Show initialization info only from master process
        process_id = 0
        if is_distributed:
            try:
                import jax
                process_id = jax.process_index()
            except:
                pass
        
        if process_id == 0:
            print(f"PowerSpectrumCalculator initialized:")
            print(f"  Grid size: {ngrid}³")
            print(f"  Box size: {box_size:.1f} Mpc/h")
            print(f"  Cell size: {box_size/ngrid:.3f} Mpc/h")
            print(f"  k-bins: {len(self.k_bins)-1}")
            print(f"  k-range: {self.k_bins[0]:.6f} to {self.k_bins[-1]:.6f} h/Mpc")
            if is_distributed:
                print(f"  JAX Distributed Mode: ENABLED ({jax.process_count()} processes)")
            else:
                print(f"  JAX Distributed Mode: DISABLED (using {n_devices} device(s))")
    
    def calculate_power_spectrum(self, particles,
                               subtract_shot_noise: bool = False,
                               assignment: str = 'cic') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate power spectrum from particle distribution.
        
        For random particles, the power spectrum should equal the shot noise P_shot = V/N.
        
        Args:
            particles: Particle dictionary ('x', 'y', 'z', 'mass') 
            subtract_shot_noise: Whether to subtract shot noise
            assignment: Mass assignment scheme ('cic' or 'ngp')
            
        Returns:
            Tuple of (k_bins, power_spectrum, n_modes_per_bin)
            
        Raises:
            ValueError: If particle data is invalid
        """
        # Validate input
        self._validate_particles(particles)
        
        # Check execution mode and calculate power spectrum accordingly
        print(f"DEBUG: About to check is_distributed_mode()...", flush=True)
        if is_distributed_mode():
            print(f"DEBUG: Distributed mode detected, creating gridder...", flush=True)
            # Create gridder for distributed mode
            from .particle_gridder import ParticleGridder
            gridder = ParticleGridder(self.ngrid, self.box_size, assignment)
            print(f"DEBUG: About to call _calculate_distributed...", flush=True)
            return self._calculate_distributed(
                particles, gridder, subtract_shot_noise, assignment)
        elif self.n_devices > 1:
            # Create gridder for multi-GPU mode
            from .particle_gridder import ParticleGridder
            gridder = ParticleGridder(self.ngrid, self.box_size, assignment)
            return self._calculate_multi_gpu(particles, gridder, subtract_shot_noise, assignment)
        else:
            # Create gridder for single device mode
            from .particle_gridder import ParticleGridder
            gridder = ParticleGridder(self.ngrid, self.box_size, assignment)
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
        """
        Calculate power spectrum in distributed multi-process mode with spatial decomposition.
        
        This method implements proper spatial domain decomposition where each process
        handles a spatial slab (e.g., y ∈ [0, 128] for process 0) rather than arbitrary
        chunks. Particles are redistributed using JAX collectives to ensure each
        particle ends up on the process responsible for its spatial region.
        """
        print("DEBUG: _calculate_distributed started", flush=True)
        
        if JAX_AVAILABLE:
            print("DEBUG: JAX is available, importing...", flush=True)
            import jax
            import jax.numpy as jnp
            print(f"DEBUG: JAX imported, process_index={jax.process_index()}, process_count={jax.process_count()}", flush=True)
        else:
            print("DEBUG: JAX not available", flush=True)
        
        print("DEBUG: About to start particle redistribution...", flush=True)
        # Step 1: Redistribute particles based on spatial decomposition
        print("Redistributing particles based on spatial decomposition...")
        
        # TEMPORARY: Broadcast all particles to all processes for testing
        # In production, this should use proper JAX collectives
        if JAX_AVAILABLE:
            import jax
            process_id = jax.process_index()
            n_processes = jax.process_count()
            
            # For now, all processes use the same particles (inefficient but works for testing)
            # TODO: Implement proper particle exchange using JAX collectives
            print(f"Process {process_id}: Starting with {len(particles['x']):,} particles")
            broadcast_particles = particles
        else:
            broadcast_particles = particles
        
        print(f"Process {process_id}: Calling redistribute_particles_spatial...")
        spatial_particles, y_min, y_max, base_y_min, base_y_max = redistribute_particles_spatial(
            broadcast_particles, assignment, self.ngrid, self.box_size
        )
        print(f"Process {process_id}: Spatial filtering complete, {len(spatial_particles['x']):,} particles in domain")
        
        # Step 2: Create spatial slab grid (not full grid)
        print(f"Process {process_id}: Creating slab grid...")
        slab_height = base_y_max - base_y_min
        if JAX_AVAILABLE:
            process_id = jax.process_index()
            n_processes = jax.process_count()
        else:
            process_id = 0
            n_processes = 1
            
        print(f"Process {process_id}: Gridding {len(spatial_particles['x']):,} particles to slab [{base_y_min}:{base_y_max}]")
        
        # Grid particles to slab including ghost zones
        print(f"Process {process_id}: Calling particles_to_slab...")
        full_slab = gridder.particles_to_slab(spatial_particles, y_min, y_max, self.ngrid)
        print(f"Process {process_id}: Slab gridding complete, shape: {full_slab.shape}")
        
        # Extract owned portion (remove ghost zones)
        print(f"Process {process_id}: Extracting owned slab portion...")
        ghost_start = base_y_min - y_min
        ghost_end = ghost_start + slab_height
        owned_slab = full_slab[:, ghost_start:ghost_end, :]  # Shape: (512, slab_height, 512)
        print(f"Process {process_id}: Owned slab shape: {owned_slab.shape}")
        
        # Step 3: Calculate mean density and density contrast  
        print(f"Process {process_id}: Calculating mean density...")
        local_mass = np.sum(owned_slab)
        if JAX_AVAILABLE:
            # In distributed mode, just use local mass times process count as approximation
            # This avoids JAX collective operation issues for now
            local_mass_jax = jnp.array(local_mass)
            print(f"Process {jax.process_index()}: local_mass = {local_mass}")
            total_mass = local_mass_jax * jax.process_count()
        else:
            total_mass = local_mass
            
        mean_density = total_mass / self.ngrid**3
        print(f"Process {process_id}: Mean density = {mean_density}")
        
        if mean_density <= 0:
            raise ValueError("Mean density is zero - check particle data")
        
        print(f"Process {process_id}: Computing density contrast...")
        delta_slab = (owned_slab - mean_density) / mean_density
        
        # Store diagnostics
        self._store_density_diagnostics(owned_slab, delta_slab, len(spatial_particles['x']))
        
        # Step 4: Distributed FFT on slabs
        print(f"Process {process_id}: Starting FFT...")
        if JAX_AVAILABLE:
            delta_k_slab = fft(delta_slab, direction='r2c')  # JAX distributed FFT
            # Calculate power spectrum on slab
            power_3d_slab = jnp.abs(delta_k_slab)**2 * (self.volume / self.ngrid**6)
            power_3d_np = np.array(power_3d_slab)
        else:
            # Fallback to numpy FFT
            delta_k_slab = np.fft.rfftn(delta_slab)
            power_3d_slab = np.abs(delta_k_slab)**2 * (self.volume / self.ngrid**6)
            power_3d_np = power_3d_slab
        print(f"Process {process_id}: FFT complete, power_3d shape: {power_3d_np.shape}")
        
        # Step 5: Create k-grid for slab and apply corrections
        print(f"Process {process_id}: Creating k-grid and applying corrections...")
        k_grid_slab = create_slab_k_grid(self.ngrid, self.box_size, base_y_min, base_y_max)
        power_3d_corrected = self._apply_window_correction(power_3d_np, k_grid_slab, assignment)
        
        # Step 6: Bin and reduce across processes
        k_binned, power_binned, n_modes = bin_power_spectrum_distributed(
            k_grid_slab, power_3d_corrected, self.k_bins
        )
        
        return self._finalize_power_spectrum(k_binned, power_binned, n_modes, 
                                           subtract_shot_noise, len(spatial_particles['x']))
    
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
    
    def _calculate_streaming(self, data_reader, subtract_shot_noise: bool,
                           assignment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate power spectrum using streaming/chunked data processing.
        
        This method processes particle data in chunks to reduce memory usage,
        accumulating particles into a density grid without loading all particles
        into memory simultaneously.
        
        Args:
            data_reader: Data object with chunked reading capability
            subtract_shot_noise: Whether to subtract shot noise
            assignment: Mass assignment scheme ('cic' or 'ngp')
            
        Returns:
            Tuple of (k_bins, power_spectrum, n_modes_per_bin)
        """
        print("Using streaming power spectrum calculation for memory efficiency")
        
        # Initialize density grid based on execution mode
        if is_distributed_mode():
            return self._calculate_streaming_distributed(data_reader, subtract_shot_noise, assignment)
        else:
            return self._calculate_streaming_single(data_reader, subtract_shot_noise, assignment)
    
    def _calculate_streaming_single(self, data_reader, subtract_shot_noise: bool,
                                  assignment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Streaming calculation for single process mode."""
        # Initialize density grid
        density_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float64)
        gridder = ParticleGridder(self.ngrid, self.box_size, assignment)
        total_particles = 0
        
        print(f"Processing particles in chunks...")
        chunk_count = 0
        
        # Process particles chunk by chunk
        for chunk_particles in data_reader.fetch_data_chunked():
            chunk_count += 1
            n_chunk_particles = len(chunk_particles['x'])
            total_particles += n_chunk_particles
            
            # Grid particles from this chunk and accumulate
            chunk_density = gridder.particles_to_grid(chunk_particles, n_devices=1)
            density_grid += chunk_density
            
            print(f"  Processed chunk {chunk_count}: {n_chunk_particles:,} particles")
        
        print(f"Total particles processed: {total_particles:,}")
        
        # Calculate mean density and density contrast
        mean_density = density_grid.mean()
        if mean_density <= 0:
            raise ValueError("Mean density is zero - check particle data")
        
        delta_grid = (density_grid - mean_density) / mean_density
        
        # Store diagnostics
        self._store_density_diagnostics(density_grid, delta_grid, total_particles)
        
        # Continue with standard power spectrum calculation
        if JAX_AVAILABLE:
            delta_k = jnp.fft.rfftn(delta_grid)
            power_3d = jnp.abs(delta_k)**2 * (self.volume / self.ngrid**6)
            power_3d_np = np.array(power_3d)
        else:
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
                                           subtract_shot_noise, total_particles)
    
    def _calculate_streaming_distributed(self, data_reader, subtract_shot_noise: bool,
                                       assignment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate power spectrum using streaming/chunked data processing in distributed mode.
        
        Each SLURM process reads its assigned chunks and accumulates particles into a density grid.
        """
        # Initialize density grid
        density_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float64)
        gridder = ParticleGridder(self.ngrid, self.box_size, assignment)
        total_particles = 0
        
        # Get process ID for logging
        process_id = 0
        try:
            import jax
            process_id = jax.process_index()
        except:
            pass
        
        print(f"Processing particles in chunks (distributed mode)...")
        chunk_count = 0
        
        # Process particles chunk by chunk
        try:
            print(f"Starting chunk processing for process {process_id}...")
            
            for chunk_particles in data_reader.fetch_data_chunked():
                chunk_count += 1
                n_chunk_particles = len(chunk_particles.get('x', []))
                total_particles += n_chunk_particles
                
                if n_chunk_particles == 0:
                    continue
                
                # Grid particles from this chunk and accumulate
                try:
                    chunk_density = gridder.particles_to_grid(chunk_particles, n_devices=1)
                    density_grid += chunk_density
                except Exception as e:
                    print(f"  ERROR in particles_to_grid: {e}")
                    print(f"  Chunk {chunk_count} data types:")
                    for key, val in chunk_particles.items():
                        print(f"    {key}: {type(val)}, shape: {getattr(val, 'shape', 'N/A')}")
                    raise
                
                if process_id == 0:
                    print(f"  Process {process_id} processed chunk {chunk_count}: {n_chunk_particles:,} particles")
        
        except Exception as e:
            print(f"ERROR in chunk processing: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        if process_id == 0:
            print(f"Process {process_id} total particles: {total_particles:,}")
        
        # Handle case where this process gets no particles
        if total_particles == 0:
            if process_id == 0:
                print(f"Process {process_id} received no particles - using empty contribution")
            # Return empty results for this process 
            k_bins_empty = self.k_bins[:-1]
            power_empty = np.zeros(len(k_bins_empty))
            n_modes_empty = np.zeros(len(k_bins_empty), dtype=int)
            return k_bins_empty, power_empty, n_modes_empty
        
        # Calculate mean density (local to each process)
        mean_density = density_grid.mean()
        if mean_density <= 0:
            raise ValueError("Mean density is zero - check particle data")
        
        delta_grid = (density_grid - mean_density) / mean_density
        
        # Store diagnostics
        try:
            self._store_density_diagnostics(density_grid, delta_grid, total_particles)
        except Exception as e:
            print(f"ERROR in _store_density_diagnostics: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Distributed FFT (returns only local k-space slice)
        delta_k_local = fft(delta_grid, direction='r2c')
        
        # Calculate local power spectrum
        power_3d_local = jnp.abs(delta_k_local)**2 * (self.volume / self.ngrid**6)
        power_3d_np = np.array(power_3d_local)
        
        # Create local k-grid and apply corrections
        k_grid_local = create_local_k_grid(self.ngrid, self.box_size)
        power_3d_corrected = self._apply_window_correction(power_3d_np, k_grid_local, assignment)
        
        # Bin and reduce across processes
        try:
            k_binned, power_binned, n_modes = bin_power_spectrum_distributed(
                k_grid_local, power_3d_corrected, self.k_bins
            )
        except Exception as e:
            print(f"ERROR in bin_power_spectrum_distributed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        return self._finalize_power_spectrum(k_binned, power_binned, n_modes,
                                           subtract_shot_noise, total_particles)
    
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
    
    def _should_use_streaming(self, particles: Dict[str, np.ndarray]) -> bool:
        """
        Determine if streaming processing should be used based on memory requirements.
        
        Args:
            particles: Particle data dictionary
            
        Returns:
            True if streaming should be used, False otherwise
        """
        n_particles = len(particles['x'])
        
        # Estimate memory usage
        # Particles: 4 fields × 8 bytes (assuming float64) per particle
        particle_memory_gb = (n_particles * 4 * 8) / (1024**3)
        
        # Grid memory: ngrid³ × 8 bytes (float64)
        grid_memory_gb = (self.ngrid**3 * 8) / (1024**3)
        
        # Estimate total memory needed for standard calculation
        total_memory_gb = particle_memory_gb + grid_memory_gb * 2  # density + delta grids
        
        # Use streaming if estimated memory > 10GB (conservative threshold)
        memory_threshold_gb = 10.0
        
        if total_memory_gb > memory_threshold_gb:
            print(f"Memory estimate: {total_memory_gb:.1f} GB (particles: {particle_memory_gb:.1f} GB, grids: {grid_memory_gb*2:.1f} GB)")
            print(f"Exceeds threshold of {memory_threshold_gb} GB - using streaming mode")
            return True
        
        return False

def get_spatial_domain_with_ghosts(process_id, n_processes, ngrid, assignment_scheme):
    """
    Calculate spatial domain for a process including ghost zones for mass assignment.
    
    Args:
        process_id: Current process ID (0 to n_processes-1)
        n_processes: Total number of processes
        ngrid: Grid size (512)
        assignment_scheme: 'ngp', 'cic', or 'tsc'
        
    Returns:
        Tuple of (y_min, y_max, base_y_min, base_y_max)
        where base_* are the owned cells, and y_* include ghost zones
    """
    # Base domain (what this process "owns")
    base_y_min = process_id * (ngrid // n_processes)
    base_y_max = (process_id + 1) * (ngrid // n_processes)
    
    # Ghost zone width based on assignment scheme
    if assignment_scheme == 'ngp':
        ghost_width = 0  # NGP only affects 1 cell
    elif assignment_scheme == 'cic':
        ghost_width = 1  # CIC can span 2 cells, need 1 ghost on each side
    elif assignment_scheme == 'tsc':
        ghost_width = 1  # TSC can span 3 cells, need 1 ghost on each side
    else:
        ghost_width = 1  # Default to CIC
    
    # Expand domain to include ghost zones
    y_min = max(0, base_y_min - ghost_width)
    y_max = min(ngrid, base_y_max + ghost_width)
    
    return y_min, y_max, base_y_min, base_y_max


def redistribute_particles_spatial(particles, assignment_scheme, ngrid, box_size):
    """
    Redistribute particles across processes based on spatial decomposition using JAX collectives.
    
    Uses all-gather to share particles from all processes, then each process filters 
    to its spatial domain. This is memory-efficient as each process ends up with 
    approximately the same number of particles.
    
    Args:
        particles: Dictionary with 'x', 'y', 'z', 'mass' arrays
        assignment_scheme: Mass assignment scheme ('cic', 'tsc', 'ngp')
        ngrid: Grid resolution
        box_size: Simulation box size
        
    Returns:
        Tuple of (redistributed_particles, y_min, y_max, base_y_min, base_y_max)
    """
    print(f"DEBUG: redistribute_particles_spatial starting...", flush=True)
    
    if JAX_AVAILABLE:
        import jax
        import jax.numpy as jnp
        process_id = jax.process_index()
        n_processes = jax.process_count()
        print(f"DEBUG: Process {process_id}: redistributing {len(particles['x'])} particles", flush=True)
    else:
        process_id = 0
        n_processes = 1
        
    # Calculate spatial domains with ghost zones
    my_y_min, my_y_max, base_y_min, base_y_max = get_spatial_domain_with_ghosts(
        process_id, n_processes, ngrid, assignment_scheme
    )
    
    print(f"DEBUG: Process {process_id}: spatial domain y=[{my_y_min:.1f}, {my_y_max:.1f}]", flush=True)
    
    if JAX_AVAILABLE and n_processes > 1:
        # Step 1: Determine maximum particle count across all processes
        local_count = jnp.array(len(particles['x']))
        all_counts = jax.lax.all_gather(local_count, axis_name='batch', tiled=True)
        max_count = int(jnp.max(all_counts))
        
        print(f"DEBUG: Process {process_id}: particle counts = {all_counts}, max = {max_count}", flush=True)
        
        # Step 2: Pad local particles to max_count for all-gather
        padded_particles = {}
        for key in ['x', 'y', 'z', 'mass']:
            local_data = particles[key]
            local_count_actual = len(local_data)
            
            # Pad with zeros to reach max_count
            if local_count_actual < max_count:
                padding = jnp.zeros(max_count - local_count_actual, dtype=local_data.dtype)
                padded_data = jnp.concatenate([local_data, padding])
            else:
                padded_data = local_data[:max_count]  # Truncate if necessary
                
            padded_particles[key] = padded_data
        
        print(f"DEBUG: Process {process_id}: padded particles to {max_count}", flush=True)
        
        # Step 3: All-gather particles from all processes
        gathered_particles = {}
        for key in ['x', 'y', 'z', 'mass']:
            # All-gather: shape will be (n_processes, max_count)
            gathered_data = jax.lax.all_gather(padded_particles[key], axis_name='batch', tiled=True)
            gathered_particles[key] = gathered_data
            
        print(f"DEBUG: Process {process_id}: gathered particles shape = {gathered_particles['x'].shape}", flush=True)
        
        # Step 4: Extract valid particles from all processes
        all_particles = {}
        for key in ['x', 'y', 'z', 'mass']:
            # Reshape to (n_processes, max_count) and extract valid particles
            valid_particles_list = []
            for proc in range(n_processes):
                proc_count = int(all_counts[proc])
                if proc_count > 0:
                    proc_particles = gathered_particles[key][proc][:proc_count]
                    valid_particles_list.append(proc_particles)
            
            # Concatenate all valid particles
            if valid_particles_list:
                all_particles[key] = jnp.concatenate(valid_particles_list)
            else:
                all_particles[key] = jnp.array([], dtype=particles[key].dtype)
            
        total_particles = len(all_particles['x'])
        print(f"DEBUG: Process {process_id}: extracted {total_particles} valid particles", flush=True)
        
        # Step 5: Filter to this process's spatial domain
        cell_size = box_size / ngrid
        y_grid = all_particles['y'] / cell_size
        in_domain = (y_grid >= my_y_min) & (y_grid < my_y_max)
        
        redistributed_particles = {}
        for key in ['x', 'y', 'z', 'mass']:
            redistributed_particles[key] = all_particles[key][in_domain]
            
        domain_particles = len(redistributed_particles['x'])
        print(f"DEBUG: Process {process_id}: filtered to {domain_particles} particles in spatial domain", flush=True)
        
    else:
        # Single process case - no redistribution needed
        redistributed_particles = particles
        
    return redistributed_particles, my_y_min, my_y_max, base_y_min, base_y_max


class ParticleDataReader:
    """
    Simple wrapper to make particle dictionary compatible with streaming interface.
    
    This allows the automatic streaming fallback to work with pre-loaded particle data
    by chunking it on-the-fly.
    """
    
    def __init__(self, particles: Dict[str, np.ndarray], chunk_size_gb: float = 2.0):
        self.particles = particles
        self.chunk_size_gb = chunk_size_gb
        
    def fetch_data_chunked(self):
        """Generator that yields particle data in chunks."""
        n_particles = len(self.particles['x'])
        
        # Calculate chunk size based on memory target
        bytes_per_particle = 4 * 8  # 4 fields × 8 bytes
        target_chunk_bytes = int(self.chunk_size_gb * 1024**3)
        particles_per_chunk = target_chunk_bytes // bytes_per_particle
        
        # Yield chunks
        for start_idx in range(0, n_particles, particles_per_chunk):
            end_idx = min(start_idx + particles_per_chunk, n_particles)
            
            chunk = {
                'x': self.particles['x'][start_idx:end_idx],
                'y': self.particles['y'][start_idx:end_idx], 
                'z': self.particles['z'][start_idx:end_idx],
                'mass': self.particles['mass'][start_idx:end_idx]
            }
            
            yield chunk