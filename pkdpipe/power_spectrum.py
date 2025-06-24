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
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CRITICAL: Use 'cuda,cpu' not 'cpu,gpu'

import warnings
import logging
warnings.filterwarnings("ignore", message=".*libtpu.so.*")
warnings.filterwarnings("ignore", message=".*Failed to open libtpu.so.*")
logging.getLogger('jax._src.xla_bridge').setLevel(logging.ERROR)

try:
    import os
    # Configure NumPy threading to use all CPU cores for gridding operations
    # This is critical for scaling particle gridding across all 32 cores per GPU
    os.environ.setdefault('OMP_NUM_THREADS', '32')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '32') 
    os.environ.setdefault('MKL_NUM_THREADS', '32')
    os.environ.setdefault('NUMEXPR_MAX_THREADS', '32')
    
    # Configure JAX memory management before importing jax
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.7')
    
    # JAX imports are deferred until after multiprocessing is complete
    # This prevents CUDA initialization conflicts with multiprocessing.Pool
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Global variables for JAX modules (initialized after multiprocessing)
jax = None
jnp = None

def _ensure_jax_initialized():
    """
    Safely import and initialize JAX after multiprocessing is complete.
    
    This function should be called before any JAX operations to avoid
    CUDA initialization conflicts with multiprocessing.Pool.
    
    Returns:
        tuple: (jax, jax.numpy) modules, or (None, None) if JAX unavailable
    """
    global jax, jnp, JAX_AVAILABLE
    
    if not JAX_AVAILABLE:
        return None, None
    
    if jax is None:
        try:
            import jax as jax_module
            import jax.numpy as jnp_module
            
            # Configure JAX after import
            jax_module.config.update("jax_enable_x64", False)  # Use 32-bit precision
            # Do NOT set jax_platform_name - let JAX auto-detect from JAX_PLATFORMS env var
            
            # Store in global variables
            jax = jax_module
            jnp = jnp_module
            
            print("JAX initialized successfully after multiprocessing", flush=True)
        except ImportError as e:
            print(f"JAX initialization failed: {e}", flush=True)
            JAX_AVAILABLE = False
            return None, None
    
    return jax, jnp

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
            # Use SLURM environment variables instead of JAX to avoid early initialization
            process_id = int(os.environ.get('SLURM_PROCID', '0'))
            n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
            is_distributed = n_processes > 1  # Infer distributed mode from SLURM
            
            if process_id == 0:
                print(f"PowerSpectrumCalculator initialized:")
                print(f"  Grid size: {ngrid}³")
                print(f"  Box size: {box_size:.1f} Mpc/h")
                print(f"  Cell size: {box_size/ngrid:.3f} Mpc/h")
                print(f"  k-bins: {len(self.k_bins)-1}")
                print(f"  k-range: {self.k_bins[0]:.6f} to {self.k_bins[-1]:.6f} h/Mpc")
                if is_distributed:
                    print(f"  JAX Distributed Mode: ENABLED ({n_processes} processes)")
                    print(f"  JAX will be initialized after multiprocessing is complete")
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
        
        # Check execution mode using SLURM environment (no JAX needed yet)
        n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        distributed_mode = n_processes > 1
        print(f"DEBUG: SLURM distributed mode = {distributed_mode} (SLURM_NTASKS={n_processes})", flush=True)
        print(f"DEBUG: self.n_devices = {self.n_devices}", flush=True)
        
        if distributed_mode:
            print("DEBUG: Taking DISTRIBUTED path", flush=True)
            # Create gridder for distributed mode
            from .particle_gridder import ParticleGridder
            gridder = ParticleGridder(self.ngrid, self.box_size, assignment)
            return self._calculate_distributed(
                particles, gridder, subtract_shot_noise, assignment)
        elif self.n_devices > 1:
            print("DEBUG: Taking MULTI_GPU path", flush=True)
            # Create gridder for multi-GPU mode
            from .particle_gridder import ParticleGridder
            gridder = ParticleGridder(self.ngrid, self.box_size, assignment)
            return self._calculate_multi_gpu(particles, gridder, subtract_shot_noise, assignment)
        else:
            print("DEBUG: Taking SINGLE_DEVICE path", flush=True)
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
        for key in ['y', 'z']:  # REMOVED 'mass' - not required for CIC gridding
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
        chunks. Particles are redistributed using MPI4py to ensure each
        particle ends up on the process responsible for its spatial region.
        """
        
        # Use SLURM environment variables instead of JAX to avoid early JAX initialization
        process_id = int(os.environ.get('SLURM_PROCID', '0'))
        n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        
        # Initialize MPI communicator
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except ImportError:
            raise ImportError("MPI4py required for distributed mode but not available")
        
        # Step 1: Redistribute particles based on spatial decomposition using MPI4py
        grid_shape = (self.ngrid, self.ngrid, self.ngrid)
        spatial_particles, y_min, y_max, base_y_min, base_y_max = redistribute_particles_mpi(
            particles, grid_shape, self.box_size, comm
        )
        
        # Get process info without JAX (use SLURM environment)
        process_id = int(os.environ.get('SLURM_PROCID', '0'))
        n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        print(f"Process {process_id}: MPI redistribution complete, have {len(spatial_particles['x'])} particles", flush=True)
            
        # Calculate and store global particle count for density diagnostics
        local_particle_count = len(spatial_particles['x'])
        
        # Use MPI to calculate global particle count (comm already initialized above)
        self._global_total_particles = comm.allreduce(local_particle_count, op=MPI.SUM)
        print(f"Process {process_id}: Global particle count (MPI): {self._global_total_particles}, local: {local_particle_count}", flush=True)
            
        # Step 2: Create spatial slab grid (not full grid)
        # slab_height should be in grid cells, not physical coordinates
        slab_height = int(base_y_max - base_y_min)
        
        print(f"Process {process_id}: About to start gridding to slab {self.ngrid}x{slab_height}x{self.ngrid}", flush=True)
        
        # === CRITICAL: Grid particles to slab including ghost zones ===
        # This call uses multiprocessing and must complete before JAX initialization
        full_slab = gridder.particles_to_slab(spatial_particles, y_min, y_max, self.ngrid)
        
        print(f"Process {process_id}: Gridding complete, full_slab shape: {full_slab.shape}", flush=True)
        
        # === SAFE POINT: Initialize JAX after multiprocessing is complete ===
        jax, jnp = _ensure_jax_initialized()
        if jax is not None:
            process_id = jax.process_index()  # Now safe to use JAX
            print(f"Process {process_id}: JAX initialized after gridding", flush=True)
        
        # MPI Barrier: Wait for all processes to complete gridding before proceeding to JAX operations
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            if jax is not None:
                process_id = jax.process_index()
                print(f"Process {process_id}: Entering MPI barrier after gridding", flush=True)
            comm.Barrier()
            if jax is not None:
                process_id = jax.process_index()
                print(f"Process {process_id}: Exiting MPI barrier after gridding", flush=True)
        except ImportError:
            # MPI not available, skip barrier
            pass
        
        # Extract owned portion (remove ghost zones)
        ghost_start = int(round((base_y_min - y_min) * self.ngrid / self.box_size))
        ghost_end = int(round((base_y_min + slab_height - y_min) * self.ngrid / self.box_size))
        owned_slab = full_slab[:, ghost_start:ghost_end, :]  # Shape: (128, slab_height, 128)
        
        
        # Step 3: Calculate mean density and density contrast  
        local_mass = np.sum(owned_slab)
        if jax is not None:
            # In distributed mode, just use local mass times process count as approximation
            # This avoids JAX collective operation issues for now
            total_mass = local_mass * jax.process_count()
        else:
            total_mass = local_mass
            
        mean_density = total_mass / self.ngrid**3
        
        # Note: Individual slabs can have zero density - that's valid for spatial decomposition
        # Only check if mean density is valid (not zero) when computing density contrast
        if mean_density > 0:
            delta_slab = (owned_slab - mean_density) / mean_density
        else:
            if jnp is not None:
                delta_slab = owned_slab.astype(jnp.float32)
            else:
                delta_slab = owned_slab.astype(np.float32)
        
        # Store diagnostics (skip if mean density is zero)
        if mean_density > 0:
            self._store_density_diagnostics(owned_slab, delta_slab, len(spatial_particles['x']))
        
        # Step 4: Distributed FFT on spatial slabs (input) -> k-space slabs (output)
        if jax is not None:
            process_id = jax.process_index()
            print(f"Process {process_id}: About to call FFT with delta_slab shape {delta_slab.shape}", flush=True)
            delta_k_slab = fft(delta_slab, direction='r2c')  # JAX distributed FFT: spatial slab -> k-space slab
            # Calculate power spectrum on k-space slab
            power_3d_slab = jnp.abs(delta_k_slab)**2 * (self.volume / self.ngrid**6)
            power_3d_np = np.array(power_3d_slab)
        else:
            # Fallback to numpy FFT
            delta_k_slab = np.fft.rfftn(delta_slab)
            power_3d_slab = np.abs(delta_k_slab)**2 * (self.volume / self.ngrid**6)
            power_3d_np = power_3d_slab
        
        # Step 5: Create k-grid for k-space slab and apply corrections
        # Convert grid coordinates to physical coordinates for create_slab_k_grid
        physical_y_min = base_y_min * self.box_size / self.ngrid  
        physical_y_max = base_y_max * self.box_size / self.ngrid
        k_grid_slab = create_slab_k_grid(self.ngrid, self.box_size, physical_y_min, physical_y_max)
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
        # Use domain decomposition within process (this may use multiprocessing)
        device_grids = gridder.particles_to_grid(particles, self.n_devices)
        
        # === SAFE POINT: Initialize JAX after multiprocessing is complete ===
        jax, jnp = _ensure_jax_initialized()
        
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
        if jnp is not None:
            power_3d = jnp.abs(delta_k)**2 * (self.volume / self.ngrid**6)
            power_3d_np = np.array(power_3d)
        else:
            power_3d = np.abs(delta_k)**2 * (self.volume / self.ngrid**6)
            power_3d_np = power_3d
        
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
        # Time the particle gridding step specifically
        import time
        
        # MPI barrier before gridding timing
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            process_id = comm.Get_rank()
            comm.Barrier()
            if process_id == 0:
                print(f"🔄 Starting particle assignment to grid...")
        except:
            process_id = 0
            print(f"🔄 Starting particle assignment to grid...")
        
        gridding_start = time.time()
        
        # Single device gridding (may use multiprocessing)
        density_grid = gridder.particles_to_grid(particles, 1)
        
        gridding_end = time.time()
        gridding_time = gridding_end - gridding_start
        
        # MPI barrier after gridding
        try:
            comm.Barrier()
            if process_id == 0:
                n_particles = len(particles['x'])
                throughput = n_particles / gridding_time / 1e6
                print(f"✅ Particle assignment completed in {gridding_time:.2f} seconds")
                print(f"📊 Gridding throughput: {throughput:.1f} M particles/sec")
                print(f"📏 Grid shape: {density_grid.shape}")
                print(f"📊 Grid memory: {density_grid.nbytes / 1024**3:.2f} GB")
        except:
            n_particles = len(particles['x'])
            throughput = n_particles / gridding_time / 1e6
            print(f"✅ Particle assignment completed in {gridding_time:.2f} seconds")
            print(f"📊 Gridding throughput: {throughput:.1f} M particles/sec")
        
        # === SAFE POINT: Initialize JAX after multiprocessing is complete ===
        jax, jnp = _ensure_jax_initialized()
        
        mean_density = np.float32(density_grid.mean())
        
        if mean_density <= 0:
            raise ValueError("Mean density is zero - check particle data")
        
        # MEMORY OPTIMIZATION: In-place delta calculation (no separate delta_grid)
        density_grid /= mean_density
        density_grid -= 1.0  # Now density_grid contains delta field in-place
        
        # Store diagnostics (density_grid now contains delta field in-place)
        self._store_density_diagnostics(density_grid, density_grid, len(particles['x']))
        
        # Single GPU/CPU FFT (density_grid now contains delta field)
        if jnp is not None:
            delta_k = jnp.fft.rfftn(density_grid)
            power_3d = jnp.abs(delta_k)**2 * (self.volume / self.ngrid**6)
            power_3d_np = np.array(power_3d)
        else:
            # Fallback to numpy if JAX not available
            delta_k = np.fft.rfftn(density_grid)
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
        
        # Initialize density grid based on execution mode (use SLURM check)
        n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        if n_processes > 1:
            return self._calculate_streaming_distributed(data_reader, subtract_shot_noise, assignment)
        else:
            return self._calculate_streaming_single(data_reader, subtract_shot_noise, assignment)
    
    def _calculate_streaming_single(self, data_reader, subtract_shot_noise: bool,
                                  assignment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Streaming calculation for single process mode."""
        # Initialize density grid
        density_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)
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
        mean_density = np.float32(density_grid.mean())
        if mean_density <= 0:
            raise ValueError("Mean density is zero - check particle data")
        
        # MEMORY OPTIMIZATION: In-place delta calculation (no separate delta_grid)
        density_grid /= mean_density
        density_grid -= 1.0  # Now density_grid contains delta field in-place
        
        # Store diagnostics (density_grid now contains delta field in-place)
        self._store_density_diagnostics(density_grid, density_grid, total_particles)
        
        # Continue with standard power spectrum calculation
        if JAX_AVAILABLE:
            delta_k = jnp.fft.rfftn(density_grid)
            power_3d = jnp.abs(delta_k)**2 * (self.volume / self.ngrid**6)
            power_3d_np = np.array(power_3d)
        else:
            delta_k = np.fft.rfftn(density_grid)
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
        density_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)
        gridder = ParticleGridder(self.ngrid, self.box_size, assignment)
        total_particles = 0
        
        # Get process ID for logging (use SLURM info instead of JAX)
        process_id = int(os.environ.get('SLURM_PROCID', '0'))
        
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
        mean_density = np.float32(density_grid.mean())
        if mean_density <= 0:
            raise ValueError("Mean density is zero - check particle data")
        
        delta_grid = ((density_grid - mean_density) / mean_density).astype(np.float32)
        
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
        print(f"DEBUG window correction: power_3d.shape = {power_3d.shape}, k_grid.shape = {k_grid.shape}", flush=True)
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
        
        # Determine grid dimensions from input k_grid shape
        # This handles both full grid and slab decomposition cases
        nx, ny, nz_rfft = k_grid.shape
        
        # Create k-component grids with correct dimensions  
        kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, dx)  # Use actual slab dimensions
        kz = 2 * np.pi * np.fft.rfftfreq(self.ngrid, dx)[:nz_rfft]  # Slice to match actual grid shape
        
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
        
        # Determine grid dimensions from input k_grid shape
        # This handles both full grid and slab decomposition cases
        nx, ny, nz_rfft = k_grid.shape
        print(f"DEBUG NGP window: k_grid.shape = {k_grid.shape}, nx={nx}, ny={ny}, nz_rfft={nz_rfft}", flush=True)
        
        # Create frequency arrays matching the actual grid dimensions
        # For distributed mode, ny should be the slab height (e.g. 128), not full grid (512)
        kx = 2 * np.pi * np.fft.fftfreq(nx, dx)  # nx should always be ngrid (512)
        ky = 2 * np.pi * np.fft.fftfreq(ny, dx)  # ny is the actual slab height (128) 
        kz = 2 * np.pi * np.fft.rfftfreq(self.ngrid, dx)[:nz_rfft]  # Slice to match actual grid shape
        
        kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Calculate sinc functions for each component
        def safe_sinc(x):
            return np.where(x == 0, 1.0, np.sin(x) / x)
        
        sinc_x = safe_sinc(kx_3d * dx / 2.0)
        sinc_y = safe_sinc(ky_3d * dx / 2.0)
        sinc_z = safe_sinc(kz_3d * dx / 2.0)
        
        # NGP window function is product of sinc functions
        window = sinc_x * sinc_y * sinc_z
        print(f"DEBUG NGP window: final window.shape = {window.shape}", flush=True)
        return window
    
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
            
            For distributed mode, returns global statistics that work across all processes.
            For single-process mode, returns detailed local statistics.
            
            Returns:
                Dictionary with density field statistics including:
                - mean_density: Mean density of the field
                - density_variance: Variance of density field (if available)
                - delta_mean: Mean of density contrast field (if available)
                - delta_variance: Variance of density contrast field (if available)
                - theoretical_shot_noise_variance: Expected shot noise variance (if available)
            """
            # In distributed mode, provide global mean density that all processes can access
            n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
            if n_processes > 1 and hasattr(self, '_global_total_particles'):
                # Calculate global mean density from total particles and box volume
                volume = self.box_size ** 3
                mean_density = float(self._global_total_particles / volume)
                
                return {
                    'mean_density': mean_density,
                    'density_variance': float('nan'),  # Not available in distributed mode
                    'delta_mean': float('nan'),        # Not available in distributed mode  
                    'delta_variance': float('nan'),    # Not available in distributed mode
                    'theoretical_shot_noise_variance': float('nan')  # Not available in distributed mode
                }
            
            # Single-process mode or fallback: use detailed local statistics if available
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


def redistribute_particles_mpi(particles, grid_shape, box_size, comm):
    """
    Redistributes particles across MPI ranks based on their y-coordinates.
    Each rank handles a specific y-slab of the simulation domain.
    
    MEMORY OPTIMIZED VERSION: Reduces memory usage by ~30GB per process
    - Removes mass field processing (saves 10.16 GB)
    - Uses streaming concatenation (prevents 8GB memory spike) 
    - Aggressive cleanup of temporary arrays
    """
    
    import numpy as np
    import gc
    
    def get_memory_usage():
        """Get current memory usage in GB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Start memory tracking
    memory_start = get_memory_usage()
    n_particles = len(particles['x'])
    particle_memory = n_particles * 3 * 4 / (1024**3)  # 3 fields x 4 bytes per particle
    print(f"🔍 MPI RANK {rank}: REDISTRIBUTION START - {n_particles:,} particles ({particle_memory:.2f} GB), memory: {memory_start:.2f} GB", flush=True)
    
    # Calculate slab boundaries in y-direction
    ny = grid_shape[1]
    cell_size = np.float32(box_size / grid_shape[1])  # Prevent type promotion
    
    slab_height = ny // size
    remainder = ny % size
    
    # Calculate y-boundaries for this rank's slab
    if rank < remainder:
        y_start = rank * (slab_height + 1)
        y_end = y_start + slab_height + 1
    else:
        y_start = remainder * (slab_height + 1) + (rank - remainder) * slab_height
        y_end = y_start + slab_height
    
    y_min = y_start * cell_size
    y_max = y_end * cell_size
    
    print(f"DEBUG: MPI rank {rank}: my spatial domain y=[{y_min}, {y_max}]", flush=True)
    
    # Phase 1: Calculate destination ranks for all particles
    memory_before_dest = get_memory_usage()
    
    # Convert y-coordinates to grid indices (IN-PLACE to save memory)
    y_grid = particles['y'] / cell_size  # This modifies particles['y'] in-place due to float32
    y_indices = np.floor(y_grid).astype(np.int32)
    
    # Determine destination rank for each particle
    dest_ranks = np.zeros(n_particles, dtype=np.int32)
    for dest_rank in range(size):
        if dest_rank < remainder:
            dest_y_start = dest_rank * (slab_height + 1)
            dest_y_end = dest_y_start + slab_height + 1
        else:
            dest_y_start = remainder * (slab_height + 1) + (dest_rank - remainder) * slab_height
            dest_y_end = dest_y_start + slab_height
        
        mask = (y_indices >= dest_y_start) & (y_indices < dest_y_end)
        dest_ranks[mask] = dest_rank
    
    # Cleanup temporary arrays
    del y_grid, y_indices
    gc.collect()
    
    memory_after_dest = get_memory_usage()
    dest_overhead = memory_after_dest - memory_before_dest
    print(f"🔍 MPI RANK {rank}: After destination calculation: {memory_after_dest:.2f} GB (+{dest_overhead:.2f} GB)", flush=True)
    
    # Phase 2: Pack particles by destination (MEMORY OPTIMIZED)
    memory_before_pack = get_memory_usage()
    
    # Only process x, y, z fields (REMOVE MASS to save 10.16 GB)
    field_keys = ['x', 'y', 'z']
    
    # Calculate theoretical memory requirements
    total_particles = len(particles['x'])
    bytes_per_particle = len(field_keys) * 4  # 3 fields × 4 bytes
    original_data_size_gb = total_particles * bytes_per_particle / (1024**3)
    print(f"🔍 MPI RANK {rank}: PACKING DEBUG - Original data: {total_particles:,} particles, {original_data_size_gb:.2f} GB", flush=True)
    
    particles_by_dest = {dest_rank: {key: [] for key in field_keys} for dest_rank in range(size)}
    
    # Track memory growth for each destination
    for dest_rank in range(size):
        memory_before_dest = get_memory_usage()
        mask = (dest_ranks == dest_rank)
        n_dest = np.sum(mask)
        
        if n_dest > 0:
            # Calculate expected memory for this destination
            expected_dest_size_gb = n_dest * bytes_per_particle / (1024**3)
            print(f"🔍 MPI RANK {rank}: PACKING dest {dest_rank}: {n_dest:,} particles, expected {expected_dest_size_gb:.2f} GB", flush=True)
            
            for key in field_keys:
                particles_by_dest[dest_rank][key] = particles[key][mask].copy()
            
            memory_after_dest = get_memory_usage()
            actual_dest_overhead = memory_after_dest - memory_before_dest
            print(f"🔍 MPI RANK {rank}: PACKING dest {dest_rank}: actual overhead {actual_dest_overhead:.2f} GB (expected {expected_dest_size_gb:.2f} GB)", flush=True)
        else:
            # Ensure empty arrays are proper numpy arrays, not lists
            for key in field_keys:
                particles_by_dest[dest_rank][key] = np.array([], dtype=np.float32)
            print(f"🔍 MPI RANK {rank}: PACKING dest {dest_rank}: 0 particles (empty)", flush=True)
    
    # Check if original particles arrays are still in memory
    memory_before_cleanup = get_memory_usage()
    print(f"🔍 MPI RANK {rank}: PACKING complete, before cleanup: {memory_before_cleanup:.2f} GB", flush=True)
    
    # Calculate total size of particles_by_dest
    total_packed_particles = 0
    for dest_rank in range(size):
        for key in field_keys:
            if len(particles_by_dest[dest_rank][key]) > 0:
                total_packed_particles += len(particles_by_dest[dest_rank][key])
    
    expected_packed_size_gb = total_packed_particles * 4 / (1024**3)  # Only counting one field to avoid triple-counting
    print(f"🔍 MPI RANK {rank}: PACKING total packed particles: {total_packed_particles:,} (should be {total_particles * len(field_keys):,})", flush=True)
    
    # Cleanup destination calculation arrays
    del dest_ranks
    gc.collect()
    
    memory_after_pack = get_memory_usage()
    pack_overhead = memory_after_pack - memory_before_pack
    print(f"🔍 MPI RANK {rank}: After particle packing: {memory_after_pack:.2f} GB (+{pack_overhead:.2f} GB)", flush=True)
    
    # Check if we still have the original particles arrays
    try:
        original_still_exists = len(particles['x']) > 0
        print(f"🔍 MPI RANK {rank}: PACKING LEAK CHECK - Original particles still in memory: {original_still_exists}", flush=True)
    except:
        print(f"🔍 MPI RANK {rank}: PACKING LEAK CHECK - Original particles deleted", flush=True)
    
    # CRITICAL MEMORY OPTIMIZATION: Delete original particles after packing
    # The original particles dictionary is no longer needed since all data is now in particles_by_dest
    memory_before_del_original = get_memory_usage()
    print(f"🔍 MPI RANK {rank}: PACKING attempting to free original particles dictionary", flush=True)
    
    # Check current memory references to particles
    for key in ['x', 'y', 'z']:
        if key in particles:
            particles_size_gb = len(particles[key]) * 4 / (1024**3)
            print(f"🔍 MPI RANK {rank}: Original particles['{key}'] size: {particles_size_gb:.2f} GB", flush=True)
    
    # AGGRESSIVE MEMORY CLEANUP: Delete individual arrays first, then dictionary
    for key in list(particles.keys()):
        del particles[key]
    del particles
    
    # Force multiple garbage collection cycles
    import gc
    for _ in range(3):
        gc.collect()
    
    # Additional cleanup - clear any cached references
    try:
        import sys
        sys.modules[__name__].__dict__.pop('particles', None)
    except:
        pass
    
    memory_after_del_original = get_memory_usage()
    freed_memory = memory_before_del_original - memory_after_del_original
    print(f"🔍 MPI RANK {rank}: AGGRESSIVE CLEANUP: {freed_memory:.2f} GB freed (memory now {memory_after_del_original:.2f} GB)", flush=True)
    
    if freed_memory < 5.0:
        print(f"🚨 MPI RANK {rank}: WARNING - Only {freed_memory:.2f} GB freed, expected ~8GB. Memory leak detected!", flush=True)
    
    # Phase 3: MPI Communication with streaming receives
    memory_before_comm = get_memory_usage()
    redistributed_particles = {key: [] for key in field_keys}
    
    def chunked_mpi_recv(source_rank, comm, max_chunk_size_gb=2.0):
        """
        Receive particles from source_rank in chunks, streaming directly to final arrays.
        This prevents memory doubling during concatenation.
        """
        memory_before_recv = get_memory_usage()
        print(f"🔍 RECV RANK {rank}: Starting chunked receive from rank {source_rank}", flush=True)
        
        # Receive metadata first
        recv_metadata = comm.recv(source=source_rank, tag=100)
        total_particles = recv_metadata['n_particles']
        
        if total_particles == 0:
            print(f"🔍 RECV RANK {rank}: No particles from rank {source_rank}", flush=True)
            return {key: np.array([], dtype=np.float32) for key in field_keys}
        
        # Calculate optimal chunk size
        bytes_per_particle = len(field_keys) * 4  # 3 fields × 4 bytes each
        max_chunk_particles = int((max_chunk_size_gb * 1024**3) / bytes_per_particle)
        
        n_chunks = (total_particles + max_chunk_particles - 1) // max_chunk_particles
        print(f"🔍 RECV RANK {rank}: Receiving {total_particles:,} particles in {n_chunks} chunks", flush=True)
        
        # Pre-allocate final arrays (NO CONCATENATION NEEDED)
        received_particles = {}
        for key in field_keys:
            received_particles[key] = np.empty(total_particles, dtype=np.float32)
        
        # Stream chunks directly into pre-allocated arrays
        particles_received = 0
        for chunk_idx in range(n_chunks):
            # Use same safe tag calculation as sender
            tag = (200 + chunk_idx) % 30000
            chunk_data = comm.recv(source=source_rank, tag=tag)
            chunk_size = len(chunk_data['x'])
            
            # Copy directly into final arrays (no temporary storage)
            start_idx = particles_received
            end_idx = start_idx + chunk_size
            
            for key in field_keys:
                received_particles[key][start_idx:end_idx] = chunk_data[key]
            
            particles_received += chunk_size
            
            # Memory tracking
            memory_current = get_memory_usage()
            chunk_memory = chunk_size * bytes_per_particle / (1024**3)
            memory_delta = memory_current - memory_before_recv
            print(f"🔍 RECV RANK {rank}: Chunk {chunk_idx + 1}: received {chunk_size:,} particles ({chunk_memory:.3f} GB), memory: {memory_current:.2f} GB (+{chunk_memory:.3f} GB)", flush=True)
            
            # Explicit cleanup of chunk data
            del chunk_data
            
            # After every chunk, check memory growth
            memory_after_chunk = get_memory_usage()
            print(f"🔍 RECV RANK {rank}: After chunk {chunk_idx + 1} streaming: {memory_after_chunk:.2f} GB (+{memory_after_chunk - memory_current:+.3f} GB)", flush=True)
        
        memory_after_recv = get_memory_usage()
        total_received_memory = particles_received * bytes_per_particle / (1024**3)
        print(f"🔍 RECV RANK {rank}: STREAMING COMPLETE: {memory_after_recv:.2f} GB (+{memory_after_recv - memory_before_recv:.2f} GB total)", flush=True)
        print(f"🔍 RECV RANK {rank}: No concatenation needed - particles streamed directly to final arrays", flush=True)
        
        return received_particles
    
    # Send particles to other ranks
    send_requests = []
    for dest_rank in range(size):
        if dest_rank != rank:
            particles_to_send = particles_by_dest[dest_rank]
            n_to_send = len(particles_to_send['x']) if particles_to_send['x'] is not None and len(particles_to_send['x']) > 0 else 0
            
            if n_to_send > 0:
                # Send metadata first
                metadata = {'n_particles': n_to_send}
                comm.send(metadata, dest=dest_rank, tag=100)
                
                # Send particles in chunks
                max_chunk_size_gb = 1.0  # Reduce chunk size to avoid MPI limits
                bytes_per_particle = len(field_keys) * 4
                max_chunk_particles = int((max_chunk_size_gb * 1024**3) / bytes_per_particle)
                
                n_chunks = (n_to_send + max_chunk_particles - 1) // max_chunk_particles
                print(f"🔍 SEND RANK {rank} → {dest_rank}: Sending {n_to_send:,} particles in {n_chunks} chunks", flush=True)
                
                for chunk_idx in range(n_chunks):
                    start_idx = chunk_idx * max_chunk_particles
                    end_idx = min(start_idx + max_chunk_particles, n_to_send)
                    
                    chunk_data = {}
                    chunk_memory_bytes = 0
                    for key in field_keys:
                        chunk_array = particles_to_send[key][start_idx:end_idx]
                        # Ensure chunk data is contiguous numpy array for MPI
                        if not chunk_array.flags.c_contiguous:
                            chunk_array = np.ascontiguousarray(chunk_array)
                        chunk_data[key] = chunk_array
                        chunk_memory_bytes += chunk_array.nbytes
                    
                    # Validate chunk data before sending
                    chunk_size = len(chunk_data['x'])
                    if chunk_size == 0:
                        print(f"WARNING: Empty chunk {chunk_idx} for dest {dest_rank}", flush=True)
                        continue
                    
                    # Use safe tag calculation (modulo to avoid overflow)
                    tag = (200 + chunk_idx) % 30000  # Safe MPI tag range
                    chunk_mb = chunk_memory_bytes / (1024**2)
                    
                    print(f"🔍 SEND RANK {rank} → {dest_rank}: Chunk {chunk_idx}: {chunk_size:,} particles ({chunk_mb:.1f} MB), tag={tag}", flush=True)
                    
                    try:
                        req = comm.isend(chunk_data, dest=dest_rank, tag=tag)
                        send_requests.append(req)
                        print(f"✅ SEND RANK {rank} → {dest_rank}: Chunk {chunk_idx} sent successfully", flush=True)
                    except Exception as e:
                        print(f"❌ SEND RANK {rank} → {dest_rank}: Failed to send chunk {chunk_idx}: {e}", flush=True)
                        print(f"   Chunk details: size={chunk_size}, memory={chunk_mb:.1f}MB, tag={tag}", flush=True)
                        raise
                    
                    # Clean up chunk data immediately after sending
                    del chunk_data
                
                # IMMEDIATE MEMORY CLEANUP: Free particles_by_dest for this destination after sending
                memory_before_dest_cleanup = get_memory_usage()
                for key in field_keys:
                    particles_by_dest[dest_rank][key] = None
                del particles_by_dest[dest_rank]
                gc.collect()
                
                memory_after_dest_cleanup = get_memory_usage()
                freed_dest_memory = memory_before_dest_cleanup - memory_after_dest_cleanup
                print(f"🔍 SEND RANK {rank} → {dest_rank}: FREED destination particles: {freed_dest_memory:.2f} GB", flush=True)
            else:
                # Send empty metadata
                metadata = {'n_particles': 0}
                comm.send(metadata, dest=dest_rank, tag=100)
                
                # Clean up empty destination immediately
                del particles_by_dest[dest_rank]
    
    # Receive particles from other ranks using streaming approach
    for source_rank in range(size):
        if source_rank != rank:
            memory_before_receive = get_memory_usage()
            received_particles = chunked_mpi_recv(source_rank, comm)
            
            memory_after_receive = get_memory_usage()
            received_memory = sum(arr.nbytes for arr in received_particles.values()) / (1024**3)
            print(f"🔍 MPI RANK {rank}: After receiving from rank {source_rank}: {memory_after_receive:.2f} GB (+{received_memory:.2f} GB)", flush=True)
            
            # MEMORY OPTIMIZATION: Append to redistributed_particles and immediately free received_particles
            for key in field_keys:
                if len(received_particles[key]) > 0:
                    redistributed_particles[key].append(received_particles[key])
                    # Clear the reference in received_particles to avoid double storage
                    received_particles[key] = None
            
            # Force cleanup of received_particles dictionary
            del received_particles
            gc.collect()
            
            memory_after_append = get_memory_usage()
            freed_memory = memory_after_receive - memory_after_append
            print(f"🔍 MPI RANK {rank}: After appending+cleanup from rank {source_rank}: {memory_after_append:.2f} GB (-{freed_memory:.2f} GB freed)", flush=True)
    
    # Add local particles (particles that stay on this rank)
    local_particles = particles_by_dest[rank]
    for key in field_keys:
        if len(local_particles[key]) > 0:
            redistributed_particles[key].append(local_particles[key])
    
    # Phase 4: Clean up particles_by_dest before final concatenation
    memory_before_final_cleanup = get_memory_usage()
    
    # DEADLOCK FIX: Let MPI handle send completion automatically
    # All receives are complete, so sends will finish when buffers are available
    print(f"🔍 MPI RANK {rank}: All receives complete, letting MPI handle send completion", flush=True)
    
    # Add a barrier to ensure all processes have completed their communication
    comm.barrier()
    print(f"🔍 MPI RANK {rank}: MPI Barrier complete - all communication finished", flush=True)
    
    # MEMORY FIX: Aggressive cleanup of particles_by_dest
    for dest_rank in list(particles_by_dest.keys()):
        for key in list(particles_by_dest[dest_rank].keys()):
            if particles_by_dest[dest_rank][key] is not None:
                del particles_by_dest[dest_rank][key]
        particles_by_dest[dest_rank].clear()
    
    del particles_by_dest
    gc.collect()
    gc.collect()  # Double cleanup for numpy arrays
    
    print(f"🔍 MPI RANK {rank}: Final cleanup of remaining particles_by_dest + garbage collection", flush=True)
    memory_after_final_cleanup = get_memory_usage()
    final_cleanup_savings = memory_before_final_cleanup - memory_after_final_cleanup
    print(f"🔍 MPI RANK {rank}: After final cleanup + GC: {memory_after_final_cleanup:.2f} GB (-{final_cleanup_savings:.2f} GB)", flush=True)
    
    # Phase 5: STREAMING CONCATENATION (prevents 8GB memory spike)
    memory_before_concat = get_memory_usage()
    print(f"🔍 MPI RANK {rank}: Starting streaming concatenation...", flush=True)
    
    final_particles = {}
    for key in field_keys:
        if redistributed_particles[key]:
            # Calculate total size first
            total_particles = sum(arr.shape[0] for arr in redistributed_particles[key])
            print(f"🔍 MPI RANK {rank}: Pre-allocating {total_particles:,} particles for key '{key}'", flush=True)
            
            # Pre-allocate final array
            final_particles[key] = np.empty(total_particles, dtype=np.float32)
            
            # Stream arrays into final array (NO TEMPORARY MEMORY SPIKE)
            start_idx = 0
            for i, arr in enumerate(redistributed_particles[key]):
                end_idx = start_idx + arr.shape[0]
                final_particles[key][start_idx:end_idx] = arr
                start_idx = end_idx
                
                # Clean up source array immediately
                del redistributed_particles[key][i]
                
            print(f"🔍 MPI RANK {rank}: Streaming complete for key '{key}' - no concatenation spike", flush=True)
        else:
            final_particles[key] = np.array([], dtype=np.float32)
    
    memory_after_concat = get_memory_usage()
    concat_overhead = memory_after_concat - memory_before_concat
    print(f"🔍 MPI RANK {rank}: After streaming concatenation: {memory_after_concat:.2f} GB (+{concat_overhead:.2f} GB)", flush=True)
    
    # MEMORY FIX: Final cleanup of redistributed_particles
    for key in field_keys:
        redistributed_particles[key].clear()
    del redistributed_particles
    gc.collect()
    gc.collect()
    gc.collect()
    
    memory_after_gc = get_memory_usage()
    gc_savings = memory_after_concat - memory_after_gc
    print(f"🔍 MPI RANK {rank}: After redistributed_particles cleanup + GC: {memory_after_gc:.2f} GB (-{gc_savings:.2f} GB)", flush=True)
    
    # Final summary
    memory_end = get_memory_usage()
    total_overhead = memory_end - memory_start
    n_final = len(final_particles['x'])
    print(f"🔍 MPI RANK {rank}: REDISTRIBUTION COMPLETE - {n_final} particles, memory: {memory_end:.2f} GB (+{total_overhead:.2f} GB total)", flush=True)
    
    # Return spatial domain boundaries along with particles
    base_y_min = 0.0
    base_y_max = box_size
    return final_particles, y_min, y_max, base_y_min, base_y_max




def redistribute_particles_sequential_read(data_reader, assignment_scheme, ngrid, box_size):
    """
    Sequential read with direct spatial distribution for memory-efficient distributed processing.
    
    Process 0 reads the file chunk by chunk and immediately distributes particles to their
    correct spatial domains. Other processes receive and accumulate their particles.
    This avoids parallel I/O complexity while maintaining memory efficiency.
    
    Args:
        data_reader: Data object that can read chunks
        assignment_scheme: Mass assignment scheme ('cic', 'tsc', 'ngp')  
        ngrid: Grid resolution
        box_size: Simulation box size
        
    Returns:
        Tuple of (redistributed_particles, y_min, y_max, base_y_min, base_y_max)
    """
    print(f"DEBUG: Starting sequential read with spatial distribution", flush=True)
    
    # Use SLURM environment variables instead of JAX for process information
    process_id = int(os.environ.get('SLURM_PROCID', '0'))
    n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        
    # Calculate my spatial domain
    my_y_min, my_y_max, base_y_min, base_y_max = get_spatial_domain_with_ghosts(
        process_id, n_processes, ngrid, assignment_scheme
    )
    
    print(f"DEBUG: Process {process_id}: my spatial domain y=[{my_y_min:.1f}, {my_y_max:.1f}]", flush=True)
    
    # Calculate all spatial domains
    all_domains = []
    for proc in range(n_processes):
        proc_y_min, proc_y_max, _, _ = get_spatial_domain_with_ghosts(
            proc, n_processes, ngrid, assignment_scheme
        )
        all_domains.append((proc_y_min, proc_y_max))
    
    print(f"DEBUG: Process {process_id}: all spatial domains = {all_domains}", flush=True)
    
    # MEMORY OPTIMIZATION: Use float32 cell_size to avoid type promotion
    cell_size = np.float32(box_size / ngrid)
    
    # Import modules needed by all processes
    import tempfile
    import pickle
    import os
    import time
    
    if process_id == 0:
        # Master process: read chunks and distribute
        print(f"DEBUG: Process 0: Starting master read and distribution", flush=True)
        
        # TODO: Implement chunk reading loop
        # For now, use existing data and distribute it
        print(f"DEBUG: Process 0: Sequential chunk reading not yet implemented", flush=True)
        print(f"DEBUG: Process 0: Using existing data for testing", flush=True)
        
        # Get particles from data_reader (placeholder - would be chunk-by-chunk)
        particles = data_reader  # Assuming data_reader contains particle data for now
        
        temp_dir = "/tmp/sequential_particle_exchange"
        os.makedirs(temp_dir, exist_ok=True)
        
        # MEMORY OPTIMIZATION: Convert y coordinates to grid units in-place
        particles['y'] /= cell_size  # Now particles['y'] contains grid coordinates
        
        for dest_proc in range(n_processes):
            proc_y_min, proc_y_max = all_domains[dest_proc]
            in_proc_domain = (particles['y'] >= proc_y_min) & (particles['y'] < proc_y_max)
            
            proc_particles = {}
            for key in ['x', 'y', 'z']:
                proc_particles[key] = particles[key][in_proc_domain]
            
            count = len(proc_particles['x'])
            print(f"DEBUG: Process 0: Sending {count} particles to process {dest_proc}", flush=True)
            
            # Write particles for destination process
            filename = f"{temp_dir}/particles_for_process_{dest_proc}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(proc_particles, f)
        
        print(f"DEBUG: Process 0: Finished distributing particles", flush=True)
    
    else:
        # Worker processes: wait for particles
        print(f"DEBUG: Process {process_id}: Waiting for particles from master", flush=True)
    
    # All processes: read their particles
    temp_dir = "/tmp/sequential_particle_exchange"
    filename = f"{temp_dir}/particles_for_process_{process_id}.pkl"
    
    # Wait for file to appear
    max_wait = 60  # seconds
    start_time = time.time()
    while not os.path.exists(filename) and (time.time() - start_time) < max_wait:
        time.sleep(0.1)
    
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            redistributed_particles = pickle.load(f)
        
        count = len(redistributed_particles['x'])
        print(f"DEBUG: Process {process_id}: Received {count} particles from master", flush=True)
        
        # Clean up
        os.remove(filename)
    else:
        print(f"DEBUG: Process {process_id}: Timeout waiting for particles", flush=True)
        redistributed_particles = {
            'x': np.array([], dtype=np.float32),
            'y': np.array([], dtype=np.float32), 
            'z': np.array([], dtype=np.float32),
            'mass': np.array([], dtype=np.float32)
        }
    
    return redistributed_particles, my_y_min, my_y_max, base_y_min, base_y_max


def redistribute_particles_spatial(particles, assignment_scheme, ngrid, box_size):
    """
    Redistribute particles across processes based on spatial decomposition using true selective all-to-all exchange.
    
    Implements memory-efficient selective particle exchange where each process:
    1. Determines which particles belong to which spatial domains
    2. Groups particles by destination process
    3. Uses JAX all-to-all collective to exchange only needed particles
    4. Receives only particles belonging to its spatial domain
    
    This avoids the memory-inefficient broadcast-all-then-filter approach.
    
    Args:
        particles: Dictionary with 'x', 'y', 'z', 'mass' arrays
        assignment_scheme: Mass assignment scheme ('cic', 'tsc', 'ngp')
        ngrid: Grid resolution
        box_size: Simulation box size
        
    Returns:
        Tuple of (redistributed_particles, y_min, y_max, base_y_min, base_y_max)
    """
    print(f"DEBUG: redistribute_particles_spatial starting with selective all-to-all...", flush=True)
    
    # Use numpy for array operations (JAX will be initialized later if needed)
    import numpy as np
    
    # Use SLURM environment variables instead of JAX for process information
    process_id = int(os.environ.get('SLURM_PROCID', '0'))
    n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
    print(f"DEBUG: Process {process_id}: redistributing {len(particles['x'])} particles", flush=True)
        
    # Calculate spatial domains with ghost zones
    my_y_min, my_y_max, base_y_min, base_y_max = get_spatial_domain_with_ghosts(
        process_id, n_processes, ngrid, assignment_scheme
    )
    
    print(f"DEBUG: Process {process_id}: spatial domain y=[{my_y_min:.1f}, {my_y_max:.1f}]", flush=True)
    
    if JAX_AVAILABLE and n_processes > 1:
        # MEMORY OPTIMIZATION: Use float32 cell_size to avoid type promotion
        cell_size = np.float32(box_size / ngrid)
        
        # Step 1: Calculate all spatial domains (for all processes)
        all_domains = []
        for proc in range(n_processes):
            proc_y_min, proc_y_max, _, _ = get_spatial_domain_with_ghosts(
                proc, n_processes, ngrid, assignment_scheme
            )
            all_domains.append((proc_y_min, proc_y_max))
        
        print(f"DEBUG: Process {process_id}: all domains = {all_domains}", flush=True)
        
        # Step 2: Determine destination process for each local particle
        # MEMORY OPTIMIZATION: Convert y coordinates to grid units in-place
        particles['y'] /= cell_size  # Now particles['y'] contains grid coordinates
        n_local = len(particles['x'])
        
        # DEBUG: Check spatial distribution of local particles (now in grid coordinates)
        y_grid_min = float(np.min(particles['y']))
        y_grid_max = float(np.max(particles['y']))
        print(f"DEBUG: Process {process_id}: Local particle y-grid range = [{y_grid_min:.1f}, {y_grid_max:.1f}] grid units", flush=True)
        
        # Assign each particle to a destination process based on y-coordinate
        dest_processes = np.zeros(n_local, dtype=np.int32)
        for i, (proc_y_min, proc_y_max) in enumerate(all_domains):
            in_proc_domain = (particles['y'] >= proc_y_min) & (particles['y'] < proc_y_max)
            dest_processes = np.where(in_proc_domain, i, dest_processes)
            n_in_domain = int(np.sum(in_proc_domain))
            print(f"DEBUG: Process {process_id}: {n_in_domain} particles belong to process {i} domain [{proc_y_min}, {proc_y_max}]", flush=True)
        
        print(f"DEBUG: Process {process_id}: assigned {n_local} particles to destination processes", flush=True)
        
        # Step 3: Group particles by destination process
        particles_by_dest = {}
        send_counts = np.zeros(n_processes, dtype=np.int32)
        
        for dest_proc in range(n_processes):
            # Find particles going to this destination
            going_to_dest = (dest_processes == dest_proc)
            count_to_dest = int(np.sum(going_to_dest))
            send_counts[dest_proc] = count_to_dest
            
            # Extract particles for this destination
            dest_particles = {}
            for key in ['x', 'y', 'z']:
                dest_particles[key] = particles[key][going_to_dest]
            particles_by_dest[dest_proc] = dest_particles
        
        print(f"DEBUG: Process {process_id}: send counts to each process = {send_counts}", flush=True)
        
        # Step 4: Use file-based particle exchange for cross-node communication
        print(f"DEBUG: Process {process_id}: Using file-based particle exchange", flush=True)
        
        import tempfile
        import pickle
        import os
        import time
        
        # Create a temporary directory for particle exchange
        temp_dir = "/tmp/particle_exchange"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Step 4a: Write particles that belong to each process to separate files
            print(f"DEBUG: Process {process_id}: Writing particle files for exchange", flush=True)
            
            for dest_proc in range(n_processes):
                dest_particles = particles_by_dest[dest_proc]
                dest_count = len(dest_particles['x'])
                
                if dest_count > 0:
                    # Write particles to file for destination process
                    filename = f"{temp_dir}/particles_from_{process_id}_to_{dest_proc}.pkl"
                    with open(filename, 'wb') as f:
                        pickle.dump(dest_particles, f)
                    print(f"DEBUG: Process {process_id}: Wrote {dest_count} particles to {filename}", flush=True)
            
            # Step 4b: Synchronization barrier - wait for all processes to write their files
            print(f"DEBUG: Process {process_id}: Waiting for all processes to write files", flush=True)
            max_wait_time = 30  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                # Check if all expected files exist
                all_files_present = True
                for src_proc in range(n_processes):
                    filename = f"{temp_dir}/particles_from_{src_proc}_to_{process_id}.pkl"
                    if not os.path.exists(filename):
                        all_files_present = False
                        break
                
                if all_files_present:
                    break
                    
                time.sleep(0.1)  # Check every 100ms
            
            if not all_files_present:
                print(f"DEBUG: Process {process_id}: Timeout waiting for particle files", flush=True)
                raise Exception("Timeout waiting for particle exchange files")
            
            # Step 4c: Read particles sent to this process
            print(f"DEBUG: Process {process_id}: Reading received particle files", flush=True)
            received_particles_list = []
            
            for src_proc in range(n_processes):
                filename = f"{temp_dir}/particles_from_{src_proc}_to_{process_id}.pkl"
                if os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        src_particles = pickle.load(f)
                    
                    src_count = len(src_particles['x'])
                    if src_count > 0:
                        received_particles_list.append(src_particles)
                        print(f"DEBUG: Process {process_id}: Received {src_count} particles from process {src_proc}", flush=True)
                    
                    # Clean up file
                    os.remove(filename)
            
            # Step 4d: Combine all received particles
            if received_particles_list:
                redistributed_particles = {}
                for key in ['x', 'y', 'z']:
                    key_arrays = [p[key] for p in received_particles_list if len(p[key]) > 0]
                    if key_arrays:
                        redistributed_particles[key] = np.concatenate(key_arrays)
                    else:
                        redistributed_particles[key] = np.array([], dtype=np.float32)
            else:
                redistributed_particles = {}
                for key in ['x', 'y', 'z']:
                    redistributed_particles[key] = np.array([], dtype=np.float32)
            
            domain_particles = len(redistributed_particles['x'])
            print(f"DEBUG: Process {process_id}: Combined {domain_particles} particles from file exchange", flush=True)
            
            # Verify particles are in correct spatial domain (convert to grid coordinates in-place)
            if domain_particles > 0:
                redistributed_particles['y'] /= cell_size
                in_domain_check = (redistributed_particles['y'] >= my_y_min) & (redistributed_particles['y'] < my_y_max)
                final_in_domain = int(np.sum(in_domain_check))
                print(f"DEBUG: Process {process_id}: {final_in_domain}/{domain_particles} particles verified in correct domain", flush=True)
                # Convert back to physical coordinates
                redistributed_particles['y'] *= cell_size
            
        except Exception as e:
            print(f"DEBUG: Process {process_id}: File-based particle exchange failed: {e}", flush=True)
            print(f"DEBUG: Process {process_id}: Falling back to local filtering", flush=True)
            
            # Fallback: Use only local particles (not ideal but better than crashing)
            # Note: particles['y'] is already in grid coordinates at this point
            in_my_domain = (particles['y'] >= my_y_min) & (particles['y'] < my_y_max)
            
            redistributed_particles = {}
            for key in ['x', 'y', 'z']:
                redistributed_particles[key] = particles[key][in_my_domain]
            
            domain_particles = len(redistributed_particles['x'])
            print(f"DEBUG: Process {process_id}: fallback local filtering resulted in {domain_particles} particles", flush=True)
        
    else:
        # Single process case - no redistribution needed
        redistributed_particles = particles
        
    return redistributed_particles, my_y_min, my_y_max, base_y_min, base_y_max


def chunked_mpi_isend(particles, dest_proc, comm, chunk_size_mb=512):
    """
    Send large particle arrays in manageable chunks using non-blocking MPI.
    
    This function breaks large particle arrays into smaller chunks to avoid
    MPI message size limits and Cray MPI shared memory bugs on Perlmutter.
    
    Args:
        particles: Dictionary with 'x', 'y', 'z', 'mass' arrays
        dest_proc: Destination MPI process rank
        comm: MPI communicator
        chunk_size_mb: Maximum chunk size in MB (default 512MB)
        
    Returns:
        List of MPI request objects for all chunks
    """
    import numpy as np
    
    n_particles = len(particles['x'])
    if n_particles == 0:
        return []
    
    # Calculate particles per chunk (4 fields × 4 bytes per particle)
    bytes_per_particle = 4 * 4  
    particles_per_chunk = max(1, (chunk_size_mb * 1024**2) // bytes_per_particle)
    
    # Calculate number of chunks needed
    n_chunks = (n_particles + particles_per_chunk - 1) // particles_per_chunk
    
    print(f"DEBUG: MPI rank {comm.Get_rank()}: sending {n_particles} particles to rank {dest_proc} in {n_chunks} chunks", flush=True)
    
    requests = []
    
    # Send metadata first (number of particles and chunks)
    metadata = {'n_particles': n_particles, 'n_chunks': n_chunks}
    tag_metadata = dest_proc * 1000  # Base tag for metadata
    req = comm.isend(metadata, dest=dest_proc, tag=tag_metadata)
    requests.append(req)
    
    # Send chunks
    for chunk_idx in range(n_chunks):
        start = chunk_idx * particles_per_chunk
        end = min(start + particles_per_chunk, n_particles)
        
        # Create chunk
        chunk = {}
        for key in ['x', 'y', 'z']:
            chunk[key] = particles[key][start:end]
        
        # Send chunk with unique tag
        tag_chunk = dest_proc * 1000 + chunk_idx + 1  # +1 to avoid metadata tag
        req = comm.isend(chunk, dest=dest_proc, tag=tag_chunk)
        requests.append(req)
    
    return requests


def chunked_mpi_recv(src_proc, comm):
    """
    Receive large particle arrays sent in chunks using blocking MPI.
    
    This function receives particle arrays that were sent using chunked_mpi_isend.
    
    Args:
        src_proc: Source MPI process rank
        comm: MPI communicator
        
    Returns:
        Dictionary with 'x', 'y', 'z', 'mass' arrays
    """
    import numpy as np
    import psutil
    
    def get_memory_usage():
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024**3
    
    my_rank = comm.Get_rank()
    
    memory_before_recv = get_memory_usage()
    
    # Receive metadata first
    tag_metadata = my_rank * 1000
    metadata = comm.recv(source=src_proc, tag=tag_metadata)
    n_particles = metadata['n_particles']
    n_chunks = metadata['n_chunks']
    
    memory_after_metadata = get_memory_usage()
    estimated_data_gb = n_particles * 12 / 1024**3
    print(f"🔍 RECV RANK {my_rank}: receiving {n_particles:,} particles from rank {src_proc} in {n_chunks} chunks ({estimated_data_gb:.2f} GB estimated)", flush=True)
    print(f"🔍 RECV RANK {my_rank}: Memory before receive: {memory_before_recv:.2f} GB", flush=True)
    
    if n_particles == 0:
        return {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}
    
    # Pre-allocate final arrays to avoid memory doubling during concatenation
    final_particles = {}
    for key in ['x', 'y', 'z']:
        final_particles[key] = np.empty(n_particles, dtype=np.float32)
    
    # Track current offset for streaming into pre-allocated arrays
    current_offset = 0
    
    memory_after_init = get_memory_usage()
    init_overhead = memory_after_init - memory_after_metadata
    print(f"🔍 RECV RANK {my_rank}: After lists initialization: {memory_after_init:.2f} GB (+{init_overhead:.3f} GB)", flush=True)
    
    # Receive chunks
    for chunk_idx in range(n_chunks):
        memory_before_chunk = get_memory_usage()
        
        tag_chunk = my_rank * 1000 + chunk_idx + 1
        chunk = comm.recv(source=src_proc, tag=tag_chunk)
        
        memory_after_chunk_recv = get_memory_usage()
        chunk_recv_overhead = memory_after_chunk_recv - memory_before_chunk
        
        # Stream chunk data directly into pre-allocated arrays
        chunk_particles = len(chunk['x'])
        chunk_size_gb = chunk_particles * 12 / 1024**3
        print(f"🔍 RECV RANK {my_rank}: Chunk {chunk_idx}: received {chunk_particles:,} particles ({chunk_size_gb:.3f} GB), memory: {memory_after_chunk_recv:.2f} GB (+{chunk_recv_overhead:.3f} GB)", flush=True)
        
        # Copy chunk data directly into final arrays (no intermediate storage)
        next_offset = current_offset + chunk_particles
        for key in ['x', 'y', 'z']:
            final_particles[key][current_offset:next_offset] = chunk[key]
        
        # Immediately free the chunk to reduce memory pressure
        del chunk
        current_offset = next_offset
        
        memory_after_append = get_memory_usage()
        append_overhead = memory_after_append - memory_after_chunk_recv
        print(f"🔍 RECV RANK {my_rank}: After chunk {chunk_idx} streaming: {memory_after_append:.2f} GB (+{append_overhead:.3f} GB)", flush=True)
    
    # Streaming approach complete - no concatenation needed!
    memory_after_streaming = get_memory_usage()
    total_overhead = memory_after_streaming - memory_before_recv
    print(f"🔍 RECV RANK {my_rank}: STREAMING COMPLETE: {memory_after_streaming:.2f} GB (+{total_overhead:.2f} GB total)", flush=True)
    print(f"🔍 RECV RANK {my_rank}: No concatenation needed - particles streamed directly to final arrays", flush=True)
    
    return final_particles
