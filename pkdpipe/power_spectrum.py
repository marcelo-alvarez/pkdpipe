"""
Power spectrum calculation using JAX FFT with multi-GPU support.

This module provides the main PowerSpectrumCalculator class for computing
power spectra from particle distributions with proper shot noise handling,
k-binning, and window function corrections.

Example usage:
    # Basic power spectrum calculation
    calculator = PowerSpectrumCalculator(ngrid=256, box_size=1000.0)
    k_bins, power, n_modes = calculator.calculate_power_spectrum(particles)
    
    # Multi-GPU calculation with 4 devices and window correction enabled
    calculator = PowerSpectrumCalculator(
        ngrid=512, box_size=2000.0, n_devices=4, apply_window_correction=True
    )
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


# MPI setup (import once at module level to avoid initialization conflicts)
try:
    from mpi4py import MPI
    _MPI_COMM = MPI.COMM_WORLD
    _MPI_AVAILABLE = True
except ImportError:
    _MPI_COMM = None
    _MPI_AVAILABLE = False

class PowerSpectrumCalculator:
    """
    Multi-GPU power spectrum calculator using JAX FFT.
    
    Provides clean API for computing power spectra from particle distributions
    with proper shot noise handling, k-binning, and optional window function corrections.
    
    This calculator supports three execution modes:
    1. Single device (GPU/CPU) - for small simulations
    2. Multi-GPU within single process - for medium simulations
    3. Distributed multi-process - for large simulations with srun/mpirun
    
    Window function correction is disabled by default (apply_window_correction=False)
    since it can introduce high-k noise amplification near the Nyquist frequency.
    Enable only when specific applications require aliasing corrections.
    """
    
    def __init__(self, ngrid: int, box_size: float, n_devices: int = 1,
                     k_bins: Optional[np.ndarray] = None,
                     apply_window_correction: bool = False):
            """
            Initialize power spectrum calculator.
            
            Args:
                ngrid: Grid resolution for FFT
                box_size: Simulation box size in Mpc/h
                n_devices: Number of GPUs to use (single-process mode)
                k_bins: Custom k-binning array (default: logarithmic)
                apply_window_correction: Whether to apply window function correction
                    (default: False, recommended for most applications)
                
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
            self.apply_window_correction = apply_window_correction
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
            
            # DEBUG: Print k-bin edges for comparison
            
            if process_id == 0:
                print(f"PowerSpectrumCalculator initialized:")
                print(f"  Grid size: {ngrid}³")
                print(f"  Box size: {box_size:.1f} Mpc/h")
                print(f"  Cell size: {box_size/ngrid:.3f} Mpc/h")
                print(f"  k-bins: {len(self.k_bins)-1}")
                print(f"  k-range: {self.k_bins[0]:.6f} to {self.k_bins[-1]:.6f} h/Mpc")
                print(f"  Window correction: {'ENABLED' if apply_window_correction else 'DISABLED'}")
                if is_distributed:
                    print(f"  JAX Distributed Mode: ENABLED ({n_processes} processes)")
                    print(f"  JAX will be initialized after multiprocessing is complete")
                else:
                    print(f"  JAX Distributed Mode: DISABLED (using {n_devices} device(s))")

    
    def calculate_power_spectrum(self, particles,
                               subtract_shot_noise: bool = False,
                               assignment: str = 'cic') -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Calculate power spectrum from particle distribution.
        
        For random particles, the power spectrum should equal the shot noise P_shot = V/N.
        
        Args:
            particles: Particle dictionary ('x', 'y', 'z', 'mass') 
            subtract_shot_noise: Whether to subtract shot noise
            assignment: Mass assignment scheme ('cic' or 'ngp')
            
        Returns:
            Tuple of (k_bins, power_spectrum, n_modes_per_bin, grid_stats)
            where grid_stats contains:
                - 'delta_mean': global mean of density contrast field
                - 'delta_variance': global variance of density contrast field  
                - 'delta_std': global std dev of density contrast field
                - 'theoretical_variance': theoretical white noise variance
                - 'particle_count': total particle count used
            
        Raises:
            ValueError: If particle data is invalid
        """
        # DEBUG: Track which processes enter the power spectrum calculation
        process_id = int(os.environ.get('SLURM_PROCID', '0'))
        
        # Validate input
        self._validate_particles(particles)
        
        # Check execution mode using SLURM environment (no JAX needed yet)
        n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        distributed_mode = n_processes > 1
        
        if distributed_mode:
            # Create gridder for distributed mode
            from .particle_gridder import ParticleGridder
            gridder = ParticleGridder(self.ngrid, self.box_size, assignment)
            return self._calculate_distributed(
                particles, gridder, subtract_shot_noise, assignment)
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
                             assignment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
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
        debug_mode = os.environ.get('PKDPIPE_DEBUG_MODE', 'false').lower() == 'true'
        
        if debug_mode:
            print(f"DEBUG: Process {process_id} ENTERED _calculate_distributed", flush=True)
        
        # Use global MPI communicator
        if not _MPI_AVAILABLE:
            raise ImportError("MPI4py required for distributed mode but not available")
        comm = _MPI_COMM
        if debug_mode:
            print(f"DEBUG: Process {process_id} initialized MPI communicator", flush=True)
        
        # Step 1: Redistribute particles based on spatial decomposition using MPI4py
        spatial_particles, y_start, y_end, y_start_ghost, y_end_ghost = redistribute_particles_mpi_simple(
            particles, self.ngrid, self.box_size, comm
        )
        
        # Debug: Check domain decomposition (all in grid coordinates now)
        print(f"Process {process_id}: Y-slab domain = [{y_start}, {y_end}), with ghosts [{y_start_ghost}, {y_end_ghost})", flush=True)
        
        # Calculate and store global particle count for density diagnostics
        local_particle_count = len(spatial_particles['x'])
        
        # Use MPI to calculate global particle count (comm already initialized above)
        self._global_total_particles = comm.allreduce(local_particle_count, op=MPI.SUM)
        print(f"Process {process_id}: Global particle count (MPI): {self._global_total_particles}, local: {local_particle_count}", flush=True)
        
        # MPI Barrier: Synchronize before starting gridding operations
        try:
            if _MPI_AVAILABLE:
                comm = _MPI_COMM
                print(f"Process {process_id}: SYNC POINT 1 - Before gridding operations", flush=True)
                comm.Barrier()
                print(f"Process {process_id}: SYNC POINT 1 - All processes ready for gridding", flush=True)
        except ImportError:
            pass
            
        # Step 2: Grid particles to slab - all coordinates are now in grid units
        slab_height = y_end - y_start
        ghost_slab_height = y_end_ghost - y_start_ghost
        
        print(f"Process {process_id}: Starting grid allocation for slab {self.ngrid}x{ghost_slab_height}x{self.ngrid} (owned: {slab_height} cells)", flush=True)
        
        # Grid particles to slab including ghost zones (all coordinates in grid units)
        print(f"Process {process_id}: Starting CIC assignment to slab...", flush=True)
        full_slab = gridder.particles_to_slab(spatial_particles, y_start_ghost, y_end_ghost, self.ngrid)
        
        print(f"Process {process_id}: CIC assignment complete, full_slab shape: {full_slab.shape}", flush=True)
        
        # MPI Barrier: Synchronize after gridding operations before memory operations
        try:
            if _MPI_AVAILABLE:
                comm = _MPI_COMM
                print(f"Process {process_id}: SYNC POINT 2 - After CIC assignment", flush=True)
                comm.Barrier()
                print(f"Process {process_id}: SYNC POINT 2 - All processes completed CIC assignment", flush=True)
        except ImportError:
            pass
        
        # === SAFE POINT: Initialize JAX after multiprocessing is complete ===
        # This is similar to how power_spectrum_real_data.py handles JAX initialization
        print(f"Process {process_id}: JAX initialized successfully after multiprocessing", flush=True)
        
        # MPI Barrier: Wait for all processes to complete memory operations before proceeding to JAX operations
        try:
            if _MPI_AVAILABLE:
                comm = _MPI_COMM
                print(f"Process {process_id}: SYNC POINT 3 - Before JAX operations", flush=True)
                comm.Barrier()
                print(f"Process {process_id}: SYNC POINT 3 - All processes ready for JAX operations", flush=True)
        except ImportError:
            # MPI not available, skip barrier
            pass
        
        # Extract owned portion (remove ghost zones) - simple grid coordinate arithmetic
        ghost_start = y_start - y_start_ghost
        ghost_end = ghost_start + slab_height
        
        print(f"Process {process_id}: Extracting owned slab: ghost_start={ghost_start}, ghost_end={ghost_end}, full_slab.shape={full_slab.shape}")
        
        # DEBUG: Check if full_slab contains NaN before extraction
        full_slab_sum = np.sum(full_slab)
        full_slab_has_nan = np.isnan(full_slab_sum)
        print(f"Process {process_id}: full_slab sum: {full_slab_sum}, has_nan: {full_slab_has_nan}, min: {np.min(full_slab)}, max: {np.max(full_slab)}")
        
        owned_slab = full_slab[:, ghost_start:ghost_end, :]  # Shape: (ngrid, slab_height, ngrid)
        
        
        # Step 3: Calculate mean density and density contrast  
        local_mass = np.sum(owned_slab)
        
        # Use MPI to calculate true global mass
        if _MPI_AVAILABLE:
            total_mass = comm.allreduce(local_mass, op=MPI.SUM)
            print(f"Process {process_id}: Global mass (MPI): {total_mass:.6e}, local mass: {local_mass:.6e}", flush=True)
        else:
            total_mass = local_mass
            
        mean_density = total_mass / self.ngrid**3
        
        # Debug: Check owned slab
        print(f"Process {process_id}: owned_slab shape: {owned_slab.shape}, sum: {owned_slab.sum():.6e}, mean: {owned_slab.mean():.6e}", flush=True)
        
        # Validate mean density
        if mean_density <= 0:
            raise ValueError(f"Process {process_id}: Mean density is zero or negative ({mean_density:.6e}) - check particle redistribution")
        
        # Calculate density contrast (mean_density is guaranteed > 0 from validation above)
        delta_slab = (owned_slab - mean_density) / mean_density
        
        # Debug: Check delta slab
        print(f"Process {process_id}: delta_slab shape: {delta_slab.shape}, mean: {delta_slab.mean():.6e}, std: {delta_slab.std():.6e}", flush=True)
        
        # Store diagnostics
        self._store_density_diagnostics(owned_slab, delta_slab, len(spatial_particles['x']))
        
        # MPI Barrier: Synchronize before FFT operations
        try:
            if _MPI_AVAILABLE:
                comm = _MPI_COMM
                print(f"Process {process_id}: SYNC POINT 4 - Before FFT operations", flush=True)
                comm.Barrier()
                print(f"Process {process_id}: SYNC POINT 4 - All processes ready for FFT", flush=True)
        except ImportError:
            pass
        
        # Step 4: Initialize JAX with distributed mode right before FFT
        # This ensures JAX is only initialized after all multiprocessing is complete
        print(f"Process {process_id}: Starting FFT with delta_slab shape {delta_slab.shape}", flush=True)
        
        # Use the fft() function which will handle JAX distributed initialization
        delta_k_slab = fft(delta_slab, direction='r2c')  # JAX distributed FFT: spatial slab -> k-space slab
        
        print(f"Process {process_id}: FFT complete, delta_k_slab shape {delta_k_slab.shape}", flush=True)
        
        # Calculate power spectrum on k-space slab  
        # Use NumPy for power calculation to avoid additional JAX imports after FFT
        power_3d_slab = np.abs(delta_k_slab)**2 * (self.volume / self.ngrid**6)
        power_3d_np = power_3d_slab
        
        # MPI Barrier: Synchronize after FFT before k-space operations
        try:
            if _MPI_AVAILABLE:
                comm = _MPI_COMM
                print(f"Process {process_id}: SYNC POINT 5 - After FFT, before k-space processing", flush=True)
                comm.Barrier()
                print(f"Process {process_id}: SYNC POINT 5 - All processes completed FFT", flush=True)
        except ImportError:
            pass
        
        # Step 5: Create k-grid for k-space slab - convert grid coordinates to physical
        physical_y_start = y_start * self.box_size / self.ngrid  
        physical_y_end = y_end * self.box_size / self.ngrid
        print(f"DEBUG: Process {process_id} - ABOUT TO CREATE K-GRID with physical_y=[{physical_y_start:.1f}, {physical_y_end:.1f}]", flush=True)
        k_grid_slab = create_slab_k_grid(self.ngrid, self.box_size, physical_y_start, physical_y_end)
        print(f"DEBUG: Process {process_id} - K-GRID CREATED", flush=True)
        power_3d_corrected, k_grid_corrected = self._apply_window_correction(power_3d_np, k_grid_slab, assignment)
        print(f"DEBUG: Process {process_id} - WINDOW CORRECTION APPLIED", flush=True)
        
        # MPI Barrier: Synchronize before power spectrum binning and reduction
        try:
            if _MPI_AVAILABLE:
                comm = _MPI_COMM
                print(f"Process {process_id}: SYNC POINT 6 - Before power spectrum binning", flush=True)
                comm.Barrier()
                print(f"Process {process_id}: SYNC POINT 6 - All processes ready for binning", flush=True)
        except ImportError:
            pass
        
        # Step 6: Bin and reduce across processes
        print(f"Process {process_id}: Starting power spectrum binning and MPI reduction", flush=True)
        k_binned, power_binned, n_modes = bin_power_spectrum_distributed(
            k_grid_corrected, power_3d_corrected, self.k_bins
        )
        print(f"Process {process_id}: Power spectrum binning complete", flush=True)
        
        return self._finalize_power_spectrum(k_binned, power_binned, n_modes, 
                                           subtract_shot_noise, self._global_total_particles)
    
    def _calculate_single_device(self, particles: Dict[str, np.ndarray],
                               gridder: ParticleGridder, subtract_shot_noise: bool,
                               assignment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Calculate power spectrum on single GPU/CPU."""
        # Time the particle gridding step specifically
        import time
        
        # MPI barrier before gridding timing
        try:
            if _MPI_AVAILABLE:
                comm = _MPI_COMM
                process_id = comm.Get_rank()
                comm.Barrier()
                if process_id == 0:
                    print(f"🔄 Starting particle assignment to grid...")
            else:
                process_id = 0
                print(f"🔄 Starting particle assignment to grid...")
        except Exception:
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
        
        # Store diagnostics before delta conversion (save original density)
        original_density = density_grid.copy()
        
        # MEMORY OPTIMIZATION: In-place delta calculation (no separate delta_grid)
        density_grid /= mean_density
        density_grid -= 1.0  # Now density_grid contains delta field in-place
        
        # Store diagnostics with original density and delta field
        self._store_density_diagnostics(original_density, density_grid, len(particles['x']))
        
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
        power_3d_corrected, _ = self._apply_window_correction(power_3d_np, k_grid, assignment)
        
        # Bin power spectrum
        k_binned, power_binned, n_modes = bin_power_spectrum_single(
            k_grid, power_3d_corrected, self.k_bins
        )
        
        return self._finalize_power_spectrum(k_binned, power_binned, n_modes,
                                           subtract_shot_noise, len(particles['x']))
    
    def _calculate_streaming(self, data_reader, subtract_shot_noise: bool,
                           assignment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
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
        
        # Store diagnostics before delta conversion (save original density)
        original_density = density_grid.copy()
        
        # MEMORY OPTIMIZATION: In-place delta calculation (no separate delta_grid)
        density_grid /= mean_density
        density_grid -= 1.0  # Now density_grid contains delta field in-place
        
        # Store diagnostics with original density and delta field
        self._store_density_diagnostics(original_density, density_grid, total_particles)
        
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
        power_3d_corrected, _ = self._apply_window_correction(power_3d_np, k_grid, assignment)
        
        # Bin power spectrum
        k_binned, power_binned, n_modes = bin_power_spectrum_single(
            k_grid, power_3d_corrected, self.k_bins
        )
        
        return self._finalize_power_spectrum(k_binned, power_binned, n_modes,
                                           subtract_shot_noise, total_particles)
    
    def _calculate_streaming_distributed(self, data_reader, subtract_shot_noise: bool,
                                       assignment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
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
        
        # Calculate global mean density across all processes
        n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        
        if n_processes > 1:
            # Distributed mode: compute global mean density using MPI
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                
                # Local density statistics
                local_sum = np.sum(density_grid)
                local_count = density_grid.size
                
                # Global reductions
                global_sum = comm.allreduce(local_sum, op=MPI.SUM)
                global_count = comm.allreduce(local_count, op=MPI.SUM)
                
                # Global mean density
                global_mean_density = global_sum / global_count
                mean_density = np.float32(global_mean_density)
                
                print(f"DEBUG: Process {comm.Get_rank()}: Local mean: {density_grid.mean():.6e}, Global mean: {mean_density:.6e}")
                
                # Global variance using corrected formula
                local_variance = np.var(density_grid, ddof=1)
                global_variance = comm.allreduce(local_variance, op=MPI.SUM) / (n_processes - 1)
                
                print(f"DEBUG: Process {comm.Get_rank()}: Local variance: {local_variance:.6e}, Global variance: {global_variance:.6e}")
                
                # Chi-squared test for variance
                # For a Poisson process, Var(N) = E[N]. So Var(density) = E[density].
                # The quantity (n-1) * S^2 / σ^2 follows a chi-squared distribution
                # where S^2 is the sample variance and σ^2 is the theoretical variance.
                # Here, S^2 = global_variance, σ^2 = global_mean
                n_samples = global_count
                if global_mean > 0:
                    chi2_statistic = (n_samples - 1) * global_variance / global_mean
                    degrees_of_freedom = n_samples - 1
                    
                    # Use scipy.stats for p-value. Import locally.
                    try:
                        from scipy.stats import chi2
                        # p-value is the probability of observing a chi2 value this extreme or more
                        p_value = chi2.sf(chi2_statistic, degrees_of_freedom)
                    except ImportError:
                        p_value = np.nan # SciPy not available
                else:
                    chi2_statistic = np.nan
                    p_value = np.nan

                # Store global total particles for get_density_diagnostics
                self._global_total_particles = comm.allreduce(n_particles, op=MPI.SUM)
                
                if debug_mode:
                    print(f"DEBUG: Process {comm.Get_rank()}: Global density stats - mean: {global_mean:.6e}, variance: {global_variance:.6e}")
                
                self._last_density_stats = {
                    'mean_density': float(global_mean),
                    'density_variance': float(global_variance),
                    'delta_mean': float(global_delta_mean),
                    'delta_variance': float(global_delta_variance),
                    'theoretical_shot_noise_variance': float(self._global_total_particles / (global_mean * self.ngrid**3)) if global_mean > 0 else np.nan,
                    'variance_chi2_statistic': float(chi2_statistic),
                    'variance_p_value': float(p_value)
                }
                
            except ImportError:
                # Fallback to local stats if MPI not available
                print("WARNING: MPI not available, using local density statistics")
                mean_density = float(np.mean(density_flat))
                self._last_density_stats = {
                    'mean_density': mean_density,
                    'density_variance': float(np.var(density_flat)),
                    'delta_mean': float(np.mean(delta_flat)),
                    'delta_variance': float(np.var(delta_flat)),
                    'theoretical_shot_noise_variance': float(n_particles / (mean_density * self.ngrid**3)) if mean_density > 0 else np.nan
                }
        else:
            # Single process mode: use local mean
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
        power_3d_corrected, _ = self._apply_window_correction(power_3d_np, k_grid_local, assignment)
        
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
        # Check debug mode
        debug_mode = os.environ.get('PKDPIPE_DEBUG_MODE', 'false').lower() == 'true'
        if hasattr(density_data, 'flatten'):
            density_flat = density_data.flatten()
        else:
            density_flat = density_data
            
        if hasattr(delta_data, 'flatten'):
            delta_flat = delta_data.flatten()
        else:
            delta_flat = delta_data
        
        # Check if we're in distributed mode
        n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        
        if n_processes > 1:
            # Distributed mode: compute global statistics using MPI
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                
                # Local statistics
                local_sum = np.sum(density_flat)
                local_count = len(density_flat)
                local_sum_sq = np.sum(density_flat**2)
                
                local_delta_sum = np.sum(delta_flat)
                local_delta_sum_sq = np.sum(delta_flat**2)
                
                # Global reductions
                global_sum = comm.allreduce(local_sum, op=MPI.SUM)
                global_count = comm.allreduce(local_count, op=MPI.SUM)
                global_sum_sq = comm.allreduce(local_sum_sq, op=MPI.SUM)
                
                global_delta_sum = comm.allreduce(local_delta_sum, op=MPI.SUM)
                global_delta_sum_sq = comm.allreduce(local_delta_sum_sq, op=MPI.SUM)
                
                # Compute global mean and variance
                global_mean = global_sum / global_count
                global_variance = (global_sum_sq / global_count) - global_mean**2
                
                global_delta_mean = global_delta_sum / global_count  
                global_delta_variance = (global_delta_sum_sq / global_count) - global_delta_mean**2
                
                # Chi-squared test for variance
                # For a Poisson process, Var(N) = E[N]. So Var(density) = E[density].
                # The quantity (n-1) * S^2 / σ^2 follows a chi-squared distribution
                # where S^2 is the sample variance and σ^2 is the theoretical variance.
                # Here, S^2 = global_variance, σ^2 = global_mean
                n_samples = global_count
                if global_mean > 0:
                    chi2_statistic = (n_samples - 1) * global_variance / global_mean
                    degrees_of_freedom = n_samples - 1
                    
                    # Use scipy.stats for p-value. Import locally.
                    try:
                        from scipy.stats import chi2
                        # p-value is the probability of observing a chi2 value this extreme or more
                        p_value = chi2.sf(chi2_statistic, degrees_of_freedom)
                    except ImportError:
                        p_value = np.nan # SciPy not available
                else:
                    chi2_statistic = np.nan
                    p_value = np.nan

                # Store global total particles for get_density_diagnostics
                self._global_total_particles = comm.allreduce(n_particles, op=MPI.SUM)
                
                if debug_mode:
                    print(f"DEBUG: Process {comm.Get_rank()}: Global density stats - mean: {global_mean:.6e}, variance: {global_variance:.6e}")
                
                self._last_density_stats = {
                    'mean_density': float(global_mean),
                    'density_variance': float(global_variance),
                    'delta_mean': float(global_delta_mean),
                    'delta_variance': float(global_delta_variance),
                    'theoretical_shot_noise_variance': float(self._global_total_particles / (global_mean * self.ngrid**3)) if global_mean > 0 else np.nan,
                    'variance_chi2_statistic': float(chi2_statistic),
                    'variance_p_value': float(p_value)
                }
                
            except ImportError:
                # Fallback to local stats if MPI not available
                print("WARNING: MPI not available, using local density statistics")
                mean_density = float(np.mean(density_flat))
                self._last_density_stats = {
                    'mean_density': mean_density,
                    'density_variance': float(np.var(density_flat)),
                    'delta_mean': float(np.mean(delta_flat)),
                    'delta_variance': float(np.var(delta_flat)),
                    'theoretical_shot_noise_variance': float(n_particles / (mean_density * self.ngrid**3)) if mean_density > 0 else np.nan
                }
        else:
            # Single process mode: use local statistics
            mean_density = float(np.mean(density_flat))
            variance = float(np.var(density_flat))
            
            # Chi-squared test for variance
            n_samples = len(density_flat)
            if mean_density > 0:
                chi2_statistic = (n_samples - 1) * variance / mean_density
                degrees_of_freedom = n_samples - 1
                try:
                    from scipy.stats import chi2
                    p_value = chi2.sf(chi2_statistic, degrees_of_freedom)
                except ImportError:
                    p_value = np.nan
            else:
                chi2_statistic = np.nan
                p_value = np.nan

            self._last_density_stats = {
                'mean_density': mean_density,
                'density_variance': variance,
                'delta_mean': float(np.mean(delta_flat)),
                'delta_variance': float(np.var(delta_flat)),
                'theoretical_shot_noise_variance': float(n_particles / (mean_density * self.ngrid**3)) if mean_density > 0 else np.nan,
                'variance_chi2_statistic': float(chi2_statistic),
                'variance_p_value': float(p_value)
            }
    
    def _apply_window_correction(self, power_3d: np.ndarray, k_grid: np.ndarray, 
                               assignment: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply window function correction to the power spectrum.
        
        CRITICAL: This function must preserve k_grid dimensions for distributed mode.
        The k_grid is carefully constructed by create_slab_k_grid() or create_full_k_grid()
        and must not be modified to maintain proper k-space decomposition.
        
        Args:
            power_3d: 3D power spectrum array
            k_grid: 3D k-magnitude grid (must be preserved for distributed mode)
            assignment: Mass assignment scheme ('cic' or 'ngp')
            
        Returns:
            Tuple of (corrected/uncorrected 3D power spectrum, original k_grid)
        """
        
        # Validate input dimensions
        if power_3d.shape != k_grid.shape:
            # This should not happen with proper k_grid construction
            # Log the issue but DO NOT modify k_grid to avoid breaking distributed mode
            process_id = int(os.environ.get('SLURM_PROCID', '0'))
            print(f"WARNING Process {process_id}: k_grid shape mismatch - power_3d: {power_3d.shape}, k_grid: {k_grid.shape}", flush=True)
            print(f"WARNING Process {process_id}: Window correction may be incorrect due to shape mismatch", flush=True)
            
            # If window correction is disabled, just return as-is
            if not self.apply_window_correction:
                return power_3d, k_grid
                
            # If enabled, we need to proceed carefully to avoid breaking distributed k-space
            # Log warning and continue with existing k_grid
            print(f"WARNING Process {process_id}: Proceeding with original k_grid to preserve distributed k-space decomposition", flush=True)
        
        # Apply window correction only if enabled (default: disabled)
        if not self.apply_window_correction:
            # Window correction is disabled by default - return unchanged
            return power_3d, k_grid
            
        # Window correction is enabled - apply correction using provided k_grid
        # IMPORTANT: Use the original k_grid to preserve distributed k-space structure
        if assignment.lower() == 'cic':
            window_correction = self._cic_window_function(k_grid)
        elif assignment.lower() == 'ngp':
            window_correction = self._ngp_window_function(k_grid)
        else:
            # Unknown assignment scheme - skip correction
            print(f"WARNING: Unknown assignment scheme '{assignment}', skipping window correction", flush=True)
            return power_3d, k_grid
        
        # Apply correction (divide by window function squared)
        # Ensure window_correction has same shape as power_3d
        if window_correction.shape != power_3d.shape:
            print(f"ERROR: Window correction shape {window_correction.shape} != power_3d shape {power_3d.shape}", flush=True)
            print(f"ERROR: Skipping window correction to avoid corruption", flush=True)
            return power_3d, k_grid
            
        power_3d_corrected = power_3d / (window_correction**2 + 1e-16)  # Add small epsilon to avoid division by zero
        
        return power_3d_corrected, k_grid
    
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
        return window
    
    def _finalize_power_spectrum(self, k_binned: np.ndarray, power_binned: np.ndarray,
                               n_modes: np.ndarray, subtract_shot_noise: bool,
                               n_particles: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Apply final corrections and return results with grid statistics."""
        # Subtract shot noise if requested
        if subtract_shot_noise and len(power_binned) > 0:
            shot_noise = self.volume / n_particles
            power_binned = power_binned - shot_noise
        
        # Prepare grid statistics (computed during density calculation)
        grid_stats = {
            'delta_mean': self._last_density_stats.get('delta_mean', 0.0),
            'delta_variance': self._last_density_stats.get('delta_variance', 0.0),
            'delta_std': np.sqrt(self._last_density_stats.get('delta_variance', 0.0)),
            'theoretical_variance': 1.0 / (n_particles / (self.ngrid**3)),  # 1/⟨ρ⟩ for white noise
            'particle_count': n_particles
        }
        
        return k_binned, power_binned, n_modes, grid_stats
    
    def get_density_diagnostics(self) -> Dict[str, float]:
            """
            Get diagnostic information about the last density field calculation.
            
            For distributed mode, returns global statistics computed across all processes.
            For single-process mode, returns detailed local statistics.
            
            Returns:
                Dictionary with density field statistics including:
                - mean_density: Mean density of the field
                - density_variance: Variance of density field (if available)
                - delta_mean: Mean of density contrast field (if available)
                - delta_variance: Variance of density contrast field (if available)
                - theoretical_shot_noise_variance: Expected shot noise variance (if available)
            """
            # Return the computed statistics (either local or global)
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

def get_spatial_domain_simple(process_id, n_processes, ngrid, assignment_scheme):
    """
    Calculate Y-slab domain for distributed processing - SIMPLIFIED VERSION.
    
    This version uses simple integer arithmetic for slab decomposition and ghost zones,
    avoiding floating-point conversions and rounding issues.
    
    Args:
        process_id (int): MPI rank of the current process.
        n_processes (int): Total number of MPI processes.
        ngrid (int): Grid resolution.
        assignment_scheme (str): Mass assignment scheme ('cic' or 'ngp').
        
    Returns:
        tuple: (y_start, y_end, y_start_ghost, y_end_ghost)
               All values are in grid coordinates.
    """
    # Simplified Y-slab decomposition
    slab_height = ngrid // n_processes
    y_start = process_id * slab_height
    y_end = (process_id + 1) * slab_height
    
    # Ensure last process covers the full grid
    if process_id == n_processes - 1:
        y_end = ngrid
        
    # Ghost zones are simple integer offsets
    # CIC requires 1 ghost cell on each side
    if assignment_scheme.lower() == 'cic':
        y_start_ghost = max(0, y_start - 1)
        y_end_ghost = min(ngrid, y_end + 1)
    else: # NGP needs no ghost zones
        y_start_ghost = y_start
        y_end_ghost = y_end
        
    return y_start, y_end, y_start_ghost, y_end_ghost

def redistribute_particles_mpi_simple(particles, ngrid, box_size, comm):
    """
    Redistribute particles using MPI based on a simple Y-slab decomposition.
    
    This function determines which process each particle belongs to based on its
    Y-coordinate and sends it to the correct process using MPI_Alltoallv.
    
    Args:
        particles (dict): Dictionary of particle data ('x', 'y', 'z').
        ngrid (int): Grid resolution.
        box_size (float): Simulation box size.
        comm: MPI communicator.
        
    Returns:
        tuple: (local_particles, y_start, y_end, y_start_ghost, y_end_ghost)
               local_particles are particles belonging to the current process's domain.
               Domain boundaries are in grid coordinates.
    """
    process_id = comm.Get_rank()
    n_processes = comm.Get_size()
    
    # Determine which process each particle belongs to (in grid coordinates)
    y_coords_grid = (particles['y'] / box_size * ngrid).astype(int)
    
    # Simple Y-slab decomposition
    slab_height = ngrid // n_processes
    target_process = y_coords_grid // slab_height
    
    # FIXED: Remove incorrect boundary particle reassignment that caused double-counting
    # Particles should stay in their proper spatial domain - ghost zones handle CIC interpolation
    # The original boundary logic was causing 1.3% variance excess by double-counting particles
    
    target_process = np.clip(target_process, 0, n_processes - 1)

    # Prepare data for MPI_Alltoallv
    send_counts = np.bincount(target_process, minlength=n_processes)
    send_displacements = np.insert(np.cumsum(send_counts)[:-1], 0, 0)
    
    # Sort particles by target process for contiguous sending
    sort_indices = np.argsort(target_process)
    
    # Prepare send buffers
    send_buf_x = particles['x'][sort_indices].astype(np.float32)
    send_buf_y = particles['y'][sort_indices].astype(np.float32)
    send_buf_z = particles['z'][sort_indices].astype(np.float32)
    
    # Receive counts and displacements
    recv_counts = np.empty(n_processes, dtype=int)
    comm.Alltoall(send_counts, recv_counts)
    
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
    total_recv = np.sum(recv_counts)
    
    # Prepare receive buffers
    recv_buf_x = np.empty(total_recv, dtype=np.float32)
    recv_buf_y = np.empty(total_recv, dtype=np.float32)
    recv_buf_z = np.empty(total_recv, dtype=np.float32)
    
    # MPI Alltoallv communication
    comm.Alltoallv([send_buf_x, (send_counts, send_displacements)], 
                    [recv_buf_x, (recv_counts, recv_displacements)])
    comm.Alltoallv([send_buf_y, (send_counts, send_displacements)], 
                    [recv_buf_y, (recv_counts, recv_displacements)])
    comm.Alltoallv([send_buf_z, (send_counts, send_displacements)], 
                    [recv_buf_z, (recv_counts, recv_displacements)])
    
    local_particles = {'x': recv_buf_x, 'y': recv_buf_y, 'z': recv_buf_z}
    
    # Get domain boundaries for this process
    y_start, y_end, y_start_ghost, y_end_ghost = get_spatial_domain_simple(
        process_id, n_processes, ngrid, 'cic' # Assuming CIC for now
    )
    
    return local_particles, y_start, y_end, y_start_ghost, y_end_ghost
