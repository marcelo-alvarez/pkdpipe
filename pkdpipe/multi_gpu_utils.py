"""
Multi-GPU and distributed computing utilities for power spectrum calculations.

This module provides utilities for distributed FFT computations and power spectrum
calculations across multiple GPUs using JAX's distributed computing capabilities.

Example usage:
    # Check if running in distributed mode
    if is_distributed_mode():
        # Create distributed k-grid for local process
        k_grid = create_local_k_grid(ngrid, box_size)
        
        # Bin and reduce power spectrum across processes
        k_bins, power, n_modes = bin_power_spectrum_distributed(k_grid, power_grid, k_bin_edges)
"""

import numpy as np
import os
from typing import Tuple, Optional

# JAX imports are deferred to avoid CUDA initialization conflicts with multiprocessing
# JAX will be imported when first needed by calling _ensure_jax_initialized()
try:
    # Test if JAX is available without importing it
    import importlib.util
    spec = importlib.util.find_spec("jax")
    JAX_AVAILABLE = spec is not None
except ImportError:
    JAX_AVAILABLE = False

# Global variables for JAX modules (initialized when first needed)
jax = None
jnp = None


def _ensure_jax_initialized():
    """
    Safely import and initialize JAX when first needed.
    
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
            
            # Store in global variables
            jax = jax_module
            jnp = jnp_module
            
        except ImportError as e:
            print(f"JAX initialization failed in multi_gpu_utils: {e}", flush=True)
            JAX_AVAILABLE = False
            return None, None
    
    return jax, jnp


def is_distributed_mode() -> bool:
    """
    Check if running in JAX distributed mode.
    
    Returns:
        True if JAX distributed mode is active, False otherwise
    """
    if not JAX_AVAILABLE:
        return False
    
    jax_module, _ = _ensure_jax_initialized()
    if jax_module is None:
        return False
        
    try:
        # Simply check if JAX is already in distributed mode
        process_count = jax_module.process_count()
        return process_count > 1
    except (ImportError, AttributeError, RuntimeError):
        return False


def get_process_info() -> Tuple[int, int]:
    """
    Get current process information in distributed mode.
    
    Returns:
        Tuple of (process_id, total_processes)
        Returns (0, 1) if not in distributed mode
    """
    if not is_distributed_mode():
        return 0, 1
    
    jax_module, _ = _ensure_jax_initialized()
    if jax_module is None:
        return 0, 1
        
    try:
        return jax_module.process_index(), jax_module.process_count()
    except (ImportError, AttributeError):
        return 0, 1


def create_local_k_grid(ngrid: int, box_size: float) -> np.ndarray:
    """
    Create local k-grid for this process's FFT domain slice.
    
    In distributed mode, each process handles a slice of the Y-dimension
    of the FFT grid. This function creates the appropriate k-grid for
    the local domain.
    
    Args:
        ngrid: Number of grid cells per dimension
        box_size: Size of simulation box in Mpc/h
        
    Returns:
        3D array of k-magnitudes for local domain
    """
    if not is_distributed_mode():
        # Fallback to full grid if not in distributed mode
        return create_full_k_grid(ngrid, box_size)
    
    process_id, total_processes = get_process_info()
    
    # Grid spacing
    dx = box_size / ngrid
    
    # Create 1D k arrays - kx and kz are the same on all processes
    kx = 2 * np.pi * np.fft.fftfreq(ngrid, dx)
    kz = 2 * np.pi * np.fft.rfftfreq(ngrid, dx)  # Real FFT
    
    # ky is domain-decomposed along Y-axis
    # Each process gets a slice of the full ky range
    y_slice_size = ngrid // total_processes
    y_start = process_id * y_slice_size
    y_end = y_start + y_slice_size
    
    # Handle case where ngrid is not evenly divisible by number of processes
    if process_id == total_processes - 1:
        y_end = ngrid
    
    ky_full = 2 * np.pi * np.fft.fftfreq(ngrid, dx)
    ky_local = ky_full[y_start:y_end]
    
    # Create 3D grid for local domain
    kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky_local, kz, indexing='ij')
    
    # Calculate |k|
    k_mag = np.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)
    
    return k_mag


def create_full_k_grid(ngrid: int, box_size: float) -> np.ndarray:
    """
    Create full 3D k-grid for FFT output.
    
    Args:
        ngrid: Number of grid cells per dimension
        box_size: Size of simulation box in Mpc/h
        
    Returns:
        3D array of k-magnitudes
    """
    # Grid spacing
    dx = box_size / ngrid
    
    # Create 1D k arrays with correct spacing
    kx = 2 * np.pi * np.fft.fftfreq(ngrid, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ngrid, dx)
    kz = 2 * np.pi * np.fft.rfftfreq(ngrid, dx)  # Real FFT
    
    # Create 3D grid
    kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Calculate |k|
    k_mag = np.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)
    
    return k_mag


def create_slab_k_grid(ngrid: int, box_size: float, y_min: int, y_max: int) -> np.ndarray:
    """
    Create k-space grid for a spatial slab in distributed mode.
    
    Args:
        ngrid: Full grid resolution (e.g., 512)
        box_size: Simulation box size in Mpc/h
        y_min: Start index of slab in y-direction (unused - kept for compatibility)
        y_max: End index of slab in y-direction (unused - kept for compatibility)
        
    Returns:
        k-magnitude grid for the slab portion
    """
    # Simple approach: slab height is just ngrid divided by number of processes
    # This assumes ngrid is divisible by n_processes (e.g., 512/4=128)
    slurm_ntasks = int(os.environ.get('SLURM_NTASKS', 1))
    slab_height_cells = ngrid // slurm_ntasks
    
    print(f"DEBUG create_slab_k_grid: Simple approach - ngrid={ngrid}, n_processes={slurm_ntasks}, slab_height_cells={slab_height_cells}", flush=True)
    
    # Create frequency grids
    fundamental_mode = 2 * np.pi / box_size
    kx = np.fft.fftfreq(ngrid, d=1.0) * ngrid * fundamental_mode
    ky = np.fft.fftfreq(slab_height_cells, d=1.0) * ngrid * fundamental_mode  # Note: still use ngrid for proper scaling
    kz = np.fft.rfftfreq(ngrid, d=1.0) * ngrid * fundamental_mode
    
    # Create 3D grid for the slab
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_grid = np.sqrt(KX**2 + KY**2 + KZ**2)
    
    return k_grid


def bin_power_spectrum_distributed(k_grid: np.ndarray, power_grid: np.ndarray, 
                                 k_bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin local power spectrum and reduce across processes.
    
    This function performs local binning of the power spectrum on each process,
    then uses JAX's distributed reduction operations to combine results across
    all processes.
    
    Args:
        k_grid: Local k-magnitude grid
        power_grid: Local power spectrum grid
        k_bins: k-bin edges for binning
        
    Returns:
        Tuple of (k_binned, power_binned, n_modes_per_bin)
    """
    # First, bin the local power spectrum
    k_flat = k_grid.flatten()
    power_flat = power_grid.flatten()
    
    # Remove zero mode
    nonzero = k_flat > 0
    k_flat = k_flat[nonzero]
    power_flat = power_flat[nonzero]
    
    # Initialize local bins
    n_bins = len(k_bins) - 1
    local_power_sums = np.zeros(n_bins)
    local_k_sums = np.zeros(n_bins)
    local_mode_counts = np.zeros(n_bins, dtype=int)
    
    if len(k_flat) > 0:
        # Find bin indices
        bin_indices = np.digitize(k_flat, k_bins)
        
        # Accumulate local contributions to each bin
        for i in range(1, n_bins + 1):  # digitize returns 1-based indices
            mask = bin_indices == i
            if np.any(mask):
                local_power_sums[i-1] = np.sum(power_flat[mask])
                local_k_sums[i-1] = np.sum(k_flat[mask])
                local_mode_counts[i-1] = np.sum(mask)
    
    # Reduce across all processes using JAX distributed operations
    if is_distributed_mode() and JAX_AVAILABLE:
        jax_module, jnp_module = _ensure_jax_initialized()
        if jax_module is not None and jnp_module is not None:
            try:
                # Convert to JAX arrays for reduction
                local_power_sums_jax = jnp_module.array(local_power_sums)
                local_k_sums_jax = jnp_module.array(local_k_sums)
                local_mode_counts_jax = jnp_module.array(local_mode_counts)
                
                # All-reduce to sum across processes
                global_power_sums = jax_module.lax.psum(local_power_sums_jax, axis_name=None)
                global_k_sums = jax_module.lax.psum(local_k_sums_jax, axis_name=None)
                global_mode_counts = jax_module.lax.psum(local_mode_counts_jax, axis_name=None)
                
                # Convert back to numpy
                global_power_sums = np.array(global_power_sums)
                global_k_sums = np.array(global_k_sums)
                global_mode_counts = np.array(global_mode_counts, dtype=int)
                
            except Exception:
                # Fallback if distributed operations not available
                global_power_sums = local_power_sums
                global_k_sums = local_k_sums
                global_mode_counts = local_mode_counts
    else:
        # Not in distributed mode - use local values
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


def bin_power_spectrum_single(k_grid: np.ndarray, power_grid: np.ndarray, 
                            k_bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin 3D power spectrum into 1D radial bins (single process version).
    
    Args:
        k_grid: 3D k-magnitude grid
        power_grid: 3D power spectrum grid
        k_bins: k-bin edges for binning
        
    Returns:
        Tuple of (k_binned, power_binned, n_modes_per_bin)
    """
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
    bin_indices = np.digitize(k_flat, k_bins)
    
    # Calculate binned quantities
    n_bins = len(k_bins) - 1
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


def default_k_bins(ngrid: int, box_size: float) -> np.ndarray:
    """
    Create default logarithmic k-binning based on theoretical FFT frequencies.
    
    Args:
        ngrid: Number of grid cells per dimension
        box_size: Size of simulation box in Mpc/h
        
    Returns:
        Array of k-bin edges
    """
    # Theoretical k-range for FFT grid
    k_fund = 2 * np.pi / box_size  # Fundamental mode
    k_nyquist = np.pi * ngrid / box_size  # Nyquist frequency
    
    # Use the theoretical range
    k_min = k_fund
    k_max = k_nyquist
    
    # Number of bins - use fewer bins for small grids
    n_bins = min(20, ngrid // 2)  
    
    # Ensure we have at least a few bins
    if n_bins < 5:
        n_bins = 5
    
    # Create logarithmic bins
    log_k_min = np.log10(k_min)
    log_k_max = np.log10(k_max)
    
    return np.logspace(log_k_min, log_k_max, n_bins + 1)  # n_bins+1 bin edges