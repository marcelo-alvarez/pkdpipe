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
    Check if running in distributed mode (MPI/SLURM multi-process environment).
    
    Returns:
        True if running in a multi-process distributed environment, False otherwise
    """
    # Check SLURM environment first (most reliable)
    ntasks = os.environ.get('SLURM_NTASKS', '1')
    try:
        if int(ntasks) > 1:
            return True
    except ValueError:
        pass
    
    # Check MPI environment as fallback
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_size() > 1
    except ImportError:
        pass
    
    # Legacy fallback: check JAX distributed mode (may not work if JAX not initialized)
    if JAX_AVAILABLE:
        jax_module, _ = _ensure_jax_initialized()
        if jax_module is not None:
            try:
                process_count = jax_module.process_count()
                return process_count > 1
            except (ImportError, AttributeError, RuntimeError):
                pass
    
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
    # FIXED: Use proper k-space mapping for distributed FFT
    # Each process has a slab in real space, but the k-space must represent
    # the corresponding frequencies from the full k-grid
    slurm_ntasks = int(os.environ.get('SLURM_NTASKS', 1))
    slab_height_cells = ngrid // slurm_ntasks
    
    print(f"DEBUG create_slab_k_grid: Fixed approach - ngrid={ngrid}, n_processes={slurm_ntasks}, slab_height_cells={slab_height_cells}", flush=True)
    
    # Grid spacing (same as single-process version)
    dx = box_size / ngrid
    
    # Create 1D k arrays - ALL dimensions use full grid frequency mapping
    # This ensures the k-space coverage matches the single-process version
    kx = 2 * np.pi * np.fft.fftfreq(ngrid, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ngrid, dx)  # FIXED: Use full ngrid, not slab_height_cells
    kz = 2 * np.pi * np.fft.rfftfreq(ngrid, dx)  # Real FFT
    
    # Extract the slab portion in k-space that corresponds to this process's y-slab
    # For a 2-process decomposition with ngrid=128:
    # Process 0: ky indices [0:64] (corresponds to y spatial slab [0:64])
    # Process 1: ky indices [64:128] (corresponds to y spatial slab [64:128])
    process_id = int(os.environ.get('SLURM_PROCID', '0'))
    ky_start = process_id * slab_height_cells
    ky_end = (process_id + 1) * slab_height_cells
    ky_slab = ky[ky_start:ky_end]
    
    # Create 3D grid for the k-space slab
    KX, KY_SLAB, KZ = np.meshgrid(kx, ky_slab, kz, indexing='ij')
    k_grid = np.sqrt(KX**2 + KY_SLAB**2 + KZ**2)
    
    # DEBUG: Print k-frequency ranges for each process
    print(f"DEBUG create_slab_k_grid Process {process_id}: ky_slab range: [{ky_slab.min():.6f}, {ky_slab.max():.6f}] h/Mpc", flush=True)
    print(f"DEBUG create_slab_k_grid Process {process_id}: ky_slab shape: {ky_slab.shape}, indices: [{ky_start}:{ky_end}]", flush=True)
    print(f"DEBUG create_slab_k_grid Process {process_id}: k_grid shape: {k_grid.shape}, range: [{k_grid.min():.6f}, {k_grid.max():.6f}]", flush=True)
    
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
    # DEBUG: Check if all processes reach this function
    process_id = int(os.environ.get('SLURM_PROCID', '0'))
    print(f"DEBUG: Process {process_id} ENTERED bin_power_spectrum_distributed", flush=True)
    print(f"DEBUG: Process {process_id} k_grid shape: {k_grid.shape}, power_grid shape: {power_grid.shape}", flush=True)
    print(f"DEBUG: Process {process_id} k_grid range: [{k_grid.min():.6f}, {k_grid.max():.6f}]", flush=True)
    print(f"DEBUG: Process {process_id} power_grid range: [{power_grid.min():.6e}, {power_grid.max():.6e}]", flush=True)
    
    # First, bin the local power spectrum
    k_flat = k_grid.flatten()
    power_flat = power_grid.flatten()
    
    # Remove zero mode
    nonzero = k_flat > 0
    process_id = int(os.environ.get('SLURM_PROCID', '0'))
    print(f"Process {process_id}: Before zero removal: {len(k_flat)} modes, after: {np.sum(nonzero)} modes", flush=True)
    
    # DEBUG: Print power values for specific k-modes to compare single vs distributed
    if len(k_flat) > 0:
        # Find some specific k-values to track
        k_targets = [0.031416, 0.044429, 0.054414]  # First few k-bins
        for k_target in k_targets:
            # Find modes close to this k-value (within 5%)
            mask = np.abs(k_flat - k_target) < 0.05 * k_target
            if np.any(mask):
                matching_k = k_flat[mask]
                matching_power = power_flat[mask]
                print(f"Process {process_id}: k≈{k_target:.6f}: found {len(matching_k)} modes", flush=True)
                for i in range(min(3, len(matching_k))):  # Show first 3 matches
                    print(f"  k={matching_k[i]:.6f}, P={matching_power[i]:.6e}", flush=True)
        
        # Also print some specific k-modes for detailed debugging
        if len(k_flat) >= 10:
            print(f"Process {process_id}: DETAILED K-MODE DEBUG (first 10 modes):", flush=True)
            for i in range(min(10, len(k_flat))):
                # Calculate kx, ky, kz indices from flattened index
                print(f"  Mode {i}: k={k_flat[i]:.6f}, P={power_flat[i]:.6e}", flush=True)
    
    k_flat = k_flat[nonzero]
    power_flat = power_flat[nonzero]
    
    # ALL processes compute ALL bins (some will be zero) - makes MPI reduction simple
    n_bins = len(k_bins) - 1
    local_k_sums = np.zeros(n_bins)
    local_power_sums = np.zeros(n_bins)
    local_mode_counts = np.zeros(n_bins, dtype=int)
    
    if len(k_flat) > 0:
        # Find bin indices (identical to single-process)
        bin_indices = np.digitize(k_flat, k_bins)
        
        # Count how many modes fall within vs outside bin range
        modes_in_bins = np.sum((bin_indices >= 1) & (bin_indices <= n_bins))
        modes_out_of_range = len(k_flat) - modes_in_bins
        print(f"Process {process_id}: {len(k_flat)} k-modes, {modes_in_bins} in bins, {modes_out_of_range} outside k_max", flush=True)
        
        # Fill ALL bins (IDENTICAL to single-process logic, but store sums for MPI reduction)
        for i in range(1, n_bins + 1):  # digitize returns 1-based indices
            mask = bin_indices == i
            if np.any(mask):
                # Store sums (mean * count) so MPI reduction gives correct final means
                local_k_sums[i-1] = np.mean(k_flat[mask]) * np.sum(mask)
                local_power_sums[i-1] = np.mean(power_flat[mask]) * np.sum(mask)
                local_mode_counts[i-1] = np.sum(mask)
    
    # Simple MPI reduction: sum contributions from all processes
    print(f"Process {process_id}: is_distributed_mode() = {is_distributed_mode()}", flush=True)
    if is_distributed_mode():
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            
            # Simple MPI reduction - sum across all processes
            global_power_sums = np.zeros_like(local_power_sums)
            global_k_sums = np.zeros_like(local_k_sums)
            global_mode_counts = np.zeros_like(local_mode_counts)
            
            print(f"Process {comm.Get_rank()}: BEFORE MPI - local_mode_counts.sum() = {local_mode_counts.sum()}", flush=True)
            comm.Allreduce(local_power_sums, global_power_sums, op=MPI.SUM)
            comm.Allreduce(local_k_sums, global_k_sums, op=MPI.SUM)
            comm.Allreduce(local_mode_counts, global_mode_counts, op=MPI.SUM)
            print(f"Process {comm.Get_rank()}: AFTER MPI - global_mode_counts.sum() = {global_mode_counts.sum()}", flush=True)
            
            print(f"Process {comm.Get_rank()}: MPI reduction complete - local modes: {local_mode_counts.sum()}, global modes: {global_mode_counts.sum()}", flush=True)
            
        except (ImportError, Exception) as e:
            print(f"MPI reduction failed ({e}), using local values", flush=True)
            global_power_sums = local_power_sums
            global_k_sums = local_k_sums
            global_mode_counts = local_mode_counts
    else:
        # Not in distributed mode - use local values
        global_power_sums = local_power_sums
        global_k_sums = local_k_sums
        global_mode_counts = local_mode_counts
    
    # Calculate final binned values (IDENTICAL to single-process behavior)
    k_binned = []
    power_binned = []
    n_modes = []
    
    for i in range(n_bins):
        if global_mode_counts[i] > 0:  # Only include bins that have data
            k_binned.append(global_k_sums[i] / global_mode_counts[i])      # Mean k (sum/count)
            power_binned.append(global_power_sums[i] / global_mode_counts[i])  # Mean power (sum/count)
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
    
    print(f"SINGLE PROCESS: Before zero removal: {len(k_flat)} modes, after: {np.sum(nonzero)} modes", flush=True)
    
    # DEBUG: Print power values for specific k-modes to compare with distributed
    if len(k_flat) > 0:
        # Find some specific k-values to track
        k_targets = [0.031416, 0.044429, 0.054414]  # First few k-bins
        for k_target in k_targets:
            # Find modes close to this k-value (within 5%)
            mask = np.abs(k_flat - k_target) < 0.05 * k_target
            if np.any(mask):
                matching_k = k_flat[mask]
                matching_power = power_flat[mask]
                print(f"SINGLE PROCESS: k≈{k_target:.6f}: found {len(matching_k)} modes", flush=True)
                for i in range(min(3, len(matching_k))):  # Show first 3 matches
                    print(f"  k={matching_k[i]:.6f}, P={matching_power[i]:.6e}", flush=True)
        
        # Also print some specific k-modes for detailed debugging
        if len(k_flat) >= 10:
            print(f"SINGLE PROCESS: DETAILED K-MODE DEBUG (first 10 modes):", flush=True)
            for i in range(min(10, len(k_flat))):
                print(f"  Mode {i}: k={k_flat[i]:.6f}, P={power_flat[i]:.6e}", flush=True)
    
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