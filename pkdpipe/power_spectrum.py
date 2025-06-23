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
                
    # CRITICAL FIX: Scale particle coordinates from [0,1] to [0,box_size]
    # PKDGrav3 TPS files store coordinates as fractions of box size
    memory_before_scaling = get_memory_usage()
    
    particles['x'] *= box_size
    particles['y'] *= box_size  
    particles['z'] *= box_size
    
    memory_after_scaling = get_memory_usage()
    scaling_overhead = memory_after_scaling - memory_before_scaling
    print(f"🔍 MPI RANK {rank}: After coordinate scaling: {memory_after_scaling:.2f} GB (+{scaling_overhead:.2f} GB)", flush=True)
    
    # Verify scaling worked
    y_scaled_min = float(np.min(particles['y']))
    y_scaled_max = float(np.max(particles['y']))
    
    # Calculate all spatial domains
    all_domains = []
    for proc in range(size):
        proc_y_min, proc_y_max, _, _ = get_spatial_domain_with_ghosts(
            proc, size, ngrid, assignment_scheme
        )
        all_domains.append((proc_y_min, proc_y_max))
    
    # MEMORY OPTIMIZATION: Use float32 cell_size and convert y coordinates in-place
    cell_size = np.float32(box_size / ngrid)
    
    # Step 1: Determine destination process for each local particle
    memory_before_grid_calc = get_memory_usage()
    
    # MEMORY OPTIMIZATION: Convert y coordinates to grid units in-place (no new array)
    particles['y'] /= cell_size  # Now particles['y'] contains grid coordinates
    n_local = len(particles['x'])
    
    memory_after_grid_calc = get_memory_usage()
    grid_calc_overhead = memory_after_grid_calc - memory_before_grid_calc
    print(f"🔍 MPI RANK {rank}: After in-place y grid conversion: {memory_after_grid_calc:.2f} GB (+{grid_calc_overhead:.2f} GB)", flush=True)
    
    # DEBUG: Check grid coordinate conversion
    y_grid_min = float(np.min(particles['y']))
    y_grid_max = float(np.max(particles['y']))
    
    # Show all process domains in grid coordinates
    for i, (proc_y_min, proc_y_max) in enumerate(all_domains):
        
    # Assign particles to destination processes
    memory_before_dest_calc = get_memory_usage()
    dest_processes = np.zeros(n_local, dtype=np.int32)
    for i, (proc_y_min, proc_y_max) in enumerate(all_domains):
        in_proc_domain = (particles['y'] >= proc_y_min) & (particles['y'] < proc_y_max)
        dest_processes = np.where(in_proc_domain, i, dest_processes)
    
    memory_after_dest_calc = get_memory_usage()
    dest_calc_overhead = memory_after_dest_calc - memory_before_dest_calc
    print(f"🔍 MPI RANK {rank}: After destination calculation: {memory_after_dest_calc:.2f} GB (+{dest_calc_overhead:.2f} GB)", flush=True)
    
    # Convert y coordinates back to physical units before particle exchange
    particles['y'] *= cell_size
    
    # Step 2: Count particles going to each process
    send_counts = np.zeros(size, dtype=np.int32)
    for dest_proc in range(size):
        count_to_dest = int(np.sum(dest_processes == dest_proc))
        send_counts[dest_proc] = count_to_dest
    
    
    # Step 3: Exchange send counts with all processes
    recv_counts = comm.alltoall(send_counts.tolist())
    total_recv = sum(recv_counts)
    
    
    # Step 4: Pack particles by destination process
    memory_before_packing = get_memory_usage()
    particles_by_dest = {}
    for dest_proc in range(size):
        going_to_dest = (dest_processes == dest_proc)
        dest_particles = {}
        for key in ['x', 'y', 'z', 'mass']:
            dest_particles[key] = np.array(particles[key][going_to_dest])
        particles_by_dest[dest_proc] = dest_particles
    
    memory_after_packing = get_memory_usage()
    packing_overhead = memory_after_packing - memory_before_packing
    print(f"🔍 MPI RANK {rank}: After particle packing: {memory_after_packing:.2f} GB (+{packing_overhead:.2f} GB)", flush=True)
    
    # Save dtypes before freeing the original particles array
    particle_dtypes = {key: particles[key].dtype for key in ['x', 'y', 'z', 'mass']}
    
    # Memory cleanup: Free original particles array (saves ~1.5GB per process)
    memory_before_cleanup = get_memory_usage()
    print(f"🔍 MPI RANK {rank}: freeing original particles array", flush=True)
    del particles
    del dest_processes  # Also free the destination assignment array
    
    memory_after_cleanup = get_memory_usage()
    cleanup_savings = memory_before_cleanup - memory_after_cleanup
    print(f"🔍 MPI RANK {rank}: After particle cleanup: {memory_after_cleanup:.2f} GB (-{cleanup_savings:.2f} GB)", flush=True)
    
    # Step 5: Deadlock-free MPI exchange using separate send/recv phases
    memory_before_redistribution = get_memory_usage()
    redistributed_particles = {'x': [], 'y': [], 'z': [], 'mass': []}
    
    # Phase 1: Keep my own particles first
    my_particles = particles_by_dest[rank]
    for key in ['x', 'y', 'z', 'mass']:
        if len(my_particles[key]) > 0:
            redistributed_particles[key].append(my_particles[key])
    
    print(f"🔍 MPI RANK {rank}: kept {len(my_particles['x'])} own particles", flush=True)
    
    # Phase 2: Non-blocking chunked sends to all other processes
    send_requests = []
    for dest_proc in range(size):
        if dest_proc != rank:
            send_particles = particles_by_dest[dest_proc]
            send_count = len(send_particles['x'])
            
            if send_count > 0:
                print(f"🔍 MPI RANK {rank}: sending {send_count} particles to rank {dest_proc}", flush=True)
                # Use chunked send to avoid large message problems
                chunk_requests = chunked_mpi_isend(send_particles, dest_proc, comm, chunk_size_mb=512)
                send_requests.extend(chunk_requests)
    
    memory_after_sends = get_memory_usage()
    send_overhead = memory_after_sends - memory_before_redistribution
    print(f"🔍 MPI RANK {rank}: After initiating sends: {memory_after_sends:.2f} GB (+{send_overhead:.2f} GB)", flush=True)
    
    # Phase 3: Blocking chunked receives from all other processes  
    for src_proc in range(size):
        if src_proc != rank:
            recv_count = recv_counts[src_proc]
            
            if recv_count > 0:
                memory_before_recv = get_memory_usage()
                print(f"🔍 MPI RANK {rank}: receiving {recv_count} particles from rank {src_proc}", flush=True)
                # Use chunked receive to handle large messages
                recv_particles = chunked_mpi_recv(src_proc, comm)
                
                memory_after_recv = get_memory_usage()
                recv_overhead = memory_after_recv - memory_before_recv
                print(f"🔍 MPI RANK {rank}: After receiving from rank {src_proc}: {memory_after_recv:.2f} GB (+{recv_overhead:.2f} GB)", flush=True)
                
                # Add received particles
                for key in ['x', 'y', 'z', 'mass']:
                    if len(recv_particles[key]) > 0:
                        redistributed_particles[key].append(recv_particles[key])
                        
                current_memory = get_memory_usage()
                print(f"🔍 MPI RANK {rank}: After appending particles from rank {src_proc}: {current_memory:.2f} GB", flush=True)
    
    # Phase 4: Wait for all sends to complete
    print(f"🔍 MPI RANK {rank}: waiting for {len(send_requests)} send operations to complete", flush=True)
    for req in send_requests:
        req.wait()
    
    # Memory cleanup: Free particles_by_dest array after MPI exchange (saves additional memory)
    memory_before_dest_cleanup = get_memory_usage()
    print(f"🔍 MPI RANK {rank}: freeing particles_by_dest array", flush=True)
    del particles_by_dest
    
    memory_after_dest_cleanup = get_memory_usage()
    dest_cleanup_savings = memory_before_dest_cleanup - memory_after_dest_cleanup
    print(f"🔍 MPI RANK {rank}: After particles_by_dest cleanup: {memory_after_dest_cleanup:.2f} GB (-{dest_cleanup_savings:.2f} GB)", flush=True)
    
    # Phase 5: Concatenate all received particles
    memory_before_concat = get_memory_usage()
    print(f"🔍 MPI RANK {rank}: Starting final concatenation...", flush=True)
    
    final_particles = {}
    for key in ['x', 'y', 'z', 'mass']:
        if redistributed_particles[key]:
            print(f"🔍 MPI RANK {rank}: Concatenating {len(redistributed_particles[key])} arrays for key '{key}'", flush=True)
            final_particles[key] = np.concatenate(redistributed_particles[key])
        else:
            final_particles[key] = np.array([], dtype=particle_dtypes[key])
    
    memory_after_concat = get_memory_usage()
    concat_overhead = memory_after_concat - memory_before_concat
    print(f"🔍 MPI RANK {rank}: After concatenation: {memory_after_concat:.2f} GB (+{concat_overhead:.2f} GB)", flush=True)
    
    redistributed_particles = final_particles
    
    total_memory_increase = memory_after_concat - initial_memory
    print(f"🔍 MPI RANK {rank}: REDISTRIBUTION COMPLETE - {total_recv} particles, memory: {memory_after_concat:.2f} GB (+{total_memory_increase:.2f} GB total)", flush=True)
    
    # Step 6: Verify particles are in correct spatial domain (optional debug check)
    if total_recv > 0:
        # MEMORY OPTIMIZATION: Convert to grid coordinates in-place for verification
        redistributed_particles['y'] /= cell_size
        in_domain_check = (redistributed_particles['y'] >= my_y_min) & (redistributed_particles['y'] < my_y_max)
        final_in_domain = int(np.sum(in_domain_check))
            # Convert back to physical coordinates
        redistributed_particles['y'] *= cell_size
    
    # MPI Barrier: Ensure all processes complete redistribution before proceeding
    comm.barrier()
    
    return redistributed_particles, my_y_min, my_y_max, base_y_min, base_y_max




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
    
    # Use SLURM environment variables instead of JAX for process information
    process_id = int(os.environ.get('SLURM_PROCID', '0'))
    n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        
    # Calculate my spatial domain
    my_y_min, my_y_max, base_y_min, base_y_max = get_spatial_domain_with_ghosts(
        process_id, n_processes, ngrid, assignment_scheme
    )
    
    
    # Calculate all spatial domains
    all_domains = []
    for proc in range(n_processes):
        proc_y_min, proc_y_max, _, _ = get_spatial_domain_with_ghosts(
            proc, n_processes, ngrid, assignment_scheme
        )
        all_domains.append((proc_y_min, proc_y_max))
    
    
    # MEMORY OPTIMIZATION: Use float32 cell_size to avoid type promotion
    cell_size = np.float32(box_size / ngrid)
    
    # Import modules needed by all processes
    import tempfile
    import pickle
    import os
    import time
    
    if process_id == 0:
        # Master process: read chunks and distribute
        
        # TODO: Implement chunk reading loop
        # For now, use existing data and distribute it
        
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
            for key in ['x', 'y', 'z', 'mass']:
                proc_particles[key] = particles[key][in_proc_domain]
            
            count = len(proc_particles['x'])
                
            # Write particles for destination process
            filename = f"{temp_dir}/particles_for_process_{dest_proc}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(proc_particles, f)
        
    
    else:
        # Worker processes: wait for particles
    
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
        
        # Clean up
        os.remove(filename)
    else:
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
    
    # Use numpy for array operations (JAX will be initialized later if needed)
    import numpy as np
    
    # Use SLURM environment variables instead of JAX for process information
    process_id = int(os.environ.get('SLURM_PROCID', '0'))
    n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        
    # Calculate spatial domains with ghost zones
    my_y_min, my_y_max, base_y_min, base_y_max = get_spatial_domain_with_ghosts(
        process_id, n_processes, ngrid, assignment_scheme
    )
    
    
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
        
        
        # Step 2: Determine destination process for each local particle
        # MEMORY OPTIMIZATION: Convert y coordinates to grid units in-place
        particles['y'] /= cell_size  # Now particles['y'] contains grid coordinates
        n_local = len(particles['x'])
        
        # DEBUG: Check spatial distribution of local particles (now in grid coordinates)
        y_grid_min = float(np.min(particles['y']))
        y_grid_max = float(np.max(particles['y']))
        
        # Assign each particle to a destination process based on y-coordinate
        dest_processes = np.zeros(n_local, dtype=np.int32)
        for i, (proc_y_min, proc_y_max) in enumerate(all_domains):
            in_proc_domain = (particles['y'] >= proc_y_min) & (particles['y'] < proc_y_max)
            dest_processes = np.where(in_proc_domain, i, dest_processes)
            n_in_domain = int(np.sum(in_proc_domain))
            
        
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
            for key in ['x', 'y', 'z', 'mass']:
                dest_particles[key] = particles[key][going_to_dest]
            particles_by_dest[dest_proc] = dest_particles
        
        
        # Step 4: Use file-based particle exchange for cross-node communication
        
        import tempfile
        import pickle
        import os
        import time
        
        # Create a temporary directory for particle exchange
        temp_dir = "/tmp/particle_exchange"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Step 4a: Write particles that belong to each process to separate files
                
            for dest_proc in range(n_processes):
                dest_particles = particles_by_dest[dest_proc]
                dest_count = len(dest_particles['x'])
                
                if dest_count > 0:
                    # Write particles to file for destination process
                    filename = f"{temp_dir}/particles_from_{process_id}_to_{dest_proc}.pkl"
                    with open(filename, 'wb') as f:
                        pickle.dump(dest_particles, f)
                        
            # Step 4b: Synchronization barrier - wait for all processes to write their files
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
                        raise Exception("Timeout waiting for particle exchange files")
            
            # Step 4c: Read particles sent to this process
                received_particles_list = []
            
            for src_proc in range(n_processes):
                filename = f"{temp_dir}/particles_from_{src_proc}_to_{process_id}.pkl"
                if os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        src_particles = pickle.load(f)
                    
                    src_count = len(src_particles['x'])
                    if src_count > 0:
                        received_particles_list.append(src_particles)
                                    
                    # Clean up file
                    os.remove(filename)
            
            # Step 4d: Combine all received particles
            if received_particles_list:
                redistributed_particles = {}
                for key in ['x', 'y', 'z', 'mass']:
                    key_arrays = [p[key] for p in received_particles_list if len(p[key]) > 0]
                    if key_arrays:
                        redistributed_particles[key] = np.concatenate(key_arrays)
                    else:
                        redistributed_particles[key] = np.array([], dtype=np.float32)
            else:
                redistributed_particles = {}
                for key in ['x', 'y', 'z', 'mass']:
                    redistributed_particles[key] = np.array([], dtype=np.float32)
            
            domain_particles = len(redistributed_particles['x'])
                
            # Verify particles are in correct spatial domain (convert to grid coordinates in-place)
            if domain_particles > 0:
                redistributed_particles['y'] /= cell_size
                in_domain_check = (redistributed_particles['y'] >= my_y_min) & (redistributed_particles['y'] < my_y_max)
                final_in_domain = int(np.sum(in_domain_check))
                        # Convert back to physical coordinates
                redistributed_particles['y'] *= cell_size
            
        except Exception as e:
                    
            # Fallback: Use only local particles (not ideal but better than crashing)
            # Note: particles['y'] is already in grid coordinates at this point
            in_my_domain = (particles['y'] >= my_y_min) & (particles['y'] < my_y_max)
            
            redistributed_particles = {}
            for key in ['x', 'y', 'z', 'mass']:
                redistributed_particles[key] = particles[key][in_my_domain]
            
            domain_particles = len(redistributed_particles['x'])
            
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
        for key in ['x', 'y', 'z', 'mass']:
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
    estimated_data_gb = n_particles * 24 / 1024**3
    print(f"🔍 RECV RANK {my_rank}: receiving {n_particles:,} particles from rank {src_proc} in {n_chunks} chunks ({estimated_data_gb:.2f} GB estimated)", flush=True)
    print(f"🔍 RECV RANK {my_rank}: Memory before receive: {memory_before_recv:.2f} GB", flush=True)
    
    if n_particles == 0:
        return {'x': np.array([]), 'y': np.array([]), 'z': np.array([]), 'mass': np.array([])}
    
    # Initialize result arrays as lists to accumulate chunks
    particles = {'x': [], 'y': [], 'z': [], 'mass': []}
    
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
        
        # Accumulate chunk data
        chunk_particles = len(chunk['x'])
        chunk_size_gb = chunk_particles * 24 / 1024**3
        print(f"🔍 RECV RANK {my_rank}: Chunk {chunk_idx}: received {chunk_particles:,} particles ({chunk_size_gb:.3f} GB), memory: {memory_after_chunk_recv:.2f} GB (+{chunk_recv_overhead:.3f} GB)", flush=True)
        
        for key in ['x', 'y', 'z', 'mass']:
            particles[key].append(chunk[key])
        
        memory_after_append = get_memory_usage()
        append_overhead = memory_after_append - memory_after_chunk_recv
        print(f"🔍 RECV RANK {my_rank}: After chunk {chunk_idx} append: {memory_after_append:.2f} GB (+{append_overhead:.3f} GB)", flush=True)
    
    # Concatenate all chunks - THIS IS WHERE OOM LIKELY OCCURS
    memory_before_concat = get_memory_usage()
    print(f"🔍 RECV RANK {my_rank}: Starting concatenation of {n_chunks} chunks...", flush=True)
    
    final_particles = {}
    for key in ['x', 'y', 'z', 'mass']:
        if particles[key]:
            memory_before_key = get_memory_usage()
            print(f"🔍 RECV RANK {my_rank}: Concatenating key '{key}' ({len(particles[key])} arrays)...", flush=True)
            
            final_particles[key] = np.concatenate(particles[key])
            
            memory_after_key = get_memory_usage()
            key_overhead = memory_after_key - memory_before_key
            print(f"🔍 RECV RANK {my_rank}: Key '{key}' concatenated: {memory_after_key:.2f} GB (+{key_overhead:.3f} GB)", flush=True)
        else:
            final_particles[key] = np.array([])
    
    memory_after_concat = get_memory_usage()
    concat_overhead = memory_after_concat - memory_before_concat
    total_overhead = memory_after_concat - memory_before_recv
    print(f"🔍 RECV RANK {my_rank}: CONCATENATION COMPLETE: {memory_after_concat:.2f} GB (+{concat_overhead:.2f} GB concat, +{total_overhead:.2f} GB total)", flush=True)
    
    # Clean up chunk lists to free memory
    print(f"🔍 RECV RANK {my_rank}: Cleaning up chunk lists...", flush=True)
    del particles
    
    memory_after_cleanup = get_memory_usage()
    cleanup_savings = memory_after_concat - memory_after_cleanup
    print(f"🔍 RECV RANK {my_rank}: After cleanup: {memory_after_cleanup:.2f} GB (-{cleanup_savings:.2f} GB)", flush=True)
    
    return final_particles
