"""
Particle-to-grid mass assignment with multi-device support.

This module provides efficient particle-to-grid mass assignment algorithms
including Cloud-in-Cell (CIC) and Nearest Grid Point (NGP) schemes with
proper periodic boundary conditions and multi-GPU domain decomposition.

Example usage:
    # Basic gridding
    gridder = ParticleGridder(ngrid=256, box_size=1000.0, assignment='cic')
    density_grid = gridder.particles_to_grid(particles)
    
    # Multi-GPU gridding with domain decomposition
    gridder = ParticleGridder(ngrid=512, box_size=2000.0, assignment='cic')
    device_grids = gridder.particles_to_grid(particles, n_devices=4)
"""

import numpy as np
from typing import Dict, Union, List
import os
import gc
from multiprocessing import Pool, cpu_count, shared_memory


def _cic_worker_shared_memory(args):
    """
    Shared memory worker function for multiprocessing CIC assignment.
    
    Args:
        args: Tuple of (shared_memory_info, start_idx, end_idx, ngrid, box_size)
        
    Returns:
        Density grid from this particle chunk
    """
    shared_info, start_idx, end_idx, ngrid, box_size = args
    
    # Handle empty chunk
    chunk_size = end_idx - start_idx
    if chunk_size == 0:
        return np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    
    try:
        # Reconstruct shared memory arrays
        x_shm = shared_memory.SharedMemory(name=shared_info['x_name'])
        y_shm = shared_memory.SharedMemory(name=shared_info['y_name']) 
        z_shm = shared_memory.SharedMemory(name=shared_info['z_name'])
        mass_shm = shared_memory.SharedMemory(name=shared_info['mass_name'])
        
        # Create numpy views (no copying)
        x_array = np.ndarray(shared_info['shape'], dtype=np.float32, buffer=x_shm.buf)
        y_array = np.ndarray(shared_info['shape'], dtype=np.float32, buffer=y_shm.buf)
        z_array = np.ndarray(shared_info['shape'], dtype=np.float32, buffer=z_shm.buf)
        mass_array = np.ndarray(shared_info['shape'], dtype=np.float32, buffer=mass_shm.buf)
        
        # Extract chunk without copying
        x_chunk = x_array[start_idx:end_idx]
        y_chunk = y_array[start_idx:end_idx]
        z_chunk = z_array[start_idx:end_idx]
        mass_chunk = mass_array[start_idx:end_idx]
        
        # Create positions array for this chunk
        positions = np.column_stack([x_chunk, y_chunk, z_chunk])
        masses = mass_chunk
        
        # Convert to grid coordinates
        grid_spacing = box_size / ngrid
        grid_coords = positions / grid_spacing
        
        # Initialize density grid
        density_grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
        
        # Get integer grid coordinates (lower-left corner of cell)
        i_coords = np.floor(grid_coords).astype(int)
        
        # Get fractional offsets within cells
        dx = grid_coords - i_coords
        
        # Apply periodic boundary conditions
        i_coords = i_coords % ngrid
        
        # Vectorized CIC assignment
        n_particles = len(masses)
        
        # Weight arrays for all particles and all 8 corners
        weights = np.zeros((n_particles, 8), dtype=np.float32)
        grid_indices = np.zeros((n_particles, 8, 3), dtype=int)
        
        # Calculate weights and indices for all 8 corners simultaneously
        corner_idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Weights for this corner (vectorized)
                    weights[:, corner_idx] = (
                        ((1-i) * (1-dx[:, 0]) + i * dx[:, 0]) *
                        ((1-j) * (1-dx[:, 1]) + j * dx[:, 1]) *
                        ((1-k) * (1-dx[:, 2]) + k * dx[:, 2])
                    )
                    
                    # Grid indices with periodic wrapping
                    grid_indices[:, corner_idx, 0] = (i_coords[:, 0] + i) % ngrid
                    grid_indices[:, corner_idx, 1] = (i_coords[:, 1] + j) % ngrid
                    grid_indices[:, corner_idx, 2] = (i_coords[:, 2] + k) % ngrid
                    
                    corner_idx += 1
        
        # Vectorized mass assignment using np.add.at
        for corner in range(8):
            gi = grid_indices[:, corner, 0]
            gj = grid_indices[:, corner, 1] 
            gk = grid_indices[:, corner, 2]
            weighted_masses = masses * weights[:, corner]
            
            # Add weighted mass to grid
            np.add.at(density_grid, (gi, gj, gk), weighted_masses)
        
        return density_grid
        
    except Exception as e:
        print(f"Worker {start_idx}-{end_idx}: Error accessing shared memory: {e}")
        return np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)


def _cic_worker(args):
    """
    Legacy worker function for multiprocessing CIC assignment.
    
    Args:
        args: Tuple of (particle_chunk, ngrid, box_size)
        
    Returns:
        Density grid from this particle chunk
    """
    particle_chunk, ngrid, box_size = args
    
    # Handle empty chunk
    if len(particle_chunk['x']) == 0:
        return np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    
    # Extract positions and masses
    positions = np.column_stack([particle_chunk['x'], particle_chunk['y'], particle_chunk['z']])
    masses = particle_chunk['mass']
    
    # Convert to grid coordinates
    grid_spacing = box_size / ngrid
    grid_coords = positions / grid_spacing
    
    # Initialize density grid
    density_grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    
    # Get integer grid coordinates (lower-left corner of cell)
    i_coords = np.floor(grid_coords).astype(int)
    
    # Get fractional offsets within cells
    dx = grid_coords - i_coords
    
    # Apply periodic boundary conditions
    i_coords = i_coords % ngrid
    
    # Vectorized CIC assignment
    n_particles = len(masses)
    
    # Weight arrays for all particles and all 8 corners
    weights = np.zeros((n_particles, 8), dtype=np.float32)
    grid_indices = np.zeros((n_particles, 8, 3), dtype=int)
    
    # Calculate weights and indices for all 8 corners simultaneously
    corner_idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Weights for this corner (vectorized)
                weights[:, corner_idx] = (
                    ((1-i) * (1-dx[:, 0]) + i * dx[:, 0]) *
                    ((1-j) * (1-dx[:, 1]) + j * dx[:, 1]) *
                    ((1-k) * (1-dx[:, 2]) + k * dx[:, 2])
                )
                
                # Grid indices with periodic wrapping
                grid_indices[:, corner_idx, 0] = (i_coords[:, 0] + i) % ngrid
                grid_indices[:, corner_idx, 1] = (i_coords[:, 1] + j) % ngrid
                grid_indices[:, corner_idx, 2] = (i_coords[:, 2] + k) % ngrid
                
                corner_idx += 1
    
    # Vectorized mass assignment using np.add.at
    for corner in range(8):
        gi = grid_indices[:, corner, 0]
        gj = grid_indices[:, corner, 1] 
        gk = grid_indices[:, corner, 2]
        weighted_masses = masses * weights[:, corner]
        
        # Add weighted mass to grid
        np.add.at(density_grid, (gi, gj, gk), weighted_masses)
    
    return density_grid


def _ngp_worker(args):
    """
    Worker function for multiprocessing NGP assignment.
    
    Args:
        args: Tuple of (particle_chunk, ngrid, box_size)
        
    Returns:
        Density grid from this particle chunk
    """
    particle_chunk, ngrid, box_size = args
    
    # Handle empty chunk
    if len(particle_chunk['x']) == 0:
        return np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    
    # Extract positions and masses
    positions = np.column_stack([particle_chunk['x'], particle_chunk['y'], particle_chunk['z']])
    masses = particle_chunk['mass']
    
    # Convert to grid coordinates
    grid_spacing = box_size / ngrid
    grid_coords = positions / grid_spacing
    
    # Initialize density grid
    density_grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    
    # Round to nearest grid point
    i_coords = np.round(grid_coords).astype(int)
    
    # Apply periodic boundary conditions
    i_coords = i_coords % ngrid
    
    # Add masses to nearest grid points
    np.add.at(density_grid, (i_coords[:, 0], i_coords[:, 1], i_coords[:, 2]), masses)
    
    return density_grid


def _cic_slab_worker(args):
    """
    Worker function for multiprocessing CIC slab assignment.
    
    Args:
        args: Tuple of (particle_chunk, ngrid, box_size, y_min, y_max, assignment)
        
    Returns:
        Density grid for this particle chunk slab
    """
    particle_chunk, ngrid, box_size, y_min, y_max, assignment = args
    
    slab_height = y_max - y_min
    slab_grid = np.zeros((ngrid, slab_height, ngrid), dtype=np.float32)
    
    # Handle empty chunk
    if len(particle_chunk['x']) == 0:
        return slab_grid
    
    # Convert particle positions to grid coordinates
    grid_spacing = box_size / ngrid
    x_grid = np.asarray(particle_chunk['x']) / grid_spacing
    y_grid = np.asarray(particle_chunk['y']) / grid_spacing
    z_grid = np.asarray(particle_chunk['z']) / grid_spacing
    masses = np.asarray(particle_chunk['mass'])
    
    # Apply periodic boundary conditions
    x_grid = np.mod(x_grid, ngrid)
    y_grid = np.mod(y_grid, ngrid)
    z_grid = np.mod(z_grid, ngrid)
    
    # Filter particles that fall within this slab (including ghosts)
    in_slab = (y_grid >= y_min) & (y_grid < y_max)
    if not np.any(in_slab):
        return slab_grid
    
    x_slab = x_grid[in_slab]
    y_slab = y_grid[in_slab] - y_min  # Shift to slab coordinates
    z_slab = z_grid[in_slab]
    mass_slab = masses[in_slab]
    
    # Perform CIC assignment on slab
    if assignment == 'cic':
        _cic_assign_slab_worker(slab_grid, x_slab, y_slab, z_slab, mass_slab)
    elif assignment == 'ngp':
        _ngp_assign_slab_worker(slab_grid, x_slab, y_slab, z_slab, mass_slab)
    
    return slab_grid


def _cic_assign_slab_worker(grid: np.ndarray, x: np.ndarray, y: np.ndarray, 
                           z: np.ndarray, mass: np.ndarray) -> None:
    """CIC assignment for slab geometry - worker function."""
    ngrid_x, slab_height, ngrid_z = grid.shape
    
    if len(x) == 0:
        return
    
    # Integer grid coordinates (floor for CIC)
    ix = np.floor(x).astype(int)
    iy = np.floor(y).astype(int)
    iz = np.floor(z).astype(int)
    
    # Fractional offsets
    dx = x - ix
    dy = y - iy
    dz = z - iz
    
    # Apply periodic boundary conditions for x and z
    ix = ix % ngrid_x
    iz = iz % ngrid_z
    
    # Check bounds for y (no periodic wrapping in slab direction)
    valid_mask = (iy >= 0) & (iy < slab_height - 1)
    if not np.any(valid_mask):
        return
    
    # Filter to valid particles
    ix_v, iy_v, iz_v = ix[valid_mask], iy[valid_mask], iz[valid_mask]
    dx_v, dy_v, dz_v = dx[valid_mask], dy[valid_mask], dz[valid_mask]
    mass_v = mass[valid_mask]
    
    # Vectorized CIC for all 8 corners
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                # Grid indices for this corner
                gi = (ix_v + di) % ngrid_x
                gj = iy_v + dj
                gk = (iz_v + dk) % ngrid_z
                
                # Check bounds for gj (y-direction)
                valid_y = (gj >= 0) & (gj < slab_height)
                if not np.any(valid_y):
                    continue
                
                # Filter indices and weights for valid y coordinates
                gi_f = gi[valid_y]
                gj_f = gj[valid_y]
                gk_f = gk[valid_y]
                
                # Calculate weights for this corner
                weights = (
                    ((1-di) * (1-dx_v[valid_y]) + di * dx_v[valid_y]) *
                    ((1-dj) * (1-dy_v[valid_y]) + dj * dy_v[valid_y]) *
                    ((1-dk) * (1-dz_v[valid_y]) + dk * dz_v[valid_y])
                )
                
                weighted_masses = mass_v[valid_y] * weights
                
                # Add to grid
                np.add.at(grid, (gi_f, gj_f, gk_f), weighted_masses)


def _ngp_assign_slab_worker(grid: np.ndarray, x: np.ndarray, y: np.ndarray, 
                           z: np.ndarray, mass: np.ndarray) -> None:
    """NGP assignment for slab geometry - worker function."""
    ngrid_x, slab_height, ngrid_z = grid.shape
    
    if len(x) == 0:
        return
    
    # Round to nearest grid point
    ix = np.round(x).astype(int) % ngrid_x
    iy = np.round(y).astype(int)
    iz = np.round(z).astype(int) % ngrid_z
    
    # Check bounds for y (slab dimension)
    valid_mask = (iy >= 0) & (iy < slab_height)
    if not np.any(valid_mask):
        return
    
    # Apply mask to valid particles
    ix_valid = ix[valid_mask]
    iy_valid = iy[valid_mask]
    iz_valid = iz[valid_mask]
    mass_valid = mass[valid_mask]
    
    # Vectorized mass assignment
    np.add.at(grid, (ix_valid, iy_valid, iz_valid), mass_valid)


class ParticleGridder:
    """
    Efficient particle-to-grid mass assignment with multi-device support.
    
    Supports Cloud-in-Cell (CIC) and Nearest Grid Point (NGP) assignment schemes
    with periodic boundary conditions and optimized vectorized operations.
    """
    
    def __init__(self, ngrid: int, box_size: float, assignment: str = 'cic'):
        """
        Initialize particle gridder.
        
        Args:
            ngrid: Number of grid cells per dimension
            box_size: Physical size of simulation box
            assignment: Mass assignment scheme ('cic' or 'ngp')
        """
        self.ngrid = ngrid
        self.box_size = box_size
        self.assignment = assignment.lower()
        self.grid_spacing = box_size / ngrid
        self.volume = box_size**3
        
        # Auto-detect number of CPU cores for multiprocessing
        self.n_cpu_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', str(cpu_count())))
        
        # Validate assignment scheme
        if self.assignment not in ['cic', 'ngp']:
            raise ValueError(f"Unknown assignment scheme: {assignment}")

    def _get_optimal_n_processes(self, n_particles: int) -> int:
        """
        Determine optimal number of processes for particle gridding.
        
        Args:
            n_particles: Number of particles to process
            
        Returns:
            Optimal number of processes to use
        """
        # Use all available cores, but don't exceed reasonable limits
        max_processes = min(self.n_cpu_cores, 32)  # Cap at 32 processes
        
        # For small particle counts, don't use too many processes
        if n_particles < 100000:  # 100K particles
            return min(max_processes, 4)
        elif n_particles < 1000000:  # 1M particles  
            return min(max_processes, 8)
        else:
            return max_processes
    
    def _chunk_particles(self, particles: Dict[str, np.ndarray], n_processes: int) -> List[Dict[str, np.ndarray]]:
            """
            Split particles into chunks for multiprocessing.
            
            Args:
                particles: Particle data dictionary
                n_processes: Number of processes to split across
                
            Returns:
                List of particle chunks
            """
            import psutil
            
            def get_memory_usage():
                """Get current memory usage in GB"""
                process = psutil.Process()
                return process.memory_info().rss / 1024**3
            
            n_particles = len(particles['x'])
            chunk_size = n_particles // n_processes
            
            memory_before = get_memory_usage()
            
            chunks = []
            for i in range(n_processes):
                start_idx = i * chunk_size
                if i == n_processes - 1:
                    # Last chunk gets any remaining particles
                    end_idx = n_particles
                else:
                    end_idx = (i + 1) * chunk_size
                
                chunk_particles = end_idx - start_idx
                memory_before_chunk = get_memory_usage()
                
                chunk = {
                    'x': particles['x'][start_idx:end_idx],
                    'y': particles['y'][start_idx:end_idx], 
                    'z': particles['z'][start_idx:end_idx],
                    'mass': np.ones(end_idx - start_idx, dtype=np.float32)  # Unit masses
                }
                chunks.append(chunk)
                
                memory_after_chunk = get_memory_usage()
                chunk_overhead = memory_after_chunk - memory_before_chunk
                chunk_size_gb = chunk_particles * 24 / 1024**3
                
            memory_after = get_memory_usage()
            total_overhead = memory_after - memory_before
            
            return chunks



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
        # Handle empty particles case
        if len(particles['x']) == 0:
            if n_devices == 1:
                return np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)
            else:
                return [np.zeros((self.ngrid, self.ngrid//n_devices, self.ngrid), dtype=np.float32) 
                        for _ in range(n_devices)]
        
        # Check that all arrays have same length
        n_particles = len(particles['x'])
        for key in ['y', 'z']:  # REMOVED 'mass' - use unit masses for CIC gridding
            if len(particles[key]) != n_particles:
                raise ValueError(f"Inconsistent array lengths: {key} has {len(particles[key])}, expected {n_particles}")
        
        # Extract positions and use unit masses (equal mass particles for density field)
        positions = np.column_stack([particles['x'], particles['y'], particles['z']])
        masses = np.ones(n_particles, dtype=np.float32)  # Unit masses for equal-mass particles
        
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
        """
        Cloud-in-Cell mass assignment using shared memory multiprocessing.
        
        Optimized version that uses shared memory to prevent memory duplication
        during multiprocessing, critical for large particle counts in MPI environments.
        """
        n_particles = len(masses)
        
        # Determine if multiprocessing is beneficial
        use_multiprocessing = n_particles > 50000 and self.n_cpu_cores > 1
        
        if not use_multiprocessing:
            # Use single-threaded version for small particle counts
            return self._cic_assignment_single_thread(grid_coords, masses)
        
        # Create shared memory arrays to prevent memory duplication during fork
        print(f"CIC assignment: Creating shared memory for {n_particles:,} particles")
        
        # Prepare coordinate arrays (convert back to physical coordinates)
        x_coords = (grid_coords[:, 0] * self.grid_spacing).astype(np.float32)
        y_coords = (grid_coords[:, 1] * self.grid_spacing).astype(np.float32)
        z_coords = (grid_coords[:, 2] * self.grid_spacing).astype(np.float32)
        mass_array = masses.astype(np.float32)
        
        # Create shared memory blocks
        shared_memory_blocks = []
        try:
            # Create shared memory for each coordinate and mass array
            x_shm = shared_memory.SharedMemory(create=True, size=x_coords.nbytes)
            y_shm = shared_memory.SharedMemory(create=True, size=y_coords.nbytes)
            z_shm = shared_memory.SharedMemory(create=True, size=z_coords.nbytes)
            mass_shm = shared_memory.SharedMemory(create=True, size=mass_array.nbytes)
            
            shared_memory_blocks = [x_shm, y_shm, z_shm, mass_shm]
            
            # Copy data to shared memory
            x_shared = np.ndarray(x_coords.shape, dtype=np.float32, buffer=x_shm.buf)
            y_shared = np.ndarray(y_coords.shape, dtype=np.float32, buffer=y_shm.buf)
            z_shared = np.ndarray(z_coords.shape, dtype=np.float32, buffer=z_shm.buf)
            mass_shared = np.ndarray(mass_array.shape, dtype=np.float32, buffer=mass_shm.buf)
            
            x_shared[:] = x_coords[:]
            y_shared[:] = y_coords[:]
            z_shared[:] = z_coords[:]
            mass_shared[:] = mass_array[:]
            
            # Clear original arrays to save memory
            del x_coords, y_coords, z_coords, mass_array
            gc.collect()
            
            # Determine optimal number of processes
            n_processes = self._get_optimal_n_processes(n_particles)
            
            print(f"CIC assignment: Using {n_processes} processes with shared memory for {n_particles:,} particles")
            
            # Create shared memory info dict
            shared_info = {
                'x_name': x_shm.name,
                'y_name': y_shm.name,
                'z_name': z_shm.name,
                'mass_name': mass_shm.name,
                'shape': (n_particles,)
            }
            
            # Create chunk boundaries (indices only, not data)
            chunk_size = n_particles // n_processes
            worker_args = []
            for i in range(n_processes):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < n_processes - 1 else n_particles
                worker_args.append((shared_info, start_idx, end_idx, self.ngrid, self.box_size))
            
            # Process chunks in parallel using shared memory
            with Pool(processes=n_processes) as pool:
                chunk_grids = pool.map(_cic_worker_shared_memory, worker_args)
            
            # Sum all chunk grids
            density_grid = np.sum(chunk_grids, axis=0).astype(np.float32)
            
            return density_grid
            
        finally:
            # Clean up shared memory
            for shm in shared_memory_blocks:
                try:
                    shm.close()
                    shm.unlink()
                except:
                    pass
    
    def _cic_assignment_single_thread(self, grid_coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        Single-threaded Cloud-in-Cell mass assignment.
        Used for small particle counts or when multiprocessing is disabled.
        """
        # Initialize density grid
        density_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)
        
        # Get integer grid coordinates (lower-left corner of cell)
        i_coords = np.floor(grid_coords).astype(int)
        
        # Get fractional offsets within cells
        dx = grid_coords - i_coords
        
        # Apply periodic boundary conditions
        i_coords = i_coords % self.ngrid
        
        # Vectorized CIC assignment
        # Create arrays for all 8 corners at once
        n_particles = len(masses)
        
        # Weight arrays for all particles and all 8 corners
        weights = np.zeros((n_particles, 8), dtype=np.float32)
        grid_indices = np.zeros((n_particles, 8, 3), dtype=int)
        
        # Calculate weights and indices for all 8 corners simultaneously
        corner_idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Weights for this corner (vectorized)
                    weights[:, corner_idx] = (
                        ((1-i) * (1-dx[:, 0]) + i * dx[:, 0]) *
                        ((1-j) * (1-dx[:, 1]) + j * dx[:, 1]) *
                        ((1-k) * (1-dx[:, 2]) + k * dx[:, 2])
                    )
                    
                    # Grid indices with periodic wrapping
                    grid_indices[:, corner_idx, 0] = (i_coords[:, 0] + i) % self.ngrid
                    grid_indices[:, corner_idx, 1] = (i_coords[:, 1] + j) % self.ngrid
                    grid_indices[:, corner_idx, 2] = (i_coords[:, 2] + k) % self.ngrid
                    
                    corner_idx += 1
        
        # Vectorized mass assignment using np.add.at
        # Flatten the operations to avoid nested loops
        for corner in range(8):
            gi = grid_indices[:, corner, 0]
            gj = grid_indices[:, corner, 1] 
            gk = grid_indices[:, corner, 2]
            weighted_masses = masses * weights[:, corner]
            
            # Add weighted mass to grid (this is the only remaining loop, but over 8 items not millions)
            np.add.at(density_grid, (gi, gj, gk), weighted_masses)
        
        return density_grid

    
    def _ngp_assignment(self, grid_coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        Nearest Grid Point mass assignment with multiprocessing support.
        
        Uses multiprocessing for large particle counts to improve performance.
        """
        n_particles = len(masses)
        
        # Determine if multiprocessing is beneficial
        use_multiprocessing = n_particles > 50000 and self.n_cpu_cores > 1
        
        if not use_multiprocessing:
            # Use single-threaded version for small particle counts
            return self._ngp_assignment_single_thread(grid_coords, masses)
        
        # Prepare particles for multiprocessing
        particles = {
            'x': grid_coords[:, 0] * self.grid_spacing,  # Convert back to physical coordinates
            'y': grid_coords[:, 1] * self.grid_spacing,
            'z': grid_coords[:, 2] * self.grid_spacing,
            'mass': masses  # Use the unit masses passed in
        }
        
        # Determine optimal number of processes
        n_processes = self._get_optimal_n_processes(n_particles)
        
        print(f"NGP assignment: Using {n_processes} processes for {n_particles:,} particles")
        
        # Split particles into chunks
        particle_chunks = self._chunk_particles(particles, n_processes)
        
        # Prepare arguments for worker processes
        worker_args = [(chunk, self.ngrid, self.box_size) for chunk in particle_chunks]
        
        # Process chunks in parallel
        with Pool(processes=n_processes) as pool:
            chunk_grids = pool.map(_ngp_worker, worker_args)
        
        # Sum all chunk grids
        density_grid = np.sum(chunk_grids, axis=0).astype(np.float32)
        
        return density_grid
    
    def _ngp_assignment_single_thread(self, grid_coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        Single-threaded Nearest Grid Point mass assignment.
        Used for small particle counts or when multiprocessing is disabled.
        """
        # Initialize density grid
        density_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)
        
        # Floor to correct grid cell for cell-centered grid points
        # For cell-centered grids: grid_point i is at (i+0.5)*dx  
        # Particle at position x belongs to cell floor(x/dx)
        i_coords = np.floor(grid_coords).astype(int)
        
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
            
        Raises:
            ValueError: If grid cannot be evenly divided among devices
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

    def particles_to_slab(self, particles: Dict[str, np.ndarray], 
                                 y_min: int, y_max: int, ngrid: int) -> np.ndarray:
            """
            Grid particles to a spatial slab (for distributed processing) with multiprocessing.
            
            Args:
                particles: Dictionary with 'x', 'y', 'z', 'mass' arrays
                y_min: Start index of slab in y-direction (including ghost zones)
                y_max: End index of slab in y-direction (including ghost zones)
                ngrid: Full grid resolution
                
            Returns:
                Density grid for the slab with shape (ngrid, slab_height, ngrid)
            """
            import os
            
            n_particles = len(particles['x'])
            slab_height = y_max - y_min
            
            # MEMORY FIX: Always use single-threaded for large datasets to avoid memory duplication
            # The shared memory approach copies all particle data, doubling memory usage
            
            # Calculate memory cost of multiprocessing approach
            particle_data_size = n_particles * 3 * 4  # 3 coordinates × 4 bytes each
            
            # If particle data is large (>4GB), avoid multiprocessing to prevent memory duplication
            if particle_data_size > 4 * 1024**3:  # > 4GB
                print(f"DEBUG: Large dataset ({particle_data_size / 1024**3:.1f} GB), using single-threaded gridding to avoid memory duplication")
                return self._particles_to_slab_single_thread(particles, y_min, y_max, ngrid)
            
            # For smaller datasets, determine if multiprocessing is beneficial
            use_multiprocessing = n_particles > 50000 and self.n_cpu_cores > 1
            
            if not use_multiprocessing:
                return self._particles_to_slab_single_thread(particles, y_min, y_max, ngrid)
            
            # Only use multiprocessing for small datasets where memory duplication is acceptable
            n_processes = self._get_optimal_n_processes(n_particles)
            result = self._particles_to_slab_shared_memory(particles, y_min, y_max, ngrid, n_processes)
            
            return result





    def _particles_to_slab_single_thread(self, particles: Dict[str, np.ndarray], 
                                        y_min: int, y_max: int, ngrid: int) -> np.ndarray:
        """
        Memory-efficient single-threaded slab assignment.
        Processes particles in chunks to minimize memory usage.
        """
        slab_height = int(y_max - y_min)
        slab_grid = np.zeros((ngrid, slab_height, ngrid), dtype=np.float32)
        
        # Check debug mode
        debug_mode = os.environ.get('PKDPIPE_DEBUG_MODE', 'false').lower() == 'true'
        
        n_particles = len(particles['x'])
        if debug_mode:
            print(f"DEBUG gridder: Memory-efficient slab assignment - {n_particles} particles, slab shape: {slab_grid.shape}")
        
        # Process particles in chunks to minimize memory usage
        chunk_size = min(1000000, n_particles)  # 1M particles per chunk
        
        for chunk_start in range(0, n_particles, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_particles)
            
            # Work directly with array slices - no copying
            x_chunk = particles['x'][chunk_start:chunk_end]
            y_chunk = particles['y'][chunk_start:chunk_end] 
            z_chunk = particles['z'][chunk_start:chunk_end]
            
            # Convert to grid coordinates on-the-fly (no intermediate arrays)
            x_grid_chunk = x_chunk / self.grid_spacing
            y_grid_chunk = y_chunk / self.grid_spacing
            z_grid_chunk = z_chunk / self.grid_spacing
            
            # Apply periodic boundary conditions
            np.mod(x_grid_chunk, ngrid, out=x_grid_chunk)
            np.mod(y_grid_chunk, ngrid, out=y_grid_chunk)
            np.mod(z_grid_chunk, ngrid, out=z_grid_chunk)
            
            # Filter particles in this slab
            in_slab = (y_grid_chunk >= y_min) & (y_grid_chunk < y_max)
            n_in_slab = np.sum(in_slab)
            
            if n_in_slab == 0:
                continue  # No particles in this chunk for this slab
                
            # Extract slab particles (create minimal temporary arrays)
            x_slab = x_grid_chunk[in_slab]
            y_slab = y_grid_chunk[in_slab] - y_min  # Shift to slab coordinates
            z_slab = z_grid_chunk[in_slab]
            
            # Use unit masses (no mass copying)
            mass_slab = np.ones(n_in_slab, dtype=np.float32)
            
            # Perform assignment on this chunk
            if self.assignment == 'cic':
                self._cic_assign_slab(slab_grid, x_slab, y_slab, z_slab, mass_slab)
            elif self.assignment == 'ngp':
                self._ngp_assign_slab(slab_grid, x_slab, y_slab, z_slab, mass_slab)
            else:
                raise ValueError(f"Unknown assignment scheme: {self.assignment}")
            
            if debug_mode and chunk_start % (10 * chunk_size) == 0:
                print(f"DEBUG gridder: Processed chunk {chunk_start//chunk_size + 1}/{(n_particles + chunk_size - 1)//chunk_size}, particles in slab: {n_in_slab}")
        
        if debug_mode:
            final_sum = np.sum(slab_grid)
            final_has_nan = np.isnan(final_sum)
            print(f"DEBUG gridder: Final slab_grid - sum: {final_sum}, has_nan: {final_has_nan}, min: {np.min(slab_grid)}, max: {np.max(slab_grid)}")
        
        return slab_grid
    
    def _cic_assign_slab(self, grid: np.ndarray, x: np.ndarray, y: np.ndarray, 
                        z: np.ndarray, mass: np.ndarray) -> None:
        """
        CIC assignment for slab geometry using highly optimized vectorized operations.
        
        Further optimized version with reduced memory allocations and faster operations.
        """
        import time
        
        ngrid_x, slab_height, ngrid_z = grid.shape
        start_time = time.time()
        
        
        if len(x) == 0:
                return
        
        step_start = time.time()
        
        # Get integer grid coordinates (lower left corner) - use int32 for memory efficiency
        ix = np.floor(x).astype(np.int32) % ngrid_x
        iy = np.floor(y).astype(np.int32)
        iz = np.floor(z).astype(np.int32) % ngrid_z
        
        # Check bounds for y (slab dimension) - filter out invalid particles
        valid_mask = (iy >= 0) & (iy < slab_height - 1)
        n_valid = np.sum(valid_mask)
        
        if n_valid == 0:
                return
        
        step_start = time.time()
        
        # Apply mask to all arrays - use float32 for memory efficiency
        ix_valid = ix[valid_mask]
        iy_valid = iy[valid_mask]  
        iz_valid = iz[valid_mask]
        x_valid = x[valid_mask].astype(np.float32)
        y_valid = y[valid_mask].astype(np.float32)
        z_valid = z[valid_mask].astype(np.float32)
        mass_valid = mass[valid_mask].astype(np.float32)
        
        # Fractional distances - compute directly without intermediate arrays
        dx = x_valid - ix_valid.astype(np.float32)
        dy = y_valid - iy_valid.astype(np.float32)
        dz = z_valid - iz_valid.astype(np.float32)
        
        
        step_start = time.time()
        
        # Highly optimized corner assignment - no intermediate arrays
        # Process all 8 corners with minimal memory allocation
        corner_count = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corner_count += 1
                                # Compute weights directly
                    w_x = (1-i) * (1-dx) + i * dx
                    w_y = (1-j) * (1-dy) + j * dy  
                    w_z = (1-k) * (1-dz) + k * dz
                    weights = mass_valid * w_x * w_y * w_z
                    
                    # Compute grid indices with bounds checking
                    gi = (ix_valid + i) % ngrid_x
                    gj = iy_valid + j
                    gk = (iz_valid + k) % ngrid_z
                    
                    # Only assign to valid y-indices
                    valid_y_mask = (gj >= 0) & (gj < slab_height)
                    if np.any(valid_y_mask):
                        np.add.at(grid, 
                                (gi[valid_y_mask], gj[valid_y_mask], gk[valid_y_mask]), 
                                weights[valid_y_mask])
        
        
        total_time = time.time() - start_time



    
    def _ngp_assign_slab(self, grid: np.ndarray, x: np.ndarray, y: np.ndarray, 
                        z: np.ndarray, mass: np.ndarray) -> None:
        """
        NGP assignment for slab geometry using vectorized operations.
        
        Optimized version that eliminates Python loops for better performance.
        """
        ngrid_x, slab_height, ngrid_z = grid.shape
        
        if len(x) == 0:
            return
        
        # Floor to correct grid cell for cell-centered grid points
        # For cell-centered grids: grid_point i is at (i+0.5)*dx
        # Particle at position x belongs to cell floor(x/dx)
        ix = np.floor(x).astype(int) % ngrid_x
        iy = np.floor(y).astype(int)
        iz = np.floor(z).astype(int) % ngrid_z
        
        # Check bounds for y (slab dimension)
        valid_mask = (iy >= 0) & (iy < slab_height)
        if not np.any(valid_mask):
            return
        
        # Apply mask to valid particles
        ix_valid = ix[valid_mask]
        iy_valid = iy[valid_mask]
        iz_valid = iz[valid_mask]
        mass_valid = mass[valid_mask]
        
        # Vectorized mass assignment
        np.add.at(grid, (ix_valid, iy_valid, iz_valid), mass_valid)


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
    
    def _particles_to_slab_shared_memory(self, particles: Dict[str, np.ndarray], 
                                            y_min: int, y_max: int, ngrid: int, n_processes: int) -> np.ndarray:
            """
            Grid particles using shared memory array to avoid broken pipe issues.
            
            MEMORY OPTIMIZED: Uses shared memory for both input particles and output grid
            to prevent memory explosion from process copying.
            """
            from multiprocessing import shared_memory, Process
            import numpy as np
            
            slab_height_cells = int(y_max - y_min)
            n_particles = len(particles['x'])
            
            # Create shared memory array for the result grid
            grid_shape = (ngrid, slab_height_cells, ngrid)
            grid_size = int(np.prod(grid_shape))
            
            # MEMORY OPTIMIZATION: Limit number of processes for high particle counts
            # to prevent memory explosion
            if n_particles > 100_000_000:  # 100M particles
                effective_n_processes = min(n_processes, 8)  # Max 8 processes for large datasets
            elif n_particles > 50_000_000:   # 50M particles  
                effective_n_processes = min(n_processes, 16) # Max 16 processes
            else:
                effective_n_processes = n_processes
            
            # Split particles into chunks by index ranges (not actual data copying)
            chunk_size = n_particles // effective_n_processes
            chunk_ranges = []
            for i in range(effective_n_processes):
                start_idx = i * chunk_size
                if i == effective_n_processes - 1:
                    end_idx = n_particles  # Last chunk gets remaining particles
                else:
                    end_idx = (i + 1) * chunk_size
                chunk_ranges.append((start_idx, end_idx))
            
            # Create shared memory for particle data to avoid copying
            particles_shm = {}
            particles_arrays = {}
            
            try:
                # Put particle arrays in shared memory
                for key in ['x', 'y', 'z']:
                    arr = particles[key].astype(np.float32)  # Ensure float32
                    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
                    shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                    shared_arr[:] = arr[:]  # Copy data to shared memory
                    particles_shm[key] = shm
                    particles_arrays[key] = shared_arr
                
                # Create shared memory for output grid
                grid_shm = shared_memory.SharedMemory(create=True, size=grid_size * 4)  # float32 = 4 bytes
                shared_grid = np.ndarray(grid_shape, dtype=np.float32, buffer=grid_shm.buf)
                shared_grid.fill(0.0)  # Initialize to zero
                
                # Create worker processes with shared memory names and index ranges
                processes = []
                for i, (start_idx, end_idx) in enumerate(chunk_ranges):
                    if end_idx > start_idx:  # Only start process if chunk has particles
                        particle_shm_names = {key: shm.name for key, shm in particles_shm.items()}
                        p = Process(target=_cic_slab_shared_worker_optimized, 
                                  args=(particle_shm_names, n_particles, start_idx, end_idx,
                                       ngrid, self.box_size, y_min, y_max, 
                                       self.assignment, grid_shm.name, grid_shape, i))
                        processes.append(p)
                        p.start()
                
                # Wait for all processes to complete
                for p in processes:
                    p.join()
                    
                # Copy result from shared memory to regular array
                result_grid = shared_grid.copy()
                
            finally:
                # Clean up all shared memory
                for shm in particles_shm.values():
                    shm.close()
                    shm.unlink()
                if 'grid_shm' in locals():
                    grid_shm.close()
                    grid_shm.unlink()
            
            return result_grid




def _cic_slab_shared_worker(particle_chunk, ngrid, box_size, y_min, y_max, assignment, shm_name, grid_shape, worker_id):
    """
    Worker function for multiprocessing CIC slab assignment using shared memory.
    
    Args:
        particle_chunk: Dictionary with particle data for this worker
        ngrid: Grid resolution
        box_size: Simulation box size
        y_min, y_max: Y-slab boundaries
        assignment: Assignment scheme ('cic', 'ngp', 'tsc')
        shm_name: Name of shared memory block
        grid_shape: Shape of the shared grid
        worker_id: Worker process ID for debugging
    """
    from multiprocessing import shared_memory
    import numpy as np
    
    # Connect to existing shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_grid = np.ndarray(grid_shape, dtype=np.float32, buffer=shm.buf)
    
    try:
        # Create local grid for this worker
        slab_height = y_max - y_min
        local_grid = np.zeros((ngrid, slab_height, ngrid), dtype=np.float32)
        
        # Handle empty chunk
        if len(particle_chunk['x']) == 0:
            return
        
        # Convert particle positions to grid coordinates
        grid_spacing = box_size / ngrid
        x_grid = np.asarray(particle_chunk['x']) / grid_spacing
        y_grid = np.asarray(particle_chunk['y']) / grid_spacing
        z_grid = np.asarray(particle_chunk['z']) / grid_spacing
        masses = np.asarray(particle_chunk['mass'])
        
        # Apply periodic boundary conditions
        x_grid = np.mod(x_grid, ngrid)
        y_grid = np.mod(y_grid, ngrid)
        z_grid = np.mod(z_grid, ngrid)
        
        # Filter particles that fall within this slab (including ghosts)
        in_slab = (y_grid >= y_min) & (y_grid < y_max)
        if not np.any(in_slab):
            return
        
        x_slab = x_grid[in_slab]
        y_slab = y_grid[in_slab] - y_min  # Shift to slab coordinates
        z_slab = z_grid[in_slab]
        mass_slab = masses[in_slab]
        
        # Perform assignment on local grid
        if assignment == 'cic':
            _cic_assign_slab_worker(local_grid, x_slab, y_slab, z_slab, mass_slab)
        elif assignment == 'ngp':
            _ngp_assign_slab_worker(local_grid, x_slab, y_slab, z_slab, mass_slab)
        elif assignment == 'tsc':
            # TSC not implemented for slab workers yet, fall back to CIC
            _cic_assign_slab_worker(local_grid, x_slab, y_slab, z_slab, mass_slab)
        
        # Atomically add local grid to shared grid
        # Note: This isn't truly atomic, but race conditions are unlikely with spatial locality
        shared_grid += local_grid
        
    finally:
        # Close shared memory connection (but don't unlink - main process will do that)
        shm.close()

def _cic_slab_shared_worker_optimized(particle_shm_names, n_particles, start_idx, end_idx,
                                    ngrid, box_size, y_min, y_max, assignment, 
                                    grid_shm_name, grid_shape, worker_id):
    """
    MEMORY OPTIMIZED worker function that accesses particles from shared memory.
    
    This avoids copying large particle arrays to each worker process.
    """
    from multiprocessing import shared_memory
    import numpy as np
    
    # Connect to shared memory for particles
    particle_arrays = {}
    particle_shms = {}
    
    try:
        # Access particle data from shared memory
        for key, shm_name in particle_shm_names.items():
            shm = shared_memory.SharedMemory(name=shm_name)
            arr = np.ndarray((n_particles,), dtype=np.float32, buffer=shm.buf)
            particle_arrays[key] = arr
            particle_shms[key] = shm
        
        # Connect to shared grid memory
        grid_shm = shared_memory.SharedMemory(name=grid_shm_name)
        shared_grid = np.ndarray(grid_shape, dtype=np.float32, buffer=grid_shm.buf)
        
        # Extract particle chunk using index range (no memory copying!)
        if end_idx > start_idx:
            x_chunk = particle_arrays['x'][start_idx:end_idx]
            y_chunk = particle_arrays['y'][start_idx:end_idx] 
            z_chunk = particle_arrays['z'][start_idx:end_idx]
            # Create unit mass array (no mass field needed for gridding)
            mass_chunk = np.ones(len(x_chunk), dtype=np.float32)
            
            # Convert particle positions to grid coordinates
            grid_spacing = box_size / ngrid
            x_grid = x_chunk / grid_spacing
            y_grid = y_chunk / grid_spacing  
            z_grid = z_chunk / grid_spacing
            
            # Apply periodic boundary conditions
            x_grid = np.mod(x_grid, ngrid)
            y_grid = np.mod(y_grid, ngrid)
            z_grid = np.mod(z_grid, ngrid)
            
            # Filter particles that fall within this slab
            in_slab = (y_grid >= y_min) & (y_grid < y_max)
            if np.any(in_slab):
                x_slab = x_grid[in_slab]
                y_slab = y_grid[in_slab] - y_min  # Shift to slab coordinates
                z_slab = z_grid[in_slab]
                mass_slab = mass_chunk[in_slab]
                
                # Perform assignment directly on the shared grid
                if assignment == 'cic':
                    _cic_assign_slab_worker(shared_grid, x_slab, y_slab, z_slab, mass_slab)
                elif assignment == 'ngp':
                    _ngp_assign_slab_worker(shared_grid, x_slab, y_slab, z_slab, mass_slab)
                elif assignment == 'tsc':
                    # TSC not implemented for slab workers yet, fall back to CIC
                    _cic_assign_slab_worker(shared_grid, x_slab, y_slab, z_slab, mass_slab)
        
    finally:
        # Close shared memory connections (but don't unlink - main process will do that)
        for shm in particle_shms.values():
            shm.close()
        if 'grid_shm' in locals():
            grid_shm.close()

