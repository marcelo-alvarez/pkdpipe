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
from multiprocessing import Pool, cpu_count


def _cic_worker(args):
    """
    Worker function for multiprocessing CIC assignment.
    
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
        n_particles = len(particles['x'])
        chunk_size = n_particles // n_processes
        
        chunks = []
        for i in range(n_processes):
            start_idx = i * chunk_size
            if i == n_processes - 1:
                # Last chunk gets any remaining particles
                end_idx = n_particles
            else:
                end_idx = (i + 1) * chunk_size
            
            chunk = {
                'x': particles['x'][start_idx:end_idx],
                'y': particles['y'][start_idx:end_idx], 
                'z': particles['z'][start_idx:end_idx],
                'mass': particles['mass'][start_idx:end_idx]
            }
            chunks.append(chunk)
        
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
        """
        Cloud-in-Cell mass assignment using vectorized operations with multiprocessing.
        
        Optimized version that uses multiprocessing to parallelize particle processing
        across available CPU cores for better performance with large particle counts.
        """
        n_particles = len(masses)
        
        # Determine if multiprocessing is beneficial
        use_multiprocessing = n_particles > 50000 and self.n_cpu_cores > 1
        
        if not use_multiprocessing:
            # Use single-threaded version for small particle counts
            return self._cic_assignment_single_thread(grid_coords, masses)
        
        # Prepare particles for multiprocessing
        particles = {
            'x': grid_coords[:, 0] * self.grid_spacing,  # Convert back to physical coordinates
            'y': grid_coords[:, 1] * self.grid_spacing,
            'z': grid_coords[:, 2] * self.grid_spacing,
            'mass': masses
        }
        
        # Determine optimal number of processes
        n_processes = self._get_optimal_n_processes(n_particles)
        
        print(f"CIC assignment: Using {n_processes} processes for {n_particles:,} particles")
        
        # Split particles into chunks
        particle_chunks = self._chunk_particles(particles, n_processes)
        
        # Prepare arguments for worker processes
        worker_args = [(chunk, self.ngrid, self.box_size) for chunk in particle_chunks]
        
        # Process chunks in parallel
        with Pool(processes=n_processes) as pool:
            chunk_grids = pool.map(_cic_worker, worker_args)
        
        # Sum all chunk grids
        density_grid = np.sum(chunk_grids, axis=0).astype(np.float32)
        
        return density_grid
    
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
            'mass': masses
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
            n_particles = len(particles['x'])
            print(f"DEBUG: Entering particles_to_slab with {n_particles:,} particles", flush=True)
            
            slab_height = y_max - y_min
            
            # For small particle counts or when multiprocessing isn't beneficial, use single-threaded
            use_multiprocessing = n_particles > 50000 and self.n_cpu_cores > 1
            
            if not use_multiprocessing:
                return self._particles_to_slab_single_thread(particles, y_min, y_max, ngrid)
            
            # Determine optimal number of processes
            n_processes = self._get_optimal_n_processes(n_particles)
            
            print(f"Slab assignment: Using {n_processes} processes for {n_particles:,} particles")
            
            # Split particles into chunks
            particle_chunks = self._chunk_particles(particles, n_processes)
            
            # Prepare arguments for worker processes
            worker_args = [(chunk, ngrid, self.box_size, y_min, y_max, self.assignment) 
                           for chunk in particle_chunks]
            
            # Process chunks in parallel
            with Pool(processes=n_processes) as pool:
                chunk_grids = pool.map(_cic_slab_worker, worker_args)
            
            # Sum all chunk grids
            slab_grid = np.sum(chunk_grids, axis=0).astype(np.float32)
            
            return slab_grid



    def _particles_to_slab_single_thread(self, particles: Dict[str, np.ndarray], 
                                        y_min: int, y_max: int, ngrid: int) -> np.ndarray:
        """
        Single-threaded slab assignment - used for small particle counts.
        """
        slab_height = y_max - y_min
        slab_grid = np.zeros((ngrid, slab_height, ngrid), dtype=np.float32)
        
        # Convert particle positions to grid coordinates (ensure NumPy arrays)
        x_grid = np.asarray(particles['x']) / self.grid_spacing
        y_grid = np.asarray(particles['y']) / self.grid_spacing
        z_grid = np.asarray(particles['z']) / self.grid_spacing
        masses = np.asarray(particles['mass'])
        
        # Apply periodic boundary conditions (use NumPy operations)
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
        
        # Perform mass assignment on slab
        if self.assignment == 'cic':
            self._cic_assign_slab(slab_grid, x_slab, y_slab, z_slab, mass_slab)
        elif self.assignment == 'ngp':
            self._ngp_assign_slab(slab_grid, x_slab, y_slab, z_slab, mass_slab)
        else:
            raise ValueError(f"Unknown assignment scheme: {self.assignment}")
        
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
        
        print(f"DEBUG: Starting optimized CIC assignment for {len(x)} particles to {ngrid_x}x{slab_height}x{ngrid_z} grid", flush=True)
        
        if len(x) == 0:
            print(f"DEBUG: No particles to assign, completing immediately", flush=True)
            return
        
        print(f"DEBUG: Step 1/5 - Computing grid coordinates and bounds check", flush=True)
        step_start = time.time()
        
        # Get integer grid coordinates (lower left corner) - use int32 for memory efficiency
        ix = np.floor(x).astype(np.int32) % ngrid_x
        iy = np.floor(y).astype(np.int32)
        iz = np.floor(z).astype(np.int32) % ngrid_z
        
        # Check bounds for y (slab dimension) - filter out invalid particles
        valid_mask = (iy >= 0) & (iy < slab_height - 1)
        n_valid = np.sum(valid_mask)
        print(f"DEBUG: Step 1 complete in {time.time() - step_start:.2f}s - {n_valid}/{len(x)} particles valid", flush=True)
        
        if n_valid == 0:
            print(f"DEBUG: No valid particles after bounds check, completing", flush=True)
            return
        
        print(f"DEBUG: Step 2/5 - Filtering particles and computing fractional distances", flush=True)
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
        
        print(f"DEBUG: Step 2 complete in {time.time() - step_start:.2f}s", flush=True)
        
        print(f"DEBUG: Step 3/5 - Optimized CIC mass assignment (8 corners)", flush=True)
        step_start = time.time()
        
        # Highly optimized corner assignment - no intermediate arrays
        # Process all 8 corners with minimal memory allocation
        corner_count = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corner_count += 1
                    print(f"DEBUG: Processing corner {corner_count}/8", flush=True)
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
        
        print(f"DEBUG: Step 3 complete in {time.time() - step_start:.2f}s", flush=True)
        
        total_time = time.time() - start_time
        print(f"DEBUG: Step 4/5 - CIC assignment COMPLETE for {n_valid} particles in {total_time:.2f}s total", flush=True)
        print(f"DEBUG: Performance: {n_valid/total_time:.0f} particles/second", flush=True)



    
    def _ngp_assign_slab(self, grid: np.ndarray, x: np.ndarray, y: np.ndarray, 
                        z: np.ndarray, mass: np.ndarray) -> None:
        """
        NGP assignment for slab geometry using vectorized operations.
        
        Optimized version that eliminates Python loops for better performance.
        """
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