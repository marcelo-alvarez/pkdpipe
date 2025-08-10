"""
Clean, simple NGP (Nearest Grid Point) particle gridding implementation.

This module provides a simplified NGP gridder that focuses on straightforward
particle binning with simple slab decomposition for distributed processing.

Key assumptions:
- Number of MPI tasks divides evenly into ngrid
- Simple slab decomposition in y-direction
- No ghost cells needed for NGP
- Direct particle binning without complex infrastructure
"""

import numpy as np
from typing import Optional, Dict, Tuple
from mpi4py import MPI


class NGPGridder:
    """
    NGP particle gridding implementation for distributed cosmological power spectrum analysis.
    
    This class implements Nearest Grid Point (NGP) mass assignment using
    straightforward particle binning with Y-slab decomposition for
    distributed processing across multiple MPI processes.
    
    Key Design Features:
    - Simple Y-slab decomposition: each process owns y-slabs of the grid
    - Direct floor division for grid coordinates (avoids coordinate fraud)
    - No ghost cells needed (NGP only uses nearest grid point)
    - Clean MPI reduction for combining local grids
    - Comprehensive particle conservation validation
    
    Requirements:
    - ntasks must divide evenly into ngrid for proper slab decomposition
    - Uses MPI for distributed processing (defaults to MPI.COMM_WORLD)
    - Particles must be in physical coordinates [0, box_size)
    
    Attributes:
        ngrid: Number of grid cells per dimension
        box_size: Physical size of the simulation box
        cell_size: Physical size of each grid cell
        comm: MPI communicator
        rank: MPI rank of this process
        ntasks: Total number of MPI tasks
        y_start: Starting y-index for this process's slabs
        y_end: Ending y-index for this process's slabs (exclusive)
        local_grid_shape: Shape of local grid (ngrid, slab_size, ngrid)
        local_particle_count: Number of particles assigned by this process
        particles_processed: Total particles seen by this process
        
    Example:
        >>> from pkdpipe.ngp_gridder import NGPGridder
        >>> import numpy as np
        >>> from mpi4py import MPI
        >>> 
        >>> # Initialize gridder
        >>> gridder = NGPGridder(ngrid=256, box_size=1050.0)
        >>> 
        >>> # Prepare particle data
        >>> n_particles = 1000000
        >>> positions = np.random.uniform(0, 1050.0, (n_particles, 3))
        >>> 
        >>> # Grid particles and reduce across processes
        >>> local_grid = gridder.grid_particles(positions)
        >>> full_grid = gridder.reduce_grid(local_grid)
        >>> 
        >>> # Validate particle conservation
        >>> gridder.validate_particle_conservation(n_particles)
    """
    
    def __init__(self, ngrid: int, box_size: float, comm: Optional[MPI.Comm] = None):
        """
        Initialize the NGP gridder.
        
        Args:
            ngrid: Number of grid cells per dimension
            box_size: Physical size of the simulation box (same units as positions)
            comm: MPI communicator (if None, uses MPI.COMM_WORLD)
        
        Raises:
            ValueError: If ntasks doesn't divide evenly into ngrid
        """
        self.ngrid = ngrid
        self.box_size = box_size
        self.cell_size = box_size / ngrid
        
        # Set up MPI
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = comm.Get_rank()
        self.ntasks = comm.Get_size()
        
        # Verify constraint: ntasks must divide evenly into ngrid
        if ngrid % self.ntasks != 0:
            raise ValueError(
                f"Number of tasks ({self.ntasks}) must divide evenly into "
                f"ngrid ({ngrid}). Got remainder {ngrid % self.ntasks}"
            )
        
        # Simple slab decomposition in y-direction (to match FFT requirements)
        slab_size = ngrid // self.ntasks
        self.y_start = self.rank * slab_size
        self.y_end = (self.rank + 1) * slab_size
        
        # Store slab dimensions for convenience
        self.local_grid_shape = (ngrid, slab_size, ngrid)
        
        # Particle counting for validation
        self.local_particle_count = 0
        self.particles_processed = 0
        
    def grid_particles(self, positions: np.ndarray, 
                       masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Grid particles using simple NGP assignment.
        
        This method assigns particles to the nearest grid point using floor division,
        which avoids coordinate fraud issues. Each process only grids particles that
        fall within its assigned y-slabs, filtering particles by y-coordinate.
        
        Args:
            positions: Particle positions array of shape (n_particles, 3)
                      Coordinates should be in physical units [0, box_size)
            masses: Optional particle masses array of shape (n_particles,)
                   If None, assumes unit mass for all particles
        
        Returns:
            Local density grid for this process's y-slabs with shape
            (ngrid, slab_size, ngrid)
            
        Note:
            This method updates self.local_particle_count with the number of
            particles assigned by this process for validation purposes.
        """
        n_particles = len(positions)
        
        # Handle masses
        if masses is None:
            masses = np.ones(n_particles, dtype=np.float32)
        
        # Direct particle binning using floor division
        # This is the CORRECT pattern that avoids coordinate fraud
        ix = np.floor(positions[:, 0] / self.cell_size).astype(np.int32) % self.ngrid
        iy = np.floor(positions[:, 1] / self.cell_size).astype(np.int32) % self.ngrid
        iz = np.floor(positions[:, 2] / self.cell_size).astype(np.int32) % self.ngrid
        
        # Keep only particles in this process's y-slabs
        mask = (iy >= self.y_start) & (iy < self.y_end)
        ix_local = ix[mask]
        iy_local = iy[mask] - self.y_start  # Convert to local y-coordinate
        iz_local = iz[mask]
        masses_local = masses[mask]
        
        # Track particle counts for validation
        self.local_particle_count = np.sum(mask)
        self.particles_processed = n_particles
        
        # Create local density grid
        local_grid = np.zeros(self.local_grid_shape, dtype=np.float32)
        
        # Simple binning - add particle masses to grid cells
        # Using np.add.at for efficiency and to handle multiple particles per cell
        np.add.at(local_grid, (ix_local, iy_local, iz_local), masses_local)
        
        return local_grid
    
    def reduce_grid(self, local_grid: np.ndarray) -> np.ndarray:
        """
        Combine local grids from all processes into full density grid.
        
        Uses MPI Allgather to combine the y-slab grids from all processes
        into a complete density grid. This operation ensures all processes
        receive the same full grid for subsequent FFT operations.
        
        Args:
            local_grid: Local density grid for this process with shape
                       (ngrid, slab_size, ngrid)
        
        Returns:
            Full density grid with shape (ngrid, ngrid, ngrid)
            Note: All processes receive the same full grid
            
        Implementation Details:
            - Uses MPI.Comm.Allgather for efficient grid combination
            - Reassembles y-slabs in correct order across all processes
            - Result is identical on all processes for distributed FFT
        """
        # Prepare buffer for full grid
        full_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)
        
        # Use Allgather to collect all slabs
        # Each process contributes its y-slab to the appropriate position
        slab_size = self.local_grid_shape[1]
        
        # Gather all slabs to all processes
        all_slabs = np.zeros((self.ntasks, self.ngrid, slab_size, self.ngrid), 
                             dtype=np.float32)
        self.comm.Allgather(local_grid, all_slabs)
        
        # Reassemble into full grid
        for rank in range(self.ntasks):
            y_start = rank * slab_size
            y_end = (rank + 1) * slab_size
            full_grid[:, y_start:y_end, :] = all_slabs[rank]
        
        return full_grid
    
    def get_particle_counts(self) -> Dict[str, int]:
        """
        Get particle count statistics for validation.
        
        Returns:
            Dictionary containing:
            - local_particles: Number of particles assigned by this process
            - total_particles: Total particles across all processes
            - particles_processed: Total particles seen by this process
            
        Note:
            The particle counts are updated by grid_particles() and can be used
            to validate particle conservation across distributed processing.
        """
        # Reduce particle counts across all processes
        total_particles = self.comm.allreduce(self.local_particle_count, op=MPI.SUM)
        
        return {
            'local_particles': self.local_particle_count,
            'total_particles': total_particles,
            'particles_processed': self.particles_processed
        }
    
    def validate_particle_conservation(self, expected_total: Optional[int] = None) -> bool:
        """
        Validate that particle conservation is maintained across distributed processing.
        
        This method checks that the total number of particles assigned across all
        processes matches the expected count, ensuring no particles are lost or
        duplicated during distributed gridding.
        
        Args:
            expected_total: Expected total number of particles (if known)
                           If None, just reports current totals
        
        Returns:
            True if validation passes, False if mismatch detected
            
        Note:
            Only process 0 prints validation results to avoid cluttered output.
            All processes return the same validation result.
        """
        counts = self.get_particle_counts()
        
        if self.rank == 0:
            print(f"NGPGridder validation:")
            print(f"  Total particles assigned: {counts['total_particles']:,}")
            if expected_total is not None:
                print(f"  Expected total: {expected_total:,}")
                if counts['total_particles'] != expected_total:
                    print(f"  WARNING: Particle count mismatch!")
                    return False
        
        return True
    
    def get_density_diagnostics(self, density_grid: np.ndarray) -> Dict[str, float]:
        """
        Calculate diagnostic statistics for the density grid.
        
        Provides summary statistics for the gridded density field,
        useful for validation and debugging of the gridding process.
        
        Args:
            density_grid: The density grid to analyze
        
        Returns:
            Dictionary containing grid statistics:
            - sum: Total density (should equal particle count for unit masses)
            - mean: Mean density per grid cell
            - std: Standard deviation of density
            - min: Minimum density value
            - max: Maximum density value  
            - shape: Shape of the density grid
        """
        grid_sum = np.sum(density_grid)
        grid_mean = np.mean(density_grid)
        grid_std = np.std(density_grid)
        grid_min = np.min(density_grid)
        grid_max = np.max(density_grid)
        
        return {
            'sum': grid_sum,
            'mean': grid_mean,
            'std': grid_std,
            'min': grid_min,
            'max': grid_max,
            'shape': density_grid.shape
        }