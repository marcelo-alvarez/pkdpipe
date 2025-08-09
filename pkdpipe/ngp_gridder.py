"""
Clean, simple NGP (Nearest Grid Point) particle gridding implementation.

This module provides a simplified NGP gridder that focuses on straightforward
particle binning with simple slab decomposition for distributed processing.

Key assumptions:
- Number of MPI tasks divides evenly into ngrid
- Simple slab decomposition in z-direction
- No ghost cells needed for NGP
- Direct particle binning without complex infrastructure
"""

import numpy as np
from typing import Optional, Dict, Tuple
from mpi4py import MPI


class NGPGridder:
    """
    Clean, simple NGP particle gridding implementation.
    
    This class implements Nearest Grid Point (NGP) mass assignment using
    straightforward particle binning with simple slab decomposition for
    distributed processing.
    
    Key features:
    - Simple slab decomposition (each process owns z-slabs)
    - Direct floor division for grid coordinates
    - No ghost cells (not needed for NGP)
    - Clean MPI reduction for combining local grids
    
    Attributes:
        ngrid: Number of grid cells per dimension
        box_size: Physical size of the simulation box
        cell_size: Physical size of each grid cell
        comm: MPI communicator
        rank: MPI rank of this process
        ntasks: Total number of MPI tasks
        z_start: Starting z-index for this process's slabs
        z_end: Ending z-index for this process's slabs (exclusive)
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
        
        # Simple slab decomposition in z-direction
        slab_size = ngrid // self.ntasks
        self.z_start = self.rank * slab_size
        self.z_end = (self.rank + 1) * slab_size
        
        # Store slab dimensions for convenience
        self.local_grid_shape = (ngrid, ngrid, slab_size)
        
        # Particle counting for validation
        self.local_particle_count = 0
        self.particles_processed = 0
        
    def grid_particles(self, positions: np.ndarray, 
                       masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Grid particles using simple NGP assignment.
        
        Args:
            positions: Particle positions array of shape (n_particles, 3)
            masses: Optional particle masses array of shape (n_particles,)
                   If None, assumes unit mass for all particles
        
        Returns:
            Local density grid for this process's z-slabs with shape
            (ngrid, ngrid, slab_size)
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
        
        # Keep only particles in this process's z-slabs
        mask = (iz >= self.z_start) & (iz < self.z_end)
        ix_local = ix[mask]
        iy_local = iy[mask]
        iz_local = iz[mask] - self.z_start  # Convert to local z-coordinate
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
        
        Uses MPI Allgather to combine the z-slab grids from all processes
        into a complete density grid.
        
        Args:
            local_grid: Local density grid for this process with shape
                       (ngrid, ngrid, slab_size)
        
        Returns:
            Full density grid with shape (ngrid, ngrid, ngrid)
            Note: All processes receive the same full grid
        """
        # Prepare buffer for full grid
        full_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)
        
        # Use Allgather to collect all slabs
        # Each process contributes its z-slab to the appropriate position
        slab_size = self.local_grid_shape[2]
        
        # Gather all slabs to all processes
        all_slabs = np.zeros((self.ntasks, self.ngrid, self.ngrid, slab_size), 
                             dtype=np.float32)
        self.comm.Allgather(local_grid, all_slabs)
        
        # Reassemble into full grid
        for rank in range(self.ntasks):
            z_start = rank * slab_size
            z_end = (rank + 1) * slab_size
            full_grid[:, :, z_start:z_end] = all_slabs[rank]
        
        return full_grid
    
    def get_particle_counts(self) -> Dict[str, int]:
        """
        Get particle count statistics for validation.
        
        Returns:
            Dictionary containing:
            - local_particles: Number of particles assigned by this process
            - total_particles: Total particles across all processes
            - particles_processed: Total particles seen by this process
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
        Validate that particle conservation is maintained.
        
        Args:
            expected_total: Expected total number of particles (if known)
        
        Returns:
            True if validation passes, False otherwise
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
        
        Args:
            density_grid: The density grid to analyze
        
        Returns:
            Dictionary containing grid statistics
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