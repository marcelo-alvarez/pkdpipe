"""
Base gridder class with shared infrastructure for NGP and CIC implementations.

This module provides the BaseGridder abstract class that contains all shared
MPI communication, particle redistribution, validation, and reduction logic.
The specific assignment methods (NGP vs CIC) are implemented in derived classes.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from mpi4py import MPI
from abc import ABC, abstractmethod


class BaseGridder(ABC):
    """
    Abstract base class for particle gridding implementations.
    
    This class contains all shared infrastructure for distributed particle
    gridding, including MPI communication, particle counting, validation,
    and grid reduction. Derived classes only need to implement the specific
    particle assignment method and slab-based grid reduction.
    
    Key Shared Features:
    - Y-slab decomposition for distributed processing
    - MPI communication setup and management
    - Particle conservation validation
    - Slab-based grid reduction for memory efficiency
    - Full grid reconstruction when needed for output
    - Consistent return value contracts
    - Particle counting and diagnostics
    
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
        slab_size: Number of y-slabs owned by this process
        local_particle_count: Number of particles assigned by this process
        particles_processed: Total particles seen by this process
    """
    
    def __init__(self, ngrid: int, box_size: float, comm: Optional[MPI.Comm] = None):
        """
        Initialize the base gridder with shared infrastructure.
        
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
        
        # Y-slab decomposition (shared between NGP and CIC)
        self.slab_size = ngrid // self.ntasks
        self.y_start = self.rank * self.slab_size
        self.y_end = (self.rank + 1) * self.slab_size
        
        # Particle counting for validation
        self.local_particle_count = 0
        self.particles_processed = 0
    
    @abstractmethod
    def assign_particles_to_grid(self, positions: np.ndarray, 
                               masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Abstract method for particle assignment to grid.
        
        This is the only method that differs between NGP and CIC implementations.
        Each derived class implements its specific assignment algorithm.
        
        Args:
            positions: Particle positions of shape (n_particles, 3) in physical units
            masses: Optional particle masses of shape (n_particles,). If None, uses unit mass.
        
        Returns:
            Local density grid for this process's assigned region
            
        Note:
            This method must update self.local_particle_count and self.particles_processed
            for proper validation.
        """
        pass
    
    @abstractmethod
    def reduce_grid(self, local_grid: np.ndarray) -> np.ndarray:
        """
        Abstract method for reducing local grid to final slab.
        
        This method should return only this process's Y-slab portion of the
        density grid. This is the core of the slab-based architecture that
        provides memory efficiency and consistency.
        
        Args:
            local_grid: Local density grid from assign_particles_to_grid
        
        Returns:
            Final Y-slab for this process with shape (ngrid, slab_height, ngrid)
            where slab_height = ngrid // ntasks
            
        Note:
            - NGP: May need to slice from gathered full grid to return slab
            - CIC: Should exchange ghosts then return non-ghost slab portion
            - All processes return valid slabs, no None values
        """
        pass
    
    def gather_full_grid(self, grid_slab: np.ndarray) -> Optional[np.ndarray]:
        """
        Gather Y-slabs from all processes to reconstruct full density grid.
        
        This method is used when full grid is needed for output files, statistics,
        or analysis. It should only be called when explicitly needed to maintain
        memory efficiency of slab-based processing.
        
        Args:
            grid_slab: This process's Y-slab with shape (ngrid, slab_size, ngrid)
        
        Returns:
            On rank 0: Full density grid with shape (ngrid, ngrid, ngrid)  
            On other ranks: None
        """
        if self.ntasks == 1:
            # Single process case
            return grid_slab.copy()
        
        if self.rank == 0:
            # Initialize full grid on rank 0
            full_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=grid_slab.dtype)
            
            # Place rank 0's slab
            full_grid[:, self.y_start:self.y_end, :] = grid_slab
            
            # Receive slabs from other ranks
            for source_rank in range(1, self.ntasks):
                slab_start = source_rank * self.slab_size
                slab_end = (source_rank + 1) * self.slab_size
                recv_buffer = np.zeros((self.ngrid, self.slab_size, self.ngrid), dtype=grid_slab.dtype)
                self.comm.Recv(recv_buffer, source=source_rank, tag=200 + source_rank)
                full_grid[:, slab_start:slab_end, :] = recv_buffer
            
            return full_grid
        else:
            # Send slab to rank 0 (ensure contiguous array for MPI)
            if not grid_slab.flags['C_CONTIGUOUS']:
                grid_slab = np.ascontiguousarray(grid_slab)
            self.comm.Send(grid_slab, dest=0, tag=200 + self.rank)
            
            # Non-root ranks return None
            return None
    
    def get_particle_counts(self) -> Dict[str, int]:
        """
        Get particle counting statistics for validation.
        
        Returns:
            Dictionary with local and global particle counts
        """
        # Gather particle counts using MPI reduction
        local_counts = np.array([self.local_particle_count, self.particles_processed], dtype=np.int64)
        global_counts = np.zeros(2, dtype=np.int64)
        self.comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)
        
        if self.rank == 0:
            return {
                'local_assigned': self.local_particle_count,
                'local_processed': self.particles_processed,
                'global_assigned': global_counts[0],
                'global_processed': global_counts[1]
            }
        else:
            return {
                'local_assigned': self.local_particle_count,
                'local_processed': self.particles_processed,
                'global_assigned': 0,
                'global_processed': 0
            }
    
    def validate_particle_conservation(self, expected_particles: int, tolerance: float = 1e-10) -> bool:
        """
        Validate that particle conservation is maintained.
        
        This method checks that the total number of particles assigned across all
        processes matches the expected count, ensuring no particles are lost or
        duplicated during distributed gridding.
        
        Args:
            expected_particles: Expected total number of particles
            tolerance: Relative tolerance for conservation check
        
        Returns:
            True if conservation is validated, False otherwise
        
        Raises:
            RuntimeError: If particle conservation is violated beyond tolerance
        """
        counts = self.get_particle_counts()
        
        # Gather the global expected count (in case we were passed local counts)
        expected_global = np.array([expected_particles], dtype=np.int64)
        total_expected = np.zeros(1, dtype=np.int64)
        self.comm.Reduce(expected_global, total_expected, op=MPI.SUM, root=0)
        
        if self.rank == 0:
            assigned = counts['global_assigned']
            processed = counts['global_processed']
            expected_total = total_expected[0]  # Use the global total
            
            # Check processing counts
            if processed != expected_total * self.ntasks:
                print(f"Warning: Processed {processed} particle instances, "
                      f"expected {expected_total * self.ntasks}")
            
            # Check assignment conservation
            relative_error = abs(assigned - expected_total) / expected_total
            
            if relative_error > tolerance:
                raise RuntimeError(
                    f"Particle conservation violated!\n"
                    f"  Expected: {expected_total:,}\n"
                    f"  Assigned: {assigned:,}\n"
                    f"  Relative error: {relative_error:.2e}\n"
                    f"  Tolerance: {tolerance:.2e}"
                )
            
            print(f"Particle conservation validated: {assigned:,} particles assigned")
            return True
        
        return True
    
    def reduce_grid_gather(self, local_grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Reduce local grids using MPI Gather (NGP-style).
        
        This method uses Allgather to combine y-slab grids from all processes
        into a complete density grid. All processes receive the same full grid.
        
        DEPRECATED: Use reduce_grid() for slab-based processing or 
        gather_full_grid() for full grid reconstruction.
        
        Args:
            local_grid: Local density grid for this process
        
        Returns:
            Full density grid with shape (ngrid, ngrid, ngrid) on all processes
        """
        # Prepare buffer for full grid
        full_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)
        
        # Ensure local_grid is C-contiguous for MPI
        if not local_grid.flags['C_CONTIGUOUS']:
            local_grid = np.ascontiguousarray(local_grid)
        
        # Calculate buffer sizes for Allgather
        local_size = local_grid.size
        total_size = local_size * self.ntasks
        
        # Create receive buffer - flat array that will hold all local grids
        recv_buffer = np.zeros(total_size, dtype=np.float32)
        
        # Flatten local grid for MPI transfer
        local_flat = local_grid.flatten()
        
        # Gather all local grids
        self.comm.Allgather(local_flat, recv_buffer)
        
        # Reshape and reassemble into full grid
        for rank in range(self.ntasks):
            start_idx = rank * local_size
            end_idx = (rank + 1) * local_size
            rank_data = recv_buffer[start_idx:end_idx].reshape(self.ngrid, self.slab_size, self.ngrid)
            
            y_start = rank * self.slab_size
            y_end = (rank + 1) * self.slab_size
            full_grid[:, y_start:y_end, :] = rank_data
        
        return full_grid
    
    def reduce_grid_send(self, local_grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Reduce local grids using MPI Send/Recv (CIC-style).
        
        This method gathers local grids to rank 0 only. Other ranks return None.
        This is used when only rank 0 needs the full grid (e.g., for I/O).
        
        DEPRECATED: Use reduce_grid() for slab-based processing or
        gather_full_grid() for full grid reconstruction.
        
        Args:
            local_grid: Local density grid for this process
        
        Returns:
            On rank 0: Full density grid with shape (ngrid, ngrid, ngrid)
            On other ranks: None
        """
        if self.ntasks == 1:
            # Single process case
            return local_grid.copy()
        
        if self.rank == 0:
            # Initialize full grid on rank 0
            full_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float64)
            
            # Place rank 0's data
            full_grid[:, self.y_start:self.y_end, :] = local_grid
            
            # Receive from other ranks
            for source_rank in range(1, self.ntasks):
                slab_start = source_rank * self.slab_size
                slab_end = (source_rank + 1) * self.slab_size
                recv_buffer = np.zeros((self.ngrid, self.slab_size, self.ngrid), dtype=np.float64)
                self.comm.Recv(recv_buffer, source=source_rank, tag=100 + source_rank)
                full_grid[:, slab_start:slab_end, :] = recv_buffer
            
            return full_grid
        else:
            # Send to rank 0 (ensure contiguous array for MPI)
            if not local_grid.flags['C_CONTIGUOUS']:
                local_grid = np.ascontiguousarray(local_grid)
            self.comm.Send(local_grid, dest=0, tag=100 + self.rank)
            
            # Non-root ranks return None for consistency
            return None
    
    def get_density_diagnostics(self, density_grid: np.ndarray) -> Optional[Dict]:
        """
        Calculate diagnostic statistics for the density grid.
        
        Provides summary statistics for the gridded density field,
        useful for validation and debugging of the gridding process.
        
        Args:
            density_grid: The density grid to analyze (None on non-root for CIC-style)
        
        Returns:
            On rank 0: Dictionary with grid statistics
            On other ranks: None (for CIC-style compatibility)
        """
        if density_grid is None:
            return None
        
        grid_sum = np.sum(density_grid)
        grid_mean = np.mean(density_grid)
        grid_std = np.std(density_grid)
        grid_min = np.min(density_grid)
        grid_max = np.max(density_grid)
        
        return {
            'sum': float(grid_sum),
            'mean': float(grid_mean),
            'std': float(grid_std),
            'min': float(grid_min),
            'max': float(grid_max),
            'shape': density_grid.shape
        }