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
from .base_gridder import BaseGridder


class NGPGridder(BaseGridder):
    """
    NGP (Nearest Grid Point) particle gridding implementation.
    
    This class implements simple nearest grid point mass assignment using
    floor division. It inherits all MPI communication, validation, and 
    reduction logic from BaseGridder.
    
    The only method specific to NGP is the particle assignment algorithm,
    which assigns each particle to its nearest grid point without interpolation.
    """
    
    def assign_particles_to_grid(self, positions: np.ndarray, 
                               masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Assign particles to grid using NGP (nearest grid point) method.
        
        This method assigns particles to the nearest grid point using floor division,
        which avoids coordinate fraud issues. Each process only grids particles that
        fall within its assigned y-slabs.
        
        Args:
            positions: Particle positions array of shape (n_particles, 3)
                      Coordinates should be in physical units [0, box_size)
            masses: Optional particle masses array of shape (n_particles,)
                   If None, assumes unit mass for all particles
        
        Returns:
            Local density grid for this process's y-slabs with shape
            (ngrid, slab_size, ngrid)
        """
        n_particles = len(positions)
        self.particles_processed = n_particles
        
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
        
        # Create local density grid
        local_grid_shape = (self.ngrid, self.slab_size, self.ngrid)
        local_grid = np.zeros(local_grid_shape, dtype=np.float32)
        
        # Simple binning - add particle masses to grid cells
        # Using np.add.at for efficiency and to handle multiple particles per cell
        np.add.at(local_grid, (ix_local, iy_local, iz_local), masses_local)
        
        return local_grid
    
    def grid_particles(self, positions: np.ndarray, 
                       masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Grid particles using NGP assignment.
        
        This is the main interface method that calls the abstract assignment method.
        
        Args:
            positions: Particle positions of shape (n_particles, 3) in physical units
            masses: Optional particle masses of shape (n_particles,). If None, uses unit mass.
        
        Returns:
            Local density grid for this process's y-slabs
        """
        return self.assign_particles_to_grid(positions, masses)
    
    def reduce_grid(self, local_grid: np.ndarray) -> np.ndarray:
        """
        Combine local grids from all processes into full density grid.
        
        NGP uses Allgather so all processes receive the full grid for FFT operations.
        
        Args:
            local_grid: Local density grid for this process
        
        Returns:
            Full density grid with shape (ngrid, ngrid, ngrid) on all processes
        """
        return self.reduce_grid_gather(local_grid)
    
    def get_particle_counts(self) -> Dict[str, int]:
        """
        Get particle count statistics for validation (NGP-compatible interface).
        
        Returns:
            Dictionary containing particle counts with NGP-style naming
        """
        base_counts = super().get_particle_counts()
        
        # Convert to NGP-style naming for backward compatibility
        if self.rank == 0:
            total_particles = base_counts['global_assigned']
        else:
            total_particles = self.comm.allreduce(self.local_particle_count, op=MPI.SUM)
        
        return {
            'local_particles': self.local_particle_count,
            'total_particles': total_particles,
            'particles_processed': self.particles_processed
        }
    
    def validate_particle_conservation(self, expected_total: Optional[int] = None) -> bool:
        """
        Validate particle conservation (NGP-compatible interface).
        
        Args:
            expected_total: Expected total number of particles
        
        Returns:
            True if validation passes, False if mismatch detected
        """
        if expected_total is None:
            # Just report current totals
            counts = self.get_particle_counts()
            if self.rank == 0:
                print(f"NGPGridder validation:")
                print(f"  Total particles assigned: {counts['total_particles']:,}")
            return True
        else:
            # Use base class validation with tolerance
            return super().validate_particle_conservation(expected_total)
    
    def get_density_diagnostics(self, density_grid: np.ndarray) -> Dict[str, float]:
        """
        Calculate diagnostic statistics for the density grid (NGP interface).
        
        Args:
            density_grid: The density grid to analyze
        
        Returns:
            Dictionary containing grid statistics
        """
        base_diagnostics = super().get_density_diagnostics(density_grid)
        return base_diagnostics