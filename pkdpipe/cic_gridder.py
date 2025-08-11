"""
CIC (Cloud-in-Cell) particle gridding implementation.

This module provides CIC gridding with proper ghost zone handling for
distributed processing. CIC uses trilinear interpolation to distribute
particle mass across 8 neighboring cells.

Key features:
- Y-slab decomposition matching NGP for consistency
- Ghost zone handling for boundary interpolation
- MPI communication for ghost cell exchange
- Particle conservation validation
"""

import numpy as np
from typing import Optional, Dict, Tuple
from mpi4py import MPI
from .base_gridder import BaseGridder


class CICGridder(BaseGridder):
    """
    CIC (Cloud-in-Cell) particle gridding implementation.
    
    This class implements trilinear interpolation mass assignment using
    8-point cloud-in-cell method. It inherits all MPI communication, 
    validation, and reduction logic from BaseGridder.
    
    Key CIC-specific features:
    - 8-point trilinear interpolation
    - Ghost zone handling for boundary interpolation
    - Vectorized assignment operations
    """
    
    def assign_particles_to_grid(self, positions: np.ndarray, 
                               masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Assign particles to grid using CIC (cloud-in-cell) method with ghost zones.
        
        This method assigns particles using trilinear interpolation to 8 neighboring
        cells. Ghost zones are used to handle particles near slab boundaries.
        
        Args:
            positions: Particle positions of shape (n_particles, 3) in physical units
            masses: Optional particle masses of shape (n_particles,). If None, uses unit mass.
        
        Returns:
            Local density grid including ghost zones, shape (ngrid, slab_size+2, ngrid)
        """
        n_particles = len(positions)
        self.particles_processed = n_particles
        
        if masses is None:
            masses = np.ones(n_particles, dtype=np.float64)
        
        # Local grid includes ghost zones (+2 in y for top and bottom ghosts)
        local_grid_shape = (self.ngrid, self.slab_size + 2, self.ngrid)
        local_grid = np.zeros(local_grid_shape, dtype=np.float64)
        
        if n_particles == 0:
            print("CIC: No particles to process", flush=True)
            return local_grid
        
        # Convert positions to grid coordinates
        grid_coords = positions / self.cell_size
        
        # Apply periodic boundary conditions
        grid_coords = np.fmod(grid_coords, self.ngrid)
        grid_coords[grid_coords < 0] += self.ngrid
        
        print(f"CIC: Starting particle loop with {n_particles:,} particles", flush=True)
        
        # VECTORIZED CIC assignment
        x = grid_coords[:, 0]
        y = grid_coords[:, 1] 
        z = grid_coords[:, 2]
        
        # Find lower-left-back corner cells (vectorized)
        ix = np.floor(x).astype(np.int32) % self.ngrid
        iy = np.floor(y).astype(np.int32) % self.ngrid
        iz = np.floor(z).astype(np.int32) % self.ngrid
        
        # Calculate fractional positions (vectorized)
        dx = x - np.floor(x)
        dy = y - np.floor(y)
        dz = z - np.floor(z)
        
        # Filter particles that contribute to our slab (including ghost zones)
        y_affects_slab = ((self.y_start - 1 <= iy) & (iy < self.y_end + 1)) | \
                         ((self.y_start - 1 <= (iy + 1) % self.ngrid) & ((iy + 1) % self.ngrid < self.y_end + 1))
        
        if not np.any(y_affects_slab):
            print("CIC: No particles affect this slab", flush=True)
            return local_grid
            
        # Filter to particles affecting this slab
        relevant_mask = y_affects_slab
        n_relevant = np.sum(relevant_mask)
        print(f"CIC: {n_relevant:,}/{n_particles:,} particles affect this slab", flush=True)
        self.local_particle_count = n_relevant
        
        ix = ix[relevant_mask]
        iy = iy[relevant_mask]  
        iz = iz[relevant_mask]
        dx = dx[relevant_mask]
        dy = dy[relevant_mask]
        dz = dz[relevant_mask]
        masses = masses[relevant_mask]
        
        # Vectorized 8-point interpolation
        print(f"CIC: Starting 8-point interpolation for {n_relevant:,} particles", flush=True)
        
        for di in range(2):
            for dj in range(2):  
                for dk in range(2):
                    print(f"CIC: Processing corner {di},{dj},{dk}", flush=True)
                    
                    # Cell indices with periodic wrapping
                    cell_x = (ix + di) % self.ngrid
                    cell_y = (iy + dj) % self.ngrid
                    cell_z = (iz + dk) % self.ngrid
                    
                    # Filter cells in our extended slab (including ghost zones)
                    in_slab = (self.y_start - 1 <= cell_y) & (cell_y < self.y_end + 1)
                    n_in_slab = np.sum(in_slab)
                    print(f"CIC: Corner {di},{dj},{dk}: {n_in_slab:,} particles in slab", flush=True)
                    
                    if n_in_slab == 0:
                        continue
                    
                    # Convert to local y coordinates (ghost zone at index 0)
                    local_y = cell_y - self.y_start + 1
                    
                    # Calculate weights (vectorized)
                    wx = np.where(di == 0, 1.0 - dx, dx)
                    wy = np.where(dj == 0, 1.0 - dy, dy)
                    wz = np.where(dk == 0, 1.0 - dz, dz)
                    weights = masses * wx * wy * wz
                    
                    # Add contributions only for particles in slab (vectorized)
                    print(f"CIC: Adding contributions for corner {di},{dj},{dk}", flush=True)
                    valid_mask = in_slab
                    if np.any(valid_mask):
                        # Flatten indices for np.add.at
                        flat_indices = (cell_x[valid_mask] * (local_grid_shape[1] * local_grid_shape[2]) +
                                       local_y[valid_mask] * local_grid_shape[2] +
                                       cell_z[valid_mask])
                        # Vectorized accumulation
                        np.add.at(local_grid.ravel(), flat_indices, weights[valid_mask])
                    
                    print(f"CIC: Completed corner {di},{dj},{dk}", flush=True)
        
        print(f"CIC: Completed particle loop, processed {n_particles:,} particles", flush=True)
        return local_grid
    
    def exchange_ghosts(self, local_grid: np.ndarray) -> np.ndarray:
        """
        Exchange ghost cells between neighboring MPI processes.
        
        This method handles the MPI communication to share ghost cell data
        between processes that own adjacent Y-slabs.
        
        Args:
            local_grid: Local grid with ghost zones, shape (ngrid, slab_size+2, ngrid)
        
        Returns:
            Local grid with ghost contributions from neighbors added
        """
        # Prepare send buffers
        # Bottom ghost to send to rank-1 (our y_start data)
        send_bottom = local_grid[:, 1, :].copy()  # Index 1 is y_start
        
        # Top ghost to send to rank+1 (our y_end-1 data)
        send_top = local_grid[:, -2, :].copy()  # Index -2 is y_end-1
        
        # Prepare receive buffers
        recv_bottom = np.zeros((self.ngrid, self.ngrid), dtype=np.float64)
        recv_top = np.zeros((self.ngrid, self.ngrid), dtype=np.float64)
        
        # Determine neighbors with periodic boundary conditions
        prev_rank = (self.rank - 1) % self.ntasks
        next_rank = (self.rank + 1) % self.ntasks
        
        # Exchange ghosts using non-blocking communication
        requests = []
        
        # Send to and receive from previous rank
        if self.ntasks > 1:
            req1 = self.comm.Isend(send_bottom, dest=prev_rank, tag=0)
            req2 = self.comm.Irecv(recv_bottom, source=prev_rank, tag=1)
            requests.extend([req1, req2])
            
            # Send to and receive from next rank
            req3 = self.comm.Isend(send_top, dest=next_rank, tag=1)
            req4 = self.comm.Irecv(recv_top, source=next_rank, tag=0)
            requests.extend([req3, req4])
            
            # Wait for all communications to complete
            MPI.Request.Waitall(requests)
            
            # Add received ghost contributions
            local_grid[:, 0, :] += recv_bottom  # Bottom ghost
            local_grid[:, -1, :] += recv_top    # Top ghost
        
        return local_grid
    
    def grid_particles(self, positions: np.ndarray, 
                       masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Grid particles using CIC assignment.
        
        This is the main interface method that calls the assignment method.
        
        Args:
            positions: Particle positions of shape (n_particles, 3) in physical units
            masses: Optional particle masses of shape (n_particles,). If None, uses unit mass.
        
        Returns:
            Local density grid with ghost zones for this process
        """
        return self.assign_particles_to_grid(positions, masses)
    
    def reduce_grid(self, local_grid: np.ndarray) -> np.ndarray:
        """
        Return this process's Y-slab after processing ghost exchanges.
        
        This method exchanges ghost cells between processes, then extracts
        and returns only the non-ghost slab portion. This implements the 
        slab-based architecture where all processes return valid slabs.
        
        Args:
            local_grid: Local grid with ghost zones, shape (ngrid, slab_size+2, ngrid)
        
        Returns:
            Y-slab for this process with shape (ngrid, slab_size, ngrid)
            (non-ghost portion after ghost exchange)
        """
        # Exchange ghost cells between neighboring processes
        local_grid = self.exchange_ghosts(local_grid)
        
        # Extract and return non-ghost portion (indices 1 to -1 in y)
        grid_slab = local_grid[:, 1:-1, :]  # Shape: (ngrid, slab_size, ngrid)
        
        return grid_slab