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


class CICGridder:
    """
    CIC particle gridding implementation for distributed cosmological analysis.
    
    This class implements Cloud-in-Cell (CIC) mass assignment using trilinear
    interpolation with Y-slab decomposition and ghost zone handling for
    distributed processing across multiple MPI processes.
    
    Key Design Features:
    - Y-slab decomposition matching NGP implementation
    - Ghost zones for boundary interpolation (1 cell layer)
    - Trilinear interpolation to 8 neighboring cells
    - MPI ghost cell exchange between neighboring processes
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
        slab_size: Number of y-slabs owned by this process
        local_grid_shape: Shape of local grid with ghosts (ngrid, slab_size+2, ngrid)
        local_particle_count: Number of particles assigned by this process
        particles_processed: Total particles seen by this process
    """
    
    def __init__(self, ngrid: int, box_size: float, comm: Optional[MPI.Comm] = None):
        """
        Initialize the CIC gridder.
        
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
        
        # Y-slab decomposition (matching NGP)
        self.slab_size = ngrid // self.ntasks
        self.y_start = self.rank * self.slab_size
        self.y_end = (self.rank + 1) * self.slab_size
        
        # Local grid includes ghost zones (+2 in y for top and bottom ghosts)
        self.local_grid_shape = (ngrid, self.slab_size + 2, ngrid)
        
        # Particle counting for validation
        self.local_particle_count = 0
        self.particles_processed = 0
        
    def grid_particles(self, positions: np.ndarray, 
                       masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Grid particles using CIC assignment with ghost zones.
        
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
        
        # Initialize local grid with ghost zones
        local_grid = np.zeros(self.local_grid_shape, dtype=np.float64)
        
        # Convert positions to grid coordinates
        grid_coords = positions / self.cell_size
        
        # Apply periodic boundary conditions
        grid_coords = np.fmod(grid_coords, self.ngrid)
        grid_coords[grid_coords < 0] += self.ngrid
        
        # CIC assignment
        print(f"CIC: Starting particle loop with {n_particles:,} particles", flush=True)
        
        if n_particles == 0:
            print("CIC: No particles to process", flush=True)
            return local_grid
        
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
        
        # Filter particles that contribute to our slab
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
                    
                    # Filter cells in our extended slab
                    in_slab = (self.y_start - 1 <= cell_y) & (cell_y < self.y_end + 1)
                    n_in_slab = np.sum(in_slab)
                    print(f"CIC: Corner {di},{dj},{dk}: {n_in_slab:,} particles in slab", flush=True)
                    
                    if n_in_slab == 0:
                        continue
                    
                    # Convert to local y coordinates 
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
                        flat_indices = (cell_x[valid_mask] * (self.local_grid_shape[1] * self.local_grid_shape[2]) +
                                       local_y[valid_mask] * self.local_grid_shape[2] +
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
    
    def reduce_grid(self, local_grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Reduce local grids across all MPI processes to form the full grid.
        
        This method first exchanges ghost cells, then extracts the non-ghost
        portion and performs an MPI reduction to combine all local grids.
        
        Args:
            local_grid: Local grid with ghost zones, shape (ngrid, slab_size+2, ngrid)
        
        Returns:
            On rank 0: Full reduced grid of shape (ngrid, ngrid, ngrid)
            On other ranks: None
        """
        # Exchange ghost cells
        local_grid = self.exchange_ghosts(local_grid)
        
        # Extract non-ghost portion (indices 1 to -1 in y)
        local_data = local_grid[:, 1:-1, :]  # Shape: (ngrid, slab_size, ngrid)
        
        # Prepare full grid on rank 0
        if self.rank == 0:
            full_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float64)
        else:
            full_grid = None
        
        # Gather all slabs to rank 0
        if self.ntasks == 1:
            # Single process case
            full_grid = local_data.copy()
        else:
            # Multi-process case
            if self.rank == 0:
                # Place rank 0's data
                full_grid[:, self.y_start:self.y_end, :] = local_data
                
                # Receive from other ranks
                for source_rank in range(1, self.ntasks):
                    slab_start = source_rank * self.slab_size
                    slab_end = (source_rank + 1) * self.slab_size
                    recv_buffer = np.zeros((self.ngrid, self.slab_size, self.ngrid), dtype=np.float64)
                    self.comm.Recv(recv_buffer, source=source_rank, tag=100 + source_rank)
                    full_grid[:, slab_start:slab_end, :] = recv_buffer
            else:
                # Send to rank 0 (ensure contiguous array for MPI)
                if not local_data.flags['C_CONTIGUOUS']:
                    local_data = np.ascontiguousarray(local_data)
                self.comm.Send(local_data, dest=0, tag=100 + self.rank)
                # Non-root ranks return None for consistency with NGPGridder
                full_grid = None
        
        return full_grid
    
    def get_particle_counts(self) -> Dict[str, int]:
        """
        Get particle counting statistics for validation.
        
        Returns:
            Dictionary with local and global particle counts
        """
        # Gather particle counts
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
        
        For CIC, we expect perfect conservation when all particles are processed.
        
        Args:
            expected_particles: Expected total number of particles
            tolerance: Relative tolerance for conservation check
        
        Returns:
            True if conservation is validated, False otherwise
        
        Raises:
            RuntimeError: If particle conservation is violated
        """
        counts = self.get_particle_counts()
        
        # First gather the global expected count (in case we were passed local counts)
        expected_global = np.array([expected_particles], dtype=np.int64)
        total_expected = np.zeros(1, dtype=np.int64)
        self.comm.Reduce(expected_global, total_expected, op=MPI.SUM, root=0)
        
        if self.rank == 0:
            assigned = counts['global_assigned']
            processed = counts['global_processed']
            expected_particles = total_expected[0]  # Use the global total
            
            # Check that we processed all particles
            if processed != expected_particles * self.ntasks:
                print(f"Warning: Processed {processed} particle instances, "
                      f"expected {expected_particles * self.ntasks}")
            
            # For CIC, particle count should equal input
            # (unlike NGP where repetition factor applies)
            relative_error = abs(assigned - expected_particles) / expected_particles
            
            if relative_error > tolerance:
                raise RuntimeError(
                    f"Particle conservation violated!\n"
                    f"  Expected: {expected_particles:,}\n"
                    f"  Assigned: {assigned:,}\n"
                    f"  Relative error: {relative_error:.2e}\n"
                    f"  Tolerance: {tolerance:.2e}"
                )
            
            return True
        
        return True
    
    def get_density_diagnostics(self) -> Optional[Dict]:
        """
        Get diagnostic information about the density field.
        
        Returns:
            On rank 0: Dictionary with min, max, mean, std of density
            On other ranks: None
        """
        # This would be called after reduce_grid
        # Implementation would be similar to NGPGridder
        return None