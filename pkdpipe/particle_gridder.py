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


class ParticleGridder:
    """
    Handles particle-to-grid mass assignment with various schemes.
    
    Supports Cloud-in-Cell (CIC) and Nearest Grid Point (NGP) assignment
    with proper periodic boundary condition handling and multi-device
    domain decomposition for GPU parallelization.
    """
    
    def __init__(self, ngrid: int, box_size: float, assignment: str = 'cic'):
        """
        Initialize particle gridder.
        
        Args:
            ngrid: Number of grid cells per dimension
            box_size: Size of simulation box in Mpc/h
            assignment: Mass assignment scheme ('cic' or 'ngp')
            
        Raises:
            ValueError: If assignment scheme is not supported
        """
        self.ngrid = ngrid
        self.box_size = box_size
        self.assignment = assignment.lower()
        self.grid_spacing = box_size / ngrid
        
        if self.assignment not in ['cic', 'ngp']:
            raise ValueError(f"Unknown assignment scheme: {assignment}")
    
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
                return np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float64)
            else:
                return [np.zeros((self.ngrid, self.ngrid//n_devices, self.ngrid), dtype=np.float64) 
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
        Cloud-in-Cell mass assignment using vectorized operations.
        
        Optimized version that eliminates Python loops for better performance
        with large particle counts (e.g., 238M particles).
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
        """Nearest Grid Point mass assignment."""
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
        Grid particles to a spatial slab (for distributed processing).
        
        Args:
            particles: Dictionary with 'x', 'y', 'z', 'mass' arrays
            y_min: Start index of slab in y-direction (including ghost zones)
            y_max: End index of slab in y-direction (including ghost zones)
            ngrid: Full grid resolution
            
        Returns:
            Density grid for the slab with shape (ngrid, slab_height, ngrid)
        """
        # DEBUG: Print entry to gridding function
        print(f"DEBUG: Entering particles_to_slab with {len(particles['x'])} particles", flush=True)
        
        slab_height = y_max - y_min
        slab_grid = np.zeros((ngrid, slab_height, ngrid), dtype=np.float64)
        
        # Convert particle positions to grid coordinates
        x_grid = particles['x'] / self.grid_spacing
        y_grid = particles['y'] / self.grid_spacing
        z_grid = particles['z'] / self.grid_spacing
        masses = particles['mass']
        
        # Apply periodic boundary conditions
        x_grid = x_grid % ngrid
        y_grid = y_grid % ngrid
        z_grid = z_grid % ngrid
        
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