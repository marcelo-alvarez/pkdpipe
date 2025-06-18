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
        # Validate input
        if len(particles['x']) == 0:
            raise ValueError("No particles provided")
        
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
        """Cloud-in-Cell mass assignment."""
        # Initialize density grid
        density_grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)
        
        # Get integer grid coordinates (lower-left corner of cell)
        i_coords = np.floor(grid_coords).astype(int)
        
        # Get fractional offsets within cells
        dx = grid_coords - i_coords
        
        # Apply periodic boundary conditions
        i_coords = i_coords % self.ngrid
        
        # CIC weights (trilinear interpolation)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Weights for this corner
                    weight = ((1-i) * (1-dx[:, 0]) + i * dx[:, 0]) * \
                            ((1-j) * (1-dx[:, 1]) + j * dx[:, 1]) * \
                            ((1-k) * (1-dx[:, 2]) + k * dx[:, 2])
                    
                    # Grid indices with periodic wrapping
                    gi = (i_coords[:, 0] + i) % self.ngrid
                    gj = (i_coords[:, 1] + j) % self.ngrid
                    gk = (i_coords[:, 2] + k) % self.ngrid
                    
                    # Add weighted mass to grid
                    np.add.at(density_grid, (gi, gj, gk), masses * weight)
        
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