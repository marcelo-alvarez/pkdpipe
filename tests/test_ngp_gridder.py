"""
Unit tests for the NGPGridder class.

Tests the clean, simple NGP implementation including:
- Slab decomposition
- Particle binning
- Particle conservation
- MPI functionality
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pkdpipe.ngp_gridder import NGPGridder


class TestNGPGridder:
    """Test suite for NGPGridder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock MPI communicator
        self.mock_comm = Mock()
        self.mock_comm.Get_rank.return_value = 0
        self.mock_comm.Get_size.return_value = 1
        self.mock_comm.allreduce = Mock(side_effect=lambda x, op: x)
        self.mock_comm.Allgather = Mock()
        
    def test_initialization_single_process(self):
        """Test NGPGridder initialization with single process."""
        ngrid = 64
        box_size = 100.0
        
        gridder = NGPGridder(ngrid, box_size, comm=self.mock_comm)
        
        assert gridder.ngrid == ngrid
        assert gridder.box_size == box_size
        assert gridder.cell_size == pytest.approx(box_size / ngrid)
        assert gridder.rank == 0
        assert gridder.ntasks == 1
        assert gridder.z_start == 0
        assert gridder.z_end == ngrid
        assert gridder.local_grid_shape == (ngrid, ngrid, ngrid)
        
    def test_initialization_multi_process(self):
        """Test NGPGridder initialization with multiple processes."""
        ngrid = 64
        box_size = 100.0
        ntasks = 4
        
        for rank in range(ntasks):
            mock_comm = Mock()
            mock_comm.Get_rank.return_value = rank
            mock_comm.Get_size.return_value = ntasks
            
            gridder = NGPGridder(ngrid, box_size, comm=mock_comm)
            
            slab_size = ngrid // ntasks
            expected_z_start = rank * slab_size
            expected_z_end = (rank + 1) * slab_size
            
            assert gridder.z_start == expected_z_start
            assert gridder.z_end == expected_z_end
            assert gridder.local_grid_shape == (ngrid, ngrid, slab_size)
    
    def test_initialization_invalid_ntasks(self):
        """Test that initialization fails when ntasks doesn't divide evenly into ngrid."""
        ngrid = 64
        box_size = 100.0
        
        # Set up mock comm with ntasks that doesn't divide evenly
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 5  # 64 % 5 = 4, not evenly divisible
        
        with pytest.raises(ValueError, match="must divide evenly"):
            NGPGridder(ngrid, box_size, comm=mock_comm)
    
    def test_single_particle_assignment(self):
        """Test correct grid cell assignment for a single particle."""
        ngrid = 8
        box_size = 8.0
        
        gridder = NGPGridder(ngrid, box_size, comm=self.mock_comm)
        
        # Place particle at (2.5, 3.5, 4.5)
        # Should go to cell (2, 3, 4) with NGP
        positions = np.array([[2.5, 3.5, 4.5]])
        
        local_grid = gridder.grid_particles(positions)
        
        # Check that particle was assigned to correct cell
        assert local_grid[2, 3, 4] == 1.0
        assert np.sum(local_grid) == 1.0
        assert gridder.local_particle_count == 1
    
    def test_particle_at_origin(self):
        """Test particle exactly at origin."""
        ngrid = 8
        box_size = 8.0
        
        gridder = NGPGridder(ngrid, box_size, comm=self.mock_comm)
        
        positions = np.array([[0.0, 0.0, 0.0]])
        local_grid = gridder.grid_particles(positions)
        
        assert local_grid[0, 0, 0] == 1.0
        assert np.sum(local_grid) == 1.0
    
    def test_particle_at_box_edge(self):
        """Test particle at box edge (periodic boundary)."""
        ngrid = 8
        box_size = 8.0
        
        gridder = NGPGridder(ngrid, box_size, comm=self.mock_comm)
        
        # Particle at box edge should wrap to origin
        positions = np.array([[8.0, 8.0, 8.0]])
        local_grid = gridder.grid_particles(positions)
        
        assert local_grid[0, 0, 0] == 1.0
        assert np.sum(local_grid) == 1.0
    
    def test_multiple_particles_same_cell(self):
        """Test multiple particles assigned to the same cell."""
        ngrid = 8
        box_size = 8.0
        
        gridder = NGPGridder(ngrid, box_size, comm=self.mock_comm)
        
        # Three particles all in cell (1, 1, 1)
        positions = np.array([
            [1.1, 1.2, 1.3],
            [1.4, 1.5, 1.6],
            [1.7, 1.8, 1.9]
        ])
        
        local_grid = gridder.grid_particles(positions)
        
        assert local_grid[1, 1, 1] == 3.0
        assert np.sum(local_grid) == 3.0
        assert gridder.local_particle_count == 3
    
    def test_particle_filtering_by_slab(self):
        """Test that only particles in process's z-slabs are kept."""
        ngrid = 8
        box_size = 8.0
        
        # Set up rank 1 of 2 processes (owns z-slabs 4-7)
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.Get_size.return_value = 2
        
        gridder = NGPGridder(ngrid, box_size, comm=mock_comm)
        assert gridder.z_start == 4
        assert gridder.z_end == 8
        
        # Particles at various z positions
        positions = np.array([
            [1.0, 1.0, 1.0],  # z=1, not in slab
            [1.0, 1.0, 3.0],  # z=3, not in slab
            [1.0, 1.0, 4.0],  # z=4, in slab
            [1.0, 1.0, 5.5],  # z=5, in slab
            [1.0, 1.0, 7.9],  # z=7, in slab
        ])
        
        local_grid = gridder.grid_particles(positions)
        
        # Only 3 particles should be in this process's slabs
        assert np.sum(local_grid) == 3.0
        assert gridder.local_particle_count == 3
        assert gridder.particles_processed == 5
    
    def test_particle_conservation(self):
        """Test particle conservation across all cells."""
        ngrid = 16
        box_size = 16.0
        n_particles = 1000
        
        gridder = NGPGridder(ngrid, box_size, comm=self.mock_comm)
        
        # Random particle distribution
        np.random.seed(42)
        positions = np.random.uniform(0, box_size, (n_particles, 3))
        
        local_grid = gridder.grid_particles(positions)
        
        # All particles should be accounted for
        assert np.sum(local_grid) == n_particles
        assert gridder.local_particle_count == n_particles
    
    def test_custom_masses(self):
        """Test gridding with custom particle masses."""
        ngrid = 8
        box_size = 8.0
        
        gridder = NGPGridder(ngrid, box_size, comm=self.mock_comm)
        
        positions = np.array([
            [1.5, 1.5, 1.5],
            [3.5, 3.5, 3.5]
        ])
        masses = np.array([2.0, 3.0])
        
        local_grid = gridder.grid_particles(positions, masses)
        
        assert local_grid[1, 1, 1] == 2.0
        assert local_grid[3, 3, 3] == 3.0
        assert np.sum(local_grid) == 5.0
    
    def test_grid_reduction(self):
        """Test MPI grid reduction (mocked)."""
        ngrid = 8
        box_size = 8.0
        ntasks = 2
        
        # Create gridder for rank 0
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = ntasks
        
        gridder = NGPGridder(ngrid, box_size, comm=mock_comm)
        
        # Create local grid with some data
        local_grid = np.ones((ngrid, ngrid, 4), dtype=np.float32)
        
        # Mock Allgather to simulate gathering from 2 processes
        def mock_allgather(send_buf, recv_buf):
            # Simulate rank 0 and rank 1 each contributing their slab
            recv_buf[0] = send_buf  # Rank 0's data
            recv_buf[1] = send_buf * 2  # Simulate rank 1's data
        
        mock_comm.Allgather = Mock(side_effect=mock_allgather)
        
        full_grid = gridder.reduce_grid(local_grid)
        
        # Check that full grid has correct shape
        assert full_grid.shape == (ngrid, ngrid, ngrid)
        
        # Check that slabs were assembled correctly
        assert np.all(full_grid[:, :, :4] == 1.0)  # Rank 0's slab
        assert np.all(full_grid[:, :, 4:] == 2.0)  # Rank 1's slab
    
    def test_get_particle_counts(self):
        """Test particle count statistics."""
        ngrid = 8
        box_size = 8.0
        
        gridder = NGPGridder(ngrid, box_size, comm=self.mock_comm)
        
        # Grid some particles
        positions = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        gridder.grid_particles(positions)
        
        counts = gridder.get_particle_counts()
        
        assert counts['local_particles'] == 2
        assert counts['total_particles'] == 2  # Mock allreduce returns input
        assert counts['particles_processed'] == 2
    
    def test_validate_particle_conservation_pass(self):
        """Test successful particle conservation validation."""
        ngrid = 8
        box_size = 8.0
        
        gridder = NGPGridder(ngrid, box_size, comm=self.mock_comm)
        
        # Grid exact number of expected particles
        positions = np.random.uniform(0, box_size, (100, 3))
        gridder.grid_particles(positions)
        
        # Should pass validation
        assert gridder.validate_particle_conservation(expected_total=100) == True
    
    def test_validate_particle_conservation_fail(self):
        """Test failed particle conservation validation."""
        ngrid = 8
        box_size = 8.0
        
        gridder = NGPGridder(ngrid, box_size, comm=self.mock_comm)
        
        # Grid different number than expected
        positions = np.random.uniform(0, box_size, (90, 3))
        gridder.grid_particles(positions)
        
        # Should fail validation
        assert gridder.validate_particle_conservation(expected_total=100) == False
    
    def test_density_diagnostics(self):
        """Test density grid diagnostics calculation."""
        ngrid = 8
        box_size = 8.0
        
        gridder = NGPGridder(ngrid, box_size, comm=self.mock_comm)
        
        # Create a test grid
        test_grid = np.zeros((ngrid, ngrid, ngrid))
        test_grid[0, 0, 0] = 5.0
        test_grid[1, 1, 1] = 3.0
        
        diagnostics = gridder.get_density_diagnostics(test_grid)
        
        assert diagnostics['sum'] == 8.0
        assert diagnostics['min'] == 0.0
        assert diagnostics['max'] == 5.0
        assert diagnostics['shape'] == (ngrid, ngrid, ngrid)
        assert 'mean' in diagnostics
        assert 'std' in diagnostics


class TestNGPGridderIntegration:
    """Integration tests for NGPGridder with realistic scenarios."""
    
    def test_uniform_distribution(self):
        """Test gridding a uniform particle distribution."""
        ngrid = 32
        box_size = 100.0
        n_particles = 10000
        
        # Single process for simplicity
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1
        mock_comm.allreduce = Mock(side_effect=lambda x, op: x)
        
        gridder = NGPGridder(ngrid, box_size, comm=mock_comm)
        
        # Create uniform distribution
        np.random.seed(123)
        positions = np.random.uniform(0, box_size, (n_particles, 3))
        
        local_grid = gridder.grid_particles(positions)
        
        # Check conservation
        assert np.sum(local_grid) == n_particles
        
        # Check that distribution is roughly uniform
        expected_per_cell = n_particles / (ngrid ** 3)
        mean_occupancy = np.mean(local_grid)
        
        # Should be close to expected
        assert abs(mean_occupancy - expected_per_cell) / expected_per_cell < 0.1
    
    def test_clustered_distribution(self):
        """Test gridding a clustered particle distribution."""
        ngrid = 16
        box_size = 100.0
        n_particles = 1000
        
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1
        mock_comm.allreduce = Mock(side_effect=lambda x, op: x)
        
        gridder = NGPGridder(ngrid, box_size, comm=mock_comm)
        
        # Create clustered distribution (gaussian around center)
        np.random.seed(456)
        positions = np.random.normal(loc=box_size/2, scale=box_size/10, size=(n_particles, 3))
        
        # Apply periodic boundaries
        positions = positions % box_size
        
        local_grid = gridder.grid_particles(positions)
        
        # Check conservation
        assert np.sum(local_grid) == n_particles
        
        # Check that center cells have more particles (clustering)
        center_idx = ngrid // 2
        center_region = local_grid[center_idx-2:center_idx+2, 
                                  center_idx-2:center_idx+2,
                                  center_idx-2:center_idx+2]
        
        # Center should have significantly more than uniform
        assert np.sum(center_region) > n_particles * 0.3