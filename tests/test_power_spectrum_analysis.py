"""
Tests for PowerSpectrumAnalysis high-level API.

This module tests the PowerSpectrumAnalysis class and related configuration
and result management functionality.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from pkdpipe.power_spectrum_analysis import PowerSpectrumAnalysis, PowerSpectrumConfig, PowerSpectrumResults
from pkdpipe.simulation import Simulation


class TestPowerSpectrumConfig:
    """Test PowerSpectrumConfig dataclass."""
    
    def test_config_creation_basic(self):
        """Test basic configuration creation."""
        config = PowerSpectrumConfig()
        
        assert config.ngrid == 256
        assert config.assignment == "cic"
        assert config.dataset == "xp"
        assert config.debug_synthetic is False
        assert config.bbox is None
    
    def test_config_creation_custom(self):
        """Test configuration creation with custom parameters."""
        config = PowerSpectrumConfig(
            ngrid=512,
            assignment="tsc",
            dataset="xvp",
            debug_synthetic=True,
            campaign_dir="/test/campaign",
            variant="test_variant"
        )
        
        assert config.ngrid == 512
        assert config.assignment == "tsc"
        assert config.dataset == "xvp"
        assert config.debug_synthetic is True
        assert config.campaign_dir == "/test/campaign"
        assert config.variant == "test_variant"
    
    def test_config_validation_ngrid(self):
        """Test ngrid validation."""
        with pytest.raises(ValueError, match="ngrid must be positive"):
            PowerSpectrumConfig(ngrid=0)
        
        with pytest.raises(ValueError, match="ngrid must be positive"):
            PowerSpectrumConfig(ngrid=-1)
    
    def test_config_validation_assignment(self):
        """Test assignment validation."""
        with pytest.raises(ValueError, match="assignment must be one of"):
            PowerSpectrumConfig(assignment="invalid")
    
    def test_config_validation_dataset(self):
        """Test dataset validation."""
        with pytest.raises(ValueError, match="dataset must be one of"):
            PowerSpectrumConfig(dataset="invalid")


class TestPowerSpectrumResults:
    """Test PowerSpectrumResults dataclass."""
    
    def test_results_creation(self):
        """Test PowerSpectrumResults creation."""
        k_bins = np.array([0.1, 0.2, 0.3])
        power = np.array([1000.0, 500.0, 250.0])
        n_modes = np.array([10, 20, 30])
        density_stats = {'mean_density': 1.0}
        metadata = {'ngrid': 256}
        
        results = PowerSpectrumResults(
            k_bins=k_bins,
            power_spectrum=power,
            n_modes=n_modes,
            density_stats=density_stats,
            metadata=metadata
        )
        
        assert np.array_equal(results.k_bins, k_bins)
        assert np.array_equal(results.power_spectrum, power)
        assert np.array_equal(results.n_modes, n_modes)
        assert results.density_stats == density_stats
        assert results.metadata == metadata
    
    def test_results_saving(self):
        """Test result saving to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            k_bins = np.array([0.1, 0.2, 0.3])
            power = np.array([1000.0, 500.0, 250.0])
            n_modes = np.array([10, 20, 30])
            density_stats = {'mean_density': 1.0, 'rms_density': 0.1}
            metadata = {'ngrid': 256, 'assignment': 'cic'}
            
            results = PowerSpectrumResults(
                k_bins=k_bins,
                power_spectrum=power,
                n_modes=n_modes,
                density_stats=density_stats,
                metadata=metadata
            )
            
            output_file = Path(tmpdir) / "test_results.txt"
            results.save(str(output_file))
            
            # Check file was created
            assert output_file.exists()
            
            # Check file contents
            content = output_file.read_text()
            assert "# Power Spectrum Analysis Results" in content
            assert "# ngrid: 256" in content
            assert "# assignment: cic" in content
            assert "# mean_density: 1.0" in content
            assert "# k[h/Mpc] P(k)[(Mpc/h)³] N_modes" in content
            assert "0.100000 1.000000e+03 10" in content
    
    def test_results_comparison(self):
        """Test power spectrum comparison."""
        k_bins = np.array([0.1, 0.2, 0.3])
        power1 = np.array([1000.0, 500.0, 250.0])
        power2 = np.array([1010.0, 505.0, 252.5])  # 1% difference
        n_modes = np.array([10, 20, 30])
        
        results1 = PowerSpectrumResults(
            k_bins=k_bins, power_spectrum=power1, n_modes=n_modes,
            density_stats={}, metadata={}
        )
        results2 = PowerSpectrumResults(
            k_bins=k_bins, power_spectrum=power2, n_modes=n_modes,
            density_stats={}, metadata={}
        )
        
        comparison = results1.compare_with(results2)
        
        assert 'max_relative_difference' in comparison
        assert 'mean_relative_difference' in comparison
        assert 'rms_relative_difference' in comparison
        assert comparison['max_relative_difference'] == pytest.approx(0.01, abs=1e-6)
    
    def test_results_comparison_different_kbins(self):
        """Test comparison with different k-binning."""
        k_bins1 = np.array([0.1, 0.2, 0.3])
        k_bins2 = np.array([0.1, 0.2, 0.4])  # Different
        power = np.array([1000.0, 500.0, 250.0])
        n_modes = np.array([10, 20, 30])
        
        results1 = PowerSpectrumResults(
            k_bins=k_bins1, power_spectrum=power, n_modes=n_modes,
            density_stats={}, metadata={}
        )
        results2 = PowerSpectrumResults(
            k_bins=k_bins2, power_spectrum=power, n_modes=n_modes,
            density_stats={}, metadata={}
        )
        
        with pytest.raises(ValueError, match="Cannot compare power spectra with different k-binning"):
            results1.compare_with(results2)


class TestPowerSpectrumAnalysis:
    """Test PowerSpectrumAnalysis class."""
    
    def test_analysis_initialization_basic(self):
        """Test basic PowerSpectrumAnalysis initialization."""
        analysis = PowerSpectrumAnalysis()
        
        assert analysis.config.ngrid == 256
        assert analysis.config.assignment == "cic"
        assert analysis.simulation is None
    
    def test_analysis_initialization_with_config(self):
        """Test initialization with PowerSpectrumConfig."""
        config = PowerSpectrumConfig(ngrid=512, assignment="tsc")
        analysis = PowerSpectrumAnalysis(config=config)
        
        assert analysis.config.ngrid == 512
        assert analysis.config.assignment == "tsc"
    
    def test_analysis_initialization_with_kwargs(self):
        """Test initialization with keyword arguments."""
        analysis = PowerSpectrumAnalysis(
            ngrid=1024,
            assignment="ngp",
            debug_synthetic=True
        )
        
        assert analysis.config.ngrid == 1024
        assert analysis.config.assignment == "ngp"
        assert analysis.config.debug_synthetic is True
    
    def test_analysis_from_simulation(self):
        """Test creating analysis from simulation."""
        # Create mock simulation
        params = {
            'jobname_template': 'test_job',
            'rundir': '/test/run',
            'nodes': 1, 'cpupert': 32, 'gpupern': 4, 'scrdir': '/tmp',
            'simname': 'test_simulation', 'cosmo': 'planck2018', 'scratch': False,
            'nGrid': 256, 'dBoxSize': 1000.0, 'dRedFrom': 49.0, 'iLPT': 2,
            'dRedTo': '0.0', 'nSteps': '100', 'iOutInterval': 10
        }
        simulation = Simulation(params=params)
        
        # Create analysis from simulation
        analysis = PowerSpectrumAnalysis.from_simulation(
            simulation,
            ngrid=512,
            assignment="tsc"
        )
        
        assert analysis.simulation is simulation
        assert analysis.config.ngrid == 512  # Override applied
        assert analysis.config.assignment == "tsc"  # Override applied
    
    def test_compute_bbox_from_simulation(self):
        """Test bounding box computation from simulation."""
        params = {
            'jobname_template': 'test_job',
            'rundir': '/test/run',
            'nodes': 1, 'cpupert': 32, 'gpupern': 4, 'scrdir': '/tmp',
            'simname': 'test_simulation', 'cosmo': 'planck2018', 'scratch': False,
            'nGrid': 256, 'dBoxSize': 2000.0, 'dRedFrom': 49.0, 'iLPT': 2,
            'dRedTo': '0.0', 'nSteps': '100', 'iOutInterval': 10
        }
        simulation = Simulation(params=params)
        
        analysis = PowerSpectrumAnalysis.from_simulation(simulation)
        bbox = analysis._compute_bbox_from_simulation()
        
        expected_bbox = [[0, 2000.0], [0, 2000.0], [0, 2000.0]]
        assert bbox == expected_bbox
    
    def test_compute_bbox_no_simulation(self):
        """Test bbox computation without simulation."""
        analysis = PowerSpectrumAnalysis()
        
        with pytest.raises(ValueError, match="No simulation associated"):
            analysis._compute_bbox_from_simulation()
    
    def test_get_effective_bbox_from_config(self):
        """Test effective bbox when specified in config."""
        custom_bbox = [[-500, 500], [-500, 500], [-500, 500]]
        config = PowerSpectrumConfig(bbox=custom_bbox)
        analysis = PowerSpectrumAnalysis(config=config)
        
        bbox = analysis._get_effective_bbox()
        assert bbox == custom_bbox
    
    def test_get_effective_bbox_from_simulation(self):
        """Test effective bbox computed from simulation."""
        params = {
            'jobname_template': 'test_job',
            'rundir': '/test/run',
            'nodes': 1, 'cpupert': 32, 'gpupern': 4, 'scrdir': '/tmp',
            'simname': 'test_simulation', 'cosmo': 'planck2018', 'scratch': False,
            'nGrid': 256, 'dBoxSize': 1500.0, 'dRedFrom': 49.0, 'iLPT': 2,
            'dRedTo': '0.0', 'nSteps': '100', 'iOutInterval': 10
        }
        simulation = Simulation(params=params)
        
        analysis = PowerSpectrumAnalysis.from_simulation(simulation)
        bbox = analysis._get_effective_bbox()
        
        expected_bbox = [[0, 1500.0], [0, 1500.0], [0, 1500.0]]
        assert bbox == expected_bbox
    
    def test_get_effective_bbox_fallback(self):
        """Test effective bbox fallback."""
        analysis = PowerSpectrumAnalysis()
        bbox = analysis._get_effective_bbox()
        
        expected_bbox = [[-1000, 1000], [-1000, 1000], [-1000, 1000]]
        assert bbox == expected_bbox
    
    @patch('pkdpipe.power_spectrum_analysis.generate_synthetic_particle_data')
    @patch('pkdpipe.power_spectrum_analysis.PowerSpectrumCalculator')
    def test_run_with_synthetic_data(self, mock_calculator_class, mock_generate_data):
        """Test running analysis with synthetic data."""
        # Mock synthetic data generation
        mock_particle_data = {
            'x': np.array([1.0, 2.0, 3.0]),
            'y': np.array([1.0, 2.0, 3.0]),
            'z': np.array([1.0, 2.0, 3.0])
        }
        mock_generate_data.return_value = mock_particle_data
        
        # Mock calculator
        mock_calculator = MagicMock()
        mock_calculator.calculate_power_spectrum.return_value = (
            np.array([0.1, 0.2, 0.3]),  # k_bins
            np.array([1000.0, 500.0, 250.0]),  # power_spectrum
            np.array([10, 20, 30])  # n_modes
        )
        mock_calculator.get_density_diagnostics.return_value = {'mean_density': 1.0}
        mock_calculator_class.return_value = mock_calculator
        
        # Run analysis
        analysis = PowerSpectrumAnalysis(debug_synthetic=True)
        results = analysis.run()
        
        # Verify results
        assert isinstance(results, PowerSpectrumResults)
        assert len(results.k_bins) == 3
        assert len(results.power_spectrum) == 3
        assert len(results.n_modes) == 3
        assert results.metadata['data_type'] == 'synthetic'
        assert results.metadata['n_particles'] == 3
        
        # Verify mocks were called
        mock_generate_data.assert_called_once()
        mock_calculator.calculate_power_spectrum.assert_called_once()
    
    def test_analysis_parameter_validation(self):
        """Test parameter validation in analysis."""
        with pytest.raises(ValueError, match="ngrid must be positive"):
            PowerSpectrumAnalysis(ngrid=0)
        
        with pytest.raises(ValueError, match="assignment must be one of"):
            PowerSpectrumAnalysis(assignment="invalid")
    
    @patch('pkdpipe.power_spectrum_analysis.PowerSpectrumAnalysis._setup_data_reader')
    @patch('pkdpipe.power_spectrum_analysis.PowerSpectrumCalculator')
    def test_run_with_real_data_simulation(self, mock_calculator_class, mock_setup_reader):
        """Test running analysis with real data from simulation."""
        # Mock simulation
        params = {
            'jobname_template': 'test_job',
            'rundir': '/test/run',
            'nodes': 1, 'cpupert': 32, 'gpupern': 4, 'scrdir': '/tmp',
            'simname': 'test_simulation', 'cosmo': 'planck2018', 'scratch': False,
            'nGrid': 256, 'dBoxSize': 1000.0, 'dRedFrom': 49.0, 'iLPT': 2,
            'dRedTo': '0.0', 'nSteps': '100', 'iOutInterval': 10
        }
        simulation = Simulation(params=params)
        
        # Mock data reader - return data in real API format: {"box0": structured_array}  
        mock_data_reader = MagicMock()
        
        # Create structured array matching real API behavior
        mock_structured_array = np.array(
            [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0), (4.0, 4.0, 4.0)],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
        )
        
        # Return in dictionary format as real fetch_data() does
        mock_particle_data = {"box0": mock_structured_array}
        mock_data_reader.fetch_data.return_value = mock_particle_data
        mock_setup_reader.return_value = mock_data_reader
        
        # Mock calculator
        mock_calculator = MagicMock()
        mock_calculator.calculate_power_spectrum.return_value = (
            np.array([0.1, 0.2, 0.3]),
            np.array([1000.0, 500.0, 250.0]),
            np.array([10, 20, 30])
        )
        mock_calculator.get_density_diagnostics.return_value = {'mean_density': 1.0}
        mock_calculator_class.return_value = mock_calculator
        
        # Run analysis
        analysis = PowerSpectrumAnalysis.from_simulation(simulation)
        results = analysis.run()
        
        # Verify results
        assert isinstance(results, PowerSpectrumResults)
        assert results.metadata['data_type'] == 'real_simulation'
        assert results.metadata['n_particles'] == 4
        assert results.metadata['simulation_box_size'] == 1000.0
        assert results.metadata['job_name'] == 'test_job'
        
        # Verify mocks were called
        mock_setup_reader.assert_called_once()
        mock_data_reader.fetch_data.assert_called_once()
        
        # Verify calculator was called with properly formatted particle data
        call_args = mock_calculator.calculate_power_spectrum.call_args[0]
        particles_dict = call_args[0]
        assert 'x' in particles_dict
        assert 'y' in particles_dict  
        assert 'z' in particles_dict
        assert len(particles_dict['x']) == 4