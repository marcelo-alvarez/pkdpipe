"""
Tests for Simulation-Analysis integration functionality.

This module tests the new analysis metadata extraction methods added to the
Simulation class to support PowerSpectrumAnalysis integration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from pkdpipe.simulation import Simulation


class TestSimulationAnalysisIntegration:
    """Test simulation analysis metadata extraction methods."""
    
    def test_get_analysis_metadata_basic(self):
        """Test basic analysis metadata extraction."""
        params = {
            'dBoxSize': 1000.0,
            'nGrid': 256, 
            'rundir': '/test/run',
            'jobname_template': 'test_job',  # Use template to avoid auto-generation
            'nodes': 1,
            'cpupert': 32,
            'gpupern': 4,
            'scrdir': '/tmp',
            'simname': 'test_simulation',
            'cosmo': 'planck2018',
            'scratch': False,
            'dRedFrom': 49.0,
            'iLPT': 2,
            'dRedTo': '0.0',
            'nSteps': '100',
            'iOutInterval': 10,
            'effective_h': 0.7,
            'effective_omegam': 0.3,
            'effective_omegal': 0.7,
            'effective_sigma8': 0.8,
            'effective_ns': 0.96,
            'effective_w0': -1.0,
            'effective_wa': 0.0
        }
        
        sim = Simulation(params=params)
        metadata = sim.get_analysis_metadata()
        
        assert metadata['box_size'] == 1000.0
        assert metadata['ngrid'] == 256
        assert metadata['run_directory'] == '/test/run'
        assert metadata['job_name'] == 'test_job'  # Should use template as-is
        assert metadata['cosmology_params']['h'] == 0.7
        assert metadata['cosmology_params']['omegam'] == 0.3
        assert metadata['cosmology_params']['sigma8'] == 0.8
    
    def test_get_analysis_metadata_missing_params(self):
        """Test metadata extraction with missing parameters."""
        params = {
            'dBoxSize': 500.0,
            'rundir': '/test/run',
            'jobname_template': 'minimal_job',
            'nodes': 1,
            'cpupert': 32,
            'gpupern': 4,
            'scrdir': '/tmp',
            'simname': 'test_simulation',
            'cosmo': 'planck2018',
            'scratch': False,
            'dRedFrom': 49.0,
            'iLPT': 2,
            'dRedTo': '0.0',
            'nSteps': '100',
            'iOutInterval': 10
        }
        
        sim = Simulation(params=params)
        metadata = sim.get_analysis_metadata()
        
        assert metadata['box_size'] == 500.0
        # nGrid will be set from defaults, so check that cosmology params are None
        assert metadata['cosmology_params']['h'] is None
        assert metadata['cosmology_params']['omegam'] is None
    
    def test_get_parameter_file_path_with_paths(self):
        """Test parameter file path retrieval when paths are stored."""
        params = {
            'jobname_template': 'test_job',
            'rundir': '/test/run',
            'nodes': 1,
            'cpupert': 32,
            'gpupern': 4,
            'scrdir': '/tmp',
            'simname': 'test_simulation',
            'cosmo': 'planck2018',
            'scratch': False,
            'nGrid': 256,
            'dBoxSize': 1000.0,
            'dRedFrom': 49.0,
            'iLPT': 2,
            'dRedTo': '0.0',
            'nSteps': '100',
            'iOutInterval': 10
        }
        sim = Simulation(params=params)
        
        # Mock stored paths
        sim._paths = {'parfile': Path('/test/run/test_job/test_job.par')}
        
        result = sim.get_parameter_file_path()
        assert result == '/test/run/test_job/test_job.par'
    
    def test_get_parameter_file_path_fallback(self):
        """Test parameter file path fallback when paths not stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'runs'
            job_dir = run_dir / 'test_job'
            job_dir.mkdir(parents=True)
            
            # Create parameter file
            par_file = job_dir / 'test_job.par'
            par_file.write_text('# Test parameter file')
            
            params = {
                'jobname_template': 'test_job',
                'rundir': str(run_dir),
                'nodes': 1,
                'cpupert': 32,
                'gpupern': 4,
                'scrdir': '/tmp',
                'simname': 'test_simulation',
                'cosmo': 'planck2018',
                'scratch': False,
                'nGrid': 256,
                'dBoxSize': 1000.0,
                'dRedFrom': 49.0,
                'iLPT': 2,
                'dRedTo': '0.0',
                'nSteps': '100',
                'iOutInterval': 10
            }
            sim = Simulation(params=params)
            
            result = sim.get_parameter_file_path()
            assert result == str(par_file)
    
    def test_get_parameter_file_path_not_found(self):
        """Test parameter file path when file doesn't exist."""
        params = {
            'jobname_template': 'nonexistent_job',
            'rundir': '/nonexistent/path',
            'nodes': 1,
            'cpupert': 32,
            'gpupern': 4,
            'scrdir': '/tmp',
            'simname': 'test_simulation',
            'cosmo': 'planck2018',
            'scratch': False,
            'nGrid': 256,
            'dBoxSize': 1000.0,
            'dRedFrom': 49.0,
            'iLPT': 2,
            'dRedTo': '0.0',
            'nSteps': '100',
            'iOutInterval': 10
        }
        sim = Simulation(params=params)
        
        result = sim.get_parameter_file_path()
        assert result is None
    
    def test_get_output_directory_with_paths(self):
        """Test output directory retrieval when paths are stored."""
        params = {
            'jobname_template': 'test_job',
            'rundir': '/test/run',
            'nodes': 1,
            'cpupert': 32,
            'gpupern': 4,
            'scrdir': '/tmp',
            'simname': 'test_simulation',
            'cosmo': 'planck2018',
            'scratch': False,
            'nGrid': 256,
            'dBoxSize': 1000.0,
            'dRedFrom': 49.0,
            'iLPT': 2,
            'dRedTo': '0.0',
            'nSteps': '100',
            'iOutInterval': 10
        }
        sim = Simulation(params=params)
        
        # Mock stored paths
        sim._paths = {'ach_out_name_effective': '/test/scratch/test_job'}
        
        result = sim.get_output_directory()
        assert result == '/test/scratch/test_job/output/test_job'
    
    def test_get_output_directory_fallback_scratch(self):
        """Test output directory fallback with scratch enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scratch_dir = Path(tmpdir) / 'scratch'
            job_dir = scratch_dir / 'test_job'
            output_dir = job_dir / 'output' / 'test_job'
            output_dir.mkdir(parents=True)
            
            params = {
                'jobname_template': 'test_job',
                'scrdir': str(scratch_dir),
                'scratch': True,
                'rundir': '/tmp/run',
                'nodes': 1,
                'cpupert': 32,
                'gpupern': 4,
                'simname': 'test_simulation',
                'cosmo': 'planck2018',
                'nGrid': 256,
                'dBoxSize': 1000.0,
                'dRedFrom': 49.0,
                'iLPT': 2,
                'dRedTo': '0.0',
                'nSteps': '100',
                'iOutInterval': 10
            }
            sim = Simulation(params=params)
            
            result = sim.get_output_directory()
            assert result == str(output_dir)
    
    def test_get_output_directory_fallback_no_scratch(self):
        """Test output directory fallback without scratch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'runs'
            job_dir = run_dir / 'test_job'
            output_dir = job_dir / 'output' / 'test_job'
            output_dir.mkdir(parents=True)
            
            params = {
                'jobname_template': 'test_job',
                'rundir': str(run_dir),
                'scratch': False,
                'nodes': 1,
                'cpupert': 32,
                'gpupern': 4,
                'scrdir': '/tmp',
                'simname': 'test_simulation',
                'cosmo': 'planck2018',
                'nGrid': 256,
                'dBoxSize': 1000.0,
                'dRedFrom': 49.0,
                'iLPT': 2,
                'dRedTo': '0.0',
                'nSteps': '100',
                'iOutInterval': 10
            }
            sim = Simulation(params=params)
            
            result = sim.get_output_directory()
            assert result == str(output_dir)
    
    def test_find_final_snapshot_success(self):
        """Test finding final snapshot file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'output'
            output_dir.mkdir()
            
            # Create snapshot files with different timestamps
            older_file = output_dir / 'snapshot_001.tps'
            newer_file = output_dir / 'snapshot_002.tps'
            oldest_file = output_dir / 'snapshot_000.lcp'
            
            older_file.write_text('old snapshot')
            oldest_file.write_text('oldest snapshot')
            newer_file.write_text('new snapshot')
            
            # Ensure different modification times
            import time
            import os
            os.utime(oldest_file, (1000, 1000))
            os.utime(older_file, (2000, 2000))
            os.utime(newer_file, (3000, 3000))
            
            params = {
                'jobname_template': 'test_job',
                'rundir': '/test/run',
                'nodes': 1,
                'cpupert': 32,
                'gpupern': 4,
                'scrdir': '/tmp',
                'simname': 'test_simulation',
                'cosmo': 'planck2018',
                'scratch': False,
                'nGrid': 256,
                'dBoxSize': 1000.0,
                'dRedFrom': 49.0,
                'iLPT': 2,
                'dRedTo': '0.0',
                'nSteps': '100',
                'iOutInterval': 10
            }
            sim = Simulation(params=params)
            
            # Mock get_output_directory to return our test directory
            with patch.object(sim, 'get_output_directory', return_value=str(output_dir)):
                result = sim.find_final_snapshot()
                assert result == newer_file
    
    def test_find_final_snapshot_no_files(self):
        """Test finding final snapshot when no files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'output'
            output_dir.mkdir()
            
            params = {
                'jobname_template': 'test_job',
                'rundir': '/test/run',
                'nodes': 1,
                'cpupert': 32,
                'gpupern': 4,
                'scrdir': '/tmp',
                'simname': 'test_simulation',
                'cosmo': 'planck2018',
                'scratch': False,
                'nGrid': 256,
                'dBoxSize': 1000.0,
                'dRedFrom': 49.0,
                'iLPT': 2,
                'dRedTo': '0.0',
                'nSteps': '100',
                'iOutInterval': 10
            }
            sim = Simulation(params=params)
            
            with patch.object(sim, 'get_output_directory', return_value=str(output_dir)):
                result = sim.find_final_snapshot()
                assert result is None
    
    def test_find_final_snapshot_no_output_dir(self):
        """Test finding final snapshot when output directory doesn't exist."""
        params = {
            'jobname_template': 'test_job',
            'rundir': '/test/run',
            'nodes': 1,
            'cpupert': 32,
            'gpupern': 4,
            'scrdir': '/tmp',
            'simname': 'test_simulation',
            'cosmo': 'planck2018',
            'scratch': False,
            'nGrid': 256,
            'dBoxSize': 1000.0,
            'dRedFrom': 49.0,
            'iLPT': 2,
            'dRedTo': '0.0',
            'nSteps': '100',
            'iOutInterval': 10
        }
        sim = Simulation(params=params)
        
        with patch.object(sim, 'get_output_directory', return_value=None):
            result = sim.find_final_snapshot()
            assert result is None


class TestSimulationLoadFromRunDirectory:
    """Test loading simulation from existing run directory."""
    
    def test_load_from_run_directory_success(self):
        """Test successful loading from run directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'test_run'
            run_dir.mkdir()
            
            # Create parameter file
            par_file = run_dir / 'test_simulation.par'
            par_content = """
# PKDGrav3 parameter file
nGrid 512
dBoxSize 2000.0
dRedFrom 49.0
iLPT 2
"""
            par_file.write_text(par_content)
            
            # Load simulation
            sim = Simulation.load_from_run_directory(str(run_dir))
            
            assert sim.params['jobname_actual'] == 'test_simulation'
            assert sim.params['nGrid'] == 512
            assert sim.params['dBoxSize'] == 2000.0
            assert sim.params['dRedFrom'] == 49.0
            assert sim.params['iLPT'] == 2
            assert sim.params['rundir'] == str(run_dir.parent)
    
    def test_load_from_run_directory_no_par_file(self):
        """Test loading when no parameter file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'empty_run'
            run_dir.mkdir()
            
            with pytest.raises(FileNotFoundError, match="No parameter file found"):
                Simulation.load_from_run_directory(str(run_dir))
    
    def test_load_from_run_directory_nonexistent(self):
        """Test loading from nonexistent directory."""
        with pytest.raises(ValueError, match="Run directory does not exist"):
            Simulation.load_from_run_directory('/nonexistent/path')
    
    def test_load_from_run_directory_parse_error(self):
        """Test loading with malformed parameter file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'bad_run'
            run_dir.mkdir()
            
            # Create malformed parameter file
            par_file = run_dir / 'bad_simulation.par'
            par_file.write_text('This is not a valid parameter file!')
            
            # Should still work but use defaults
            sim = Simulation.load_from_run_directory(str(run_dir))
            
            assert sim.params['jobname_actual'] == 'bad_simulation'
            # Check that defaults were applied, but values depend on preset
            assert isinstance(sim.params['nGrid'], int)
            assert isinstance(sim.params['dBoxSize'], (int, float))
    
    def test_load_from_run_directory_sets_paths(self):
        """Test that loading sets internal paths correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'test_run'
            run_dir.mkdir()
            
            par_file = run_dir / 'test_sim.par'
            par_file.write_text('# Simple par file')
            
            sim = Simulation.load_from_run_directory(str(run_dir))
            
            assert hasattr(sim, '_paths')
            assert sim._paths['parfile'] == par_file
            assert sim._paths['ach_out_name_effective'] == str(run_dir)