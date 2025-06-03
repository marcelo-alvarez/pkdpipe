import unittest
import os
import tempfile
import shutil
import sys
import logging
from io import StringIO
from pathlib import Path
from pkdpipe.simulation import Simulation
from pkdpipe.config import DEFAULT_SIMULATION_NAME

class TestPKDPipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up logging for the test class."""
        # Configure logging for tests
        logging.basicConfig(
            level=logging.DEBUG,
            format='[TEST] %(levelname)s: %(message)s',
            force=True  # Override any existing logging configuration
        )
        cls.logger = logging.getLogger(__name__)
        
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.temp_dir, 'runs')
        self.scratch_dir = os.path.join(self.temp_dir, 'scratch')
        os.makedirs(self.run_dir)
        os.makedirs(self.scratch_dir)
        
        # Log test setup
        self.logger.info(f"Test setup - temp_dir: {self.temp_dir}")
        self.logger.info(f"Test setup - run_dir: {self.run_dir}")
        self.logger.info(f"Test setup - scratch_dir: {self.scratch_dir}")
        
        # Set up test parameters matching create-test.sh
        self.test_params = {
            'jobname_template': 'test',  # Matches create-test.sh jobname
            'nodes': 2,                  # Matches create-test.sh nodes
            'nGrid': 1400,              # Matches create-test.sh nGrid
            'dBoxSize': 1050,           # Matches create-test.sh dBoxSize
            'scratch': True,            # Matches create-test.sh scratch usage
            'rundir': self.run_dir,     # Use temp directory
            'scrdir': self.scratch_dir, # Use temp directory
            'simname': DEFAULT_SIMULATION_NAME,
            'sbatch': False,            # Don't actually submit jobs in tests
            'interact': False
        }
        
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_dir):
            self.logger.info(f"Cleaning up temp directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)

    def _suppress_stdout(self):
        """Context manager to suppress stdout during execution."""
        return StringIO()
    
    def _run_with_suppressed_output(self, func):
        """Run a function with suppressed stdout to reduce test noise."""
        old_stdout = sys.stdout
        sys.stdout = self._suppress_stdout()
        try:
            return func()
        finally:
            sys.stdout = old_stdout

    def test_simulation_creation(self):
        """Test full simulation creation process matching create-test.sh functionality."""
        self.logger.info("Starting simulation creation test")
        sim = Simulation(params=self.test_params)
        self._run_with_suppressed_output(sim.create)

        # Check that job directory was created
        job_dir = os.path.join(self.run_dir, 'test')
        self.logger.debug(f"Checking job directory: {job_dir}")
        self.assertTrue(os.path.exists(job_dir), f"Job directory not created: {job_dir}")
        self.logger.info(f"✓ Job directory created: {job_dir}")

        # Check that scratch directory was created
        scratch_job_dir = os.path.join(self.scratch_dir, 'test')
        self.logger.debug(f"Checking scratch directory: {scratch_job_dir}")
        self.assertTrue(os.path.exists(scratch_job_dir), f"Scratch directory not created: {scratch_job_dir}")
        self.logger.info(f"✓ Scratch directory created: {scratch_job_dir}")

        # Check that output symlink exists
        output_link = os.path.join(job_dir, 'output')
        scratch_output_dir = os.path.join(scratch_job_dir, 'output')
        self.logger.debug(f"Checking output symlink: {output_link}")
        self.assertTrue(os.path.islink(output_link), f"Output symlink not created: {output_link}")
        symlink_target = os.readlink(output_link)
        self.assertEqual(symlink_target, scratch_output_dir, 
                        f"Symlink target mismatch. Expected: {scratch_output_dir}, Got: {symlink_target}")
        self.logger.info(f"✓ Output symlink created: {output_link} -> {symlink_target}")

        # Check that output directory exists in scratch
        output_dir = os.path.join(scratch_job_dir, 'output')
        self.logger.debug(f"Checking output directory: {output_dir}")
        self.assertTrue(os.path.exists(output_dir), f"Output directory not created: {output_dir}")
        self.logger.info(f"✓ Output directory created: {output_dir}")

        # Check essential files were created
        expected_files = [
            'test.par',          # Parameter file
            'test.sbatch',       # SLURM script
            'test.transfer',     # Transfer function
            'run.sh'            # Run script
        ]
        for file in expected_files:
            file_path = os.path.join(job_dir, file)
            self.logger.debug(f"Checking file: {file_path}")
            self.assertTrue(os.path.exists(file_path), f"Missing file: {file}")
            self.logger.info(f"✓ File created: {file}")

        # Check parameter file content
        par_file_path = os.path.join(job_dir, 'test.par')
        self.logger.debug(f"Reading parameter file: {par_file_path}")
        with open(par_file_path, 'r') as f:
            par_content = f.read()
            
        # Verify grid and box size parameters
        self.assertIn('nGrid           = 1400', par_content, "nGrid parameter not found in .par file")
        self.assertIn('dBoxSize        = 1050', par_content, "dBoxSize parameter not found in .par file")
        self.logger.info("✓ Parameter file contains correct nGrid and dBoxSize values")
        
        # Verify achOutName has correct path structure
        expected_ach_out_name = os.path.join(scratch_job_dir, 'output', 'test')
        ach_out_line = f'achOutName      = "{expected_ach_out_name}"'
        self.assertIn(ach_out_line, par_content, 
                     f"achOutName not found with expected path. Expected: {ach_out_line}")
        self.logger.info(f"✓ achOutName correctly set to: {expected_ach_out_name}")

        # Check SLURM script content
        sbatch_file_path = os.path.join(job_dir, 'test.sbatch')
        self.logger.debug(f"Reading SLURM script: {sbatch_file_path}")
        with open(sbatch_file_path, 'r') as f:
            sbatch_content = f.read()
        self.assertIn('#SBATCH -N 2', sbatch_content, "Node count not found in SLURM script")
        self.logger.info("✓ SLURM script contains correct node count")
        
        self.logger.info("Simulation creation test completed successfully")

    def test_parameter_validation(self):
        """Test parameter validation and error handling."""
        # Test with invalid nGrid
        invalid_params = self.test_params.copy()
        invalid_params['nGrid'] = -1
        with self.assertRaises(ValueError):
            sim = Simulation(params=invalid_params)

        # Test with invalid dBoxSize
        invalid_params = self.test_params.copy()
        invalid_params['dBoxSize'] = 0
        with self.assertRaises(ValueError):
            sim = Simulation(params=invalid_params)

    def test_directory_structure(self):
        """Test that directory structure is created correctly."""
        sim = Simulation(params=self.test_params)
        self._run_with_suppressed_output(sim.create)

        # Check main directory structure
        job_dir = os.path.join(self.run_dir, 'test')
        scratch_dir = os.path.join(self.scratch_dir, 'test')
        output_dir = os.path.join(scratch_dir, 'output')

        self.assertTrue(os.path.exists(job_dir))
        self.assertTrue(os.path.exists(scratch_dir))
        self.assertTrue(os.path.exists(output_dir))

    def test_file_permissions(self):
        """Test that files are created with correct permissions."""
        sim = Simulation(params=self.test_params)
        self._run_with_suppressed_output(sim.create)

        job_dir = os.path.join(self.run_dir, 'test')
        run_script = os.path.join(job_dir, 'run.sh')

        # Check run.sh is executable
        self.assertTrue(os.access(run_script, os.X_OK))

if __name__ == '__main__':
    unittest.main()