import unittest
import os
import tempfile
import shutil
from pathlib import Path
from pkdpipe.simulation import Simulation
from pkdpipe.config import DEFAULT_SIMULATION_NAME

class TestPKDPipe(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.temp_dir, 'runs')
        self.scratch_dir = os.path.join(self.temp_dir, 'scratch')
        os.makedirs(self.run_dir)
        os.makedirs(self.scratch_dir)
        
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
            shutil.rmtree(self.temp_dir)

    def test_simulation_creation(self):
        """Test full simulation creation process matching create-test.sh functionality."""
        sim = Simulation(params=self.test_params)
        sim.create()

        # Check that job directory was created
        job_dir = os.path.join(self.run_dir, 'test')
        self.assertTrue(os.path.exists(job_dir), "Job directory not created")

        # Check that scratch directory was created
        scratch_job_dir = os.path.join(self.scratch_dir, 'test')
        self.assertTrue(os.path.exists(scratch_job_dir), "Scratch directory not created")

        # Check that output symlink exists
        output_link = os.path.join(job_dir, 'output')
        self.assertTrue(os.path.islink(output_link), "Output symlink not created")
        self.assertEqual(os.readlink(output_link), scratch_job_dir)

        # Check essential files were created
        expected_files = [
            'test.par',          # Parameter file
            'test.sbatch',       # SLURM script
            'test.transfer',     # Transfer function
            'test.log',          # Log file
            'run.sh'            # Run script
        ]
        for file in expected_files:
            file_path = os.path.join(job_dir, file)
            self.assertTrue(os.path.exists(file_path), f"Missing file: {file}")

        # Check parameter file content
        with open(os.path.join(job_dir, 'test.par'), 'r') as f:
            par_content = f.read()
            self.assertIn('nGrid = 1400', par_content)
            self.assertIn('dBoxSize = 1050', par_content)

        # Check SLURM script content
        with open(os.path.join(job_dir, 'test.sbatch'), 'r') as f:
            sbatch_content = f.read()
            self.assertIn('#SBATCH --nodes=2', sbatch_content)

    def test_parameter_validation(self):
        """Test parameter validation and error handling."""
        # Test with invalid nGrid
        invalid_params = self.test_params.copy()
        invalid_params['nGrid'] = -1
        sim = Simulation(params=invalid_params)
        with self.assertRaises(Exception):
            sim.create()

        # Test with invalid dBoxSize
        invalid_params = self.test_params.copy()
        invalid_params['dBoxSize'] = 0
        sim = Simulation(params=invalid_params)
        with self.assertRaises(Exception):
            sim.create()

    def test_directory_structure(self):
        """Test that directory structure is created correctly."""
        sim = Simulation(params=self.test_params)
        sim.create()

        # Check main directory structure
        job_dir = os.path.join(self.run_dir, 'test')
        scratch_dir = os.path.join(self.scratch_dir, 'test')
        output_dir = os.path.join(scratch_dir, 'output')

        self.assertTrue(os.path.exists(job_dir))
        self.assertTrue(os.path.exists(scratch_dir))
        self.assertTrue(os.path.exists(output_dir))

        # Check output directory structure
        output_job_dir = os.path.join(output_dir, 'test')
        self.assertTrue(os.path.exists(output_job_dir))

    def test_file_permissions(self):
        """Test that files are created with correct permissions."""
        sim = Simulation(params=self.test_params)
        sim.create()

        job_dir = os.path.join(self.run_dir, 'test')
        run_script = os.path.join(job_dir, 'run.sh')

        # Check run.sh is executable
        self.assertTrue(os.access(run_script, os.X_OK))

if __name__ == '__main__':
    unittest.main()