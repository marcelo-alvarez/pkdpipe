#!/usr/bin/env python3
"""
Global pytest configuration for pkdpipe tests.

This configuration handles both serial and distributed test execution
by detecting MPI environment and coordinating test execution across processes.
"""

import pytest
import os
import sys
import logging
from pathlib import Path

# Add the package root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check debug mode
DEBUG_MODE = os.environ.get('PKDPIPE_DEBUG_MODE', 'false').lower() == 'true'

# Configure logging levels based on debug mode
if not DEBUG_MODE:
    # Suppress verbose logging from JAX and other libraries
    logging.getLogger('jax').setLevel(logging.WARNING)
    logging.getLogger('jax._src').setLevel(logging.WARNING)
    logging.getLogger('jax._src.distributed').setLevel(logging.WARNING)
    logging.getLogger('jax._src.dispatch').setLevel(logging.WARNING) 
    logging.getLogger('jax._src.interpreters').setLevel(logging.WARNING)
    logging.getLogger('jax._src.compiler').setLevel(logging.WARNING)
    logging.getLogger('jax._src.cache_key').setLevel(logging.WARNING)
    logging.getLogger('jax._src.compilation_cache').setLevel(logging.WARNING)
    logging.getLogger('jax._src.clusters').setLevel(logging.WARNING)
    logging.getLogger('jax._src.path').setLevel(logging.WARNING)
    # Also suppress other verbose loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)

def pytest_configure(config):
    """Configure pytest for MPI-aware test execution."""
    
    # Detect if we're running in MPI/SLURM environment
    is_mpi_env = 'SLURM_NTASKS' in os.environ and int(os.environ['SLURM_NTASKS']) > 1
    
    if is_mpi_env:
        ntasks = int(os.environ['SLURM_NTASKS'])
        procid = int(os.environ.get('SLURM_PROCID', '0'))
        
        print(f"Detected MPI environment: {ntasks} processes, current rank: {procid}")
        
        # Add custom markers for distributed tests
        config.addinivalue_line("markers", "distributed: mark test as requiring distributed execution")
        config.addinivalue_line("markers", "serial_only: mark test as requiring serial execution only")
        
        # Store MPI info for use in tests
        config._mpi_ntasks = ntasks
        config._mpi_procid = procid
        config._is_mpi = True
    else:
        config._is_mpi = False
        print("Single process execution")

def pytest_runtest_setup(item):
    """Setup hook to control which tests run on which processes."""
    
    # Only run this logic in MPI environment
    if not hasattr(item.config, '_is_mpi') or not item.config._is_mpi:
        return
        
    procid = item.config._mpi_procid
    ntasks = item.config._mpi_ntasks
    
    # Check test markers
    distributed_marker = item.get_closest_marker("distributed")
    serial_only_marker = item.get_closest_marker("serial_only")
    
    # Logic for test execution
    if serial_only_marker:
        # Serial-only tests: run only on rank 0
        if procid != 0:
            pytest.skip("Serial test (worker process)")
    elif distributed_marker:
        # Distributed tests: run on all ranks
        pass  # Allow execution on all ranks
    else:
        # Default behavior for unmarked tests
        # Check test module to determine if it should be distributed
        test_module = item.module.__name__
        
        if "test_campaign" in test_module:
            # Campaign tests are serial-only
            if procid != 0:
                pytest.skip("Serial test (worker process)")
        elif any(keyword in test_module for keyword in ["power_spectrum", "mpi", "distributed", "jax"]):
            # Distributed-capable tests run on all ranks
            pass
        else:
            # Default: run serial tests only on rank 0
            if procid != 0:
                pytest.skip("Serial test (worker process)")

@pytest.fixture(scope="session")
def mpi_config():
    """Provide MPI configuration to tests."""
    return {
        'is_mpi': hasattr(pytest, 'config') and getattr(pytest.config, '_is_mpi', False),
        'ntasks': getattr(pytest.config, '_mpi_ntasks', 1) if hasattr(pytest, 'config') else 1,
        'procid': getattr(pytest.config, '_mpi_procid', 0) if hasattr(pytest, 'config') else 0,
    }

def pytest_sessionfinish(session, exitstatus):
    """Synchronize all MPI processes at the end of testing."""
    if hasattr(session.config, '_is_mpi') and session.config._is_mpi:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            comm.Barrier()  # Ensure all processes finish together
            
            if session.config._mpi_procid == 0:
                print(f"\n✅ All {session.config._mpi_ntasks} MPI processes completed testing")
                print(f"ℹ️  Skipped tests on worker processes are serial-only tests that run on rank 0 only")
        except ImportError:
            pass