"""
Environment configuration utilities for pkdpipe.

Functions for setting up optimal computational environments for
simulation data processing and analysis.
"""

import os
import warnings
import logging
from typing import Optional


def setup_environment(num_threads: Optional[int] = None,
                     suppress_jax_warnings: bool = True,
                     jax_platforms: str = 'cuda,cpu') -> None:
    """
    Configure NumPy threading and JAX environment for optimal performance.
    
    Args:
        num_threads: Number of threads to use for NumPy operations. 
                    If None, defaults to 32 (optimized for Perlmutter GPUs)
        suppress_jax_warnings: Whether to suppress JAX TPU-related warnings
        jax_platforms: JAX platforms to use, in order of preference
    """
    if num_threads is None:
        num_threads = 32
    
    # Configure NumPy threading FIRST to use all CPU cores for gridding operations
    # This is critical for scaling particle gridding across all cores per GPU
    os.environ.setdefault('OMP_NUM_THREADS', str(num_threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(num_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(num_threads))
    os.environ.setdefault('NUMEXPR_MAX_THREADS', str(num_threads))
    
    # Configure JAX platforms
    os.environ['JAX_PLATFORMS'] = jax_platforms
    
    # Suppress JAX warnings if requested
    if suppress_jax_warnings:
        warnings.filterwarnings("ignore", message=".*libtpu.so.*")
        warnings.filterwarnings("ignore", message=".*Failed to open libtpu.so.*")
        logging.getLogger('jax._src.xla_bridge').setLevel(logging.ERROR)
    
    print(f"Environment configured: {num_threads} threads, JAX platforms: {jax_platforms}")


def get_optimal_thread_count() -> int:
    """
    Get the optimal number of threads for the current system.
    
    Returns:
        Recommended number of threads based on system characteristics
    """
    # Default to 32 threads which is optimal for Perlmutter A100 nodes
    # (32 cores per GPU node)
    return 32