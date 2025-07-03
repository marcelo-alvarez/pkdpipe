"""
Synthetic data generation utilities for pkdpipe.

Functions for generating synthetic particle data for testing and debugging.
"""

import numpy as np
import time
from typing import Dict, Any, Optional


def generate_synthetic_particle_data(process_id: int = 0, 
                                   box_size: float = 1050.0,
                                   n_particles_per_process: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate synthetic random particle data with the same memory footprint as real data.
    
    Args:
        process_id: Process ID for reproducible random seeds
        box_size: Size of the simulation box in Mpc/h
        n_particles_per_process: Number of particles per process. If None, uses realistic default
        
    Returns:
        Dictionary containing synthetic particle data in the same format as real data
    """
    if process_id == 0:
        print(f"\n" + "="*60)
        print("SYNTHETIC PARTICLE DATA GENERATION")
        print("="*60)
        print("⚠️  DEBUG MODE: Using synthetic random particles")
    
    # Match the particle count from real simulation
    # Real simulation has ~715M particles per process, use same count
    if n_particles_per_process is None:
        n_particles_per_process = 715_827_876  # From real data logs
    
    if process_id == 0:
        print(f"Generating {n_particles_per_process:,} synthetic particles per process")
        print(f"Particle positions: random float32 in [0, {box_size}] (box_size)")
    
    start_time = time.time()
    
    # Generate random particle positions in [0, box_size] as float32
    # MEMORY OPTIMIZATION: Create separate arrays directly instead of structured array
    np.random.seed(42 + process_id)  # Reproducible but different per process
    
    # Create separate arrays directly (avoids memory doubling during extraction)
    # CRITICAL FIX: Ensure coordinates are strictly < box_size for validation
    # Use (1.0 - epsilon) to guarantee max value < box_size
    scale_factor = box_size * (1.0 - 1e-6)  # Slightly less than box_size
    x_data = (np.random.uniform(0.0, 1.0, n_particles_per_process) * scale_factor).astype(np.float32)
    y_data = (np.random.uniform(0.0, 1.0, n_particles_per_process) * scale_factor).astype(np.float32)
    z_data = (np.random.uniform(0.0, 1.0, n_particles_per_process) * scale_factor).astype(np.float32)
    
    # Create the particle dictionary format directly (no structured array)
    particles_dict = {
        'x': x_data,
        'y': y_data, 
        'z': z_data
    }
    
    # Create the same data structure as real data loading
    result = {'box0': particles_dict}
    
    # Mock simulation parameters
    sim_params = {
        'dBoxSize': box_size,
        'box_size': box_size
    }
    
    if process_id == 0:
        elapsed_time = time.time() - start_time
        print(f"Synthetic data generation completed in {elapsed_time:.1f} seconds")
        print(f"Box size: {box_size:.1f} Mpc/h")
        print(f"Memory footprint: {n_particles_per_process * 3 * 4 / (1024**3):.2f} GB per process")
    
    return result, sim_params