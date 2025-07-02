"""
Centralized test configuration for pkdpipe tests.

This module contains all shared test parameters to ensure consistency
across all test files and avoid hardcoded values.
"""

# Core simulation parameters - used by all tests
TEST_CONFIG = {
    # Grid and box parameters
    'ngrid': 64,           # Grid size (64³ cells)
    'box_size': 100.0,     # Box size in Mpc/h
    
    # Particle parameters  
    'n_particles': 50000,  # Total number of particles for tests
    
    # Assignment schemes
    'assignment': 'ngp',   # Default assignment method
    
    # Derived parameters (computed automatically)
    'cell_size': None,     # Will be box_size / ngrid
    'volume': None,        # Will be box_size³
    'particle_density': None,  # Will be n_particles / volume
    'particles_per_cell': None,  # Will be n_particles / ngrid³
}

# Compute derived parameters
TEST_CONFIG['cell_size'] = TEST_CONFIG['box_size'] / TEST_CONFIG['ngrid']
TEST_CONFIG['volume'] = TEST_CONFIG['box_size'] ** 3
TEST_CONFIG['particle_density'] = TEST_CONFIG['n_particles'] / TEST_CONFIG['volume']
TEST_CONFIG['particles_per_cell'] = TEST_CONFIG['n_particles'] / (TEST_CONFIG['ngrid'] ** 3)

# Grid configuration for specific tests
GRID_CONFIG = {
    'ngrid': TEST_CONFIG['ngrid'],
    'box_size': TEST_CONFIG['box_size'],
    'assignment': TEST_CONFIG['assignment']
}

# Power spectrum test configuration
POWER_SPECTRUM_CONFIG = {
    'ngrid': TEST_CONFIG['ngrid'],
    'box_size': TEST_CONFIG['box_size'],
    'n_devices': 1,  # Use 1 device per process in distributed mode
    'assignment': TEST_CONFIG['assignment']
}

# Expected shot noise for random particles: P_shot = V/N
EXPECTED_SHOT_NOISE = TEST_CONFIG['volume'] / TEST_CONFIG['n_particles']

# Theoretical variance for density contrast (white noise): Var(δ) = 1/⟨ρ⟩
THEORETICAL_DELTA_VARIANCE = 1.0 / TEST_CONFIG['particles_per_cell']

# Display configuration summary
def print_test_config():
    """Print test configuration summary."""
    print("="*60)
    print("TEST CONFIGURATION")
    print("="*60)
    print(f"Grid size: {TEST_CONFIG['ngrid']}³ = {TEST_CONFIG['ngrid']**3:,} cells")
    print(f"Box size: {TEST_CONFIG['box_size']:.1f} Mpc/h")
    print(f"Cell size: {TEST_CONFIG['cell_size']:.3f} Mpc/h")
    print(f"Volume: {TEST_CONFIG['volume']:.0f} (Mpc/h)³")
    print(f"Particles: {TEST_CONFIG['n_particles']:,}")
    print(f"Particle density: {TEST_CONFIG['particle_density']:.2e} particles/(Mpc/h)³")
    print(f"Particles per cell: {TEST_CONFIG['particles_per_cell']:.3f}")
    print(f"Assignment: {TEST_CONFIG['assignment'].upper()}")
    print(f"Expected shot noise: {EXPECTED_SHOT_NOISE:.1f} (Mpc/h)³")
    print(f"Theoretical δ variance: {THEORETICAL_DELTA_VARIANCE:.1f}")
    print("="*60)

if __name__ == "__main__":
    print_test_config()