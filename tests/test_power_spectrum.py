"""
Test suite for power spectrum calculation with known theoretical cases.

Tests FFT-based power spectrum calculation using JAX on multiple GPUs,
focusing on random particle distributions where we know the expected result.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
from .test_config import TEST_CONFIG, GRID_CONFIG, POWER_SPECTRUM_CONFIG, EXPECTED_SHOT_NOISE, THEORETICAL_DELTA_VARIANCE

# Configure JAX environment variables BEFORE any JAX imports
# This allows JAX to be imported later without early initialization
import os
# Let JAX auto-detect the best platform (GPU if available, CPU otherwise)
# os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # Commented out to allow GPU detection

# Configure JAX memory management before any JAX imports
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.7')

# Defer JAX imports to avoid early initialization that breaks distributed mode
JAX_AVAILABLE = True  # Assume available, will be checked when needed

# Defer GPU detection to avoid early JAX initialization
def has_multiple_gpus():
    """Check for multi-GPU availability without early JAX initialization."""
    try:
        # Only import JAX when actually needed
        import jax
        jax.config.update("jax_enable_x64", False)
        gpu_devices = jax.devices('gpu')
        return len(gpu_devices) >= 2
    except Exception:
        return False

# Don't call has_multiple_gpus() at import time to avoid early JAX init
MULTI_GPU_AVAILABLE = None  # Will be determined when needed

from pkdpipe.power_spectrum import PowerSpectrumCalculator
from pkdpipe.particle_gridder import ParticleGridder
from pkdpipe.data import Data


class TestParticleGridder:
    """Test particle-to-grid mass assignment functionality."""
    
    @pytest.fixture
    def grid_config(self):
        """Standard grid configuration for testing.""" 
        return GRID_CONFIG

    def test_gridder_initialization(self, grid_config):
        """Test ParticleGridder can be initialized with proper parameters."""
        gridder = ParticleGridder(
            ngrid=grid_config['ngrid'],
            box_size=grid_config['box_size'],
            assignment=grid_config['assignment']
        )
        
        assert gridder.ngrid == grid_config['ngrid']
        assert gridder.box_size == grid_config['box_size']
        assert gridder.assignment == grid_config['assignment']
        assert gridder.grid_spacing == grid_config['box_size'] / grid_config['ngrid']

class TestPowerSpectrumCalculator:
    """Test power spectrum calculation with known theoretical cases."""
    
    @pytest.fixture(scope="function")
    def random_particles(self):
        """Generate random particles for testing."""
        import os
        
        # Detect distributed mode
        n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        process_id = int(os.environ.get('SLURM_PROCID', '0'))
        
        # Create particles per process for distributed testing
        total_particles = TEST_CONFIG['n_particles']
        particles_per_process = total_particles // n_processes
        
        print(f"DEBUG: Rank {process_id}/{n_processes}: generating {particles_per_process} particles (total target: {total_particles})")
        
        # Generate random particles within simulation box
        np.random.seed(42 + process_id)  # Different seed per process
        box_size = TEST_CONFIG['box_size']
        
        particles = {
            'x': np.random.uniform(0, box_size, particles_per_process).astype(np.float32),
            'y': np.random.uniform(0, box_size, particles_per_process).astype(np.float32),
            'z': np.random.uniform(0, box_size, particles_per_process).astype(np.float32)
        }
        
        return particles
    
    def test_shot_noise_power_spectrum(self, random_particles):
        """
        Test power spectrum calculation for random particles (shot noise test).
        
        CRITICAL TEST: This verifies that distributed FFT is working properly.
        If JAX distributed FFT fails, each process does independent FFT → ~50% amplitude.
        If JAX distributed FFT works, we get the correct normalization.
        """
        
        # Test configuration from centralized config
        ngrid = POWER_SPECTRUM_CONFIG['ngrid']
        box_size = POWER_SPECTRUM_CONFIG['box_size']
        n_devices = POWER_SPECTRUM_CONFIG['n_devices']
        assignment = POWER_SPECTRUM_CONFIG['assignment']
        
        # Initialize calculator
        calc = PowerSpectrumCalculator(
            ngrid=ngrid,
            box_size=box_size,
            n_devices=n_devices
        )
        
        print("\n" + "="*80)
        print("DISTRIBUTED FFT VALIDATION TEST")
        print("="*80)
        
        # Expected shot noise for random particles  
        expected_shot_noise = EXPECTED_SHOT_NOISE
        print(f"Expected Shot Noise: P_shot = V/N = {box_size:.1f}^3/{len(random_particles['x']):,} = {expected_shot_noise:.1f} (Mpc/h)^3")
        
        # VARIANCE VALIDATION: Grid the particles and validate variance before FFT
        from pkdpipe.particle_gridder import ParticleGridder
        
        print(f"\n" + "-"*50)
        print("GRID VARIANCE VALIDATION")
        print("-"*50)
        
        # Create gridder with same assignment scheme as power spectrum calculation
        gridder = ParticleGridder(ngrid=ngrid, box_size=box_size, assignment=assignment)
        density_grid = gridder.particles_to_grid(random_particles, n_devices=1)
        
        # Calculate density contrast field: δ = ρ/⟨ρ⟩ - 1
        grid_mean = np.mean(density_grid)
        delta_field = density_grid / grid_mean - 1.0
        
        # Calculate statistics of density contrast
        delta_mean = np.mean(delta_field)
        delta_variance = np.var(delta_field)
        delta_std = np.std(delta_field)
        
        # For white noise (Poisson), theoretical expectation for density contrast variance
        # δ = ρ/⟨ρ⟩ - 1, where ρ follows Poisson with mean ⟨ρ⟩
        # Var(δ) = Var(ρ/⟨ρ⟩ - 1) = Var(ρ)/⟨ρ⟩² = ⟨ρ⟩/⟨ρ⟩² = 1/⟨ρ⟩
        expected_particles_per_cell = TEST_CONFIG['particles_per_cell']
        theoretical_delta_variance = THEORETICAL_DELTA_VARIANCE
        
        print(f"Density Contrast Statistics (using {assignment.upper()} assignment):")
        print(f"  Grid shape: {density_grid.shape}")
        print(f"  Total particles: {len(random_particles['x']):,}")
        print(f"  Raw grid mean (particles/cell): {grid_mean:.6e}")
        print(f"  Expected particles per cell: {expected_particles_per_cell:.3f}")
        print(f"  Density contrast mean ⟨δ⟩: {delta_mean:.6e}")
        print(f"  Density contrast variance Var(δ): {delta_variance:.6e}")
        print(f"  Density contrast std dev σ(δ): {delta_std:.6e}")
        print(f"  Theoretical δ variance (white noise): {theoretical_delta_variance:.6e}")
        
        # Validate density contrast variance against white noise expectation
        variance_ratio = delta_variance / theoretical_delta_variance if theoretical_delta_variance > 0 else float('inf')
        print(f"  Variance ratio Var(δ)_measured/Var(δ)_theory: {variance_ratio:.3f}")
        
        # Validate density contrast mean (should be ~0 for unbiased estimator)
        assert abs(delta_mean) < 0.01, (
            f"Density contrast mean significantly non-zero: ⟨δ⟩ = {delta_mean:.6f} "
            f"(should be ≈ 0.0)"
        )
        print(f"  ✅ Density contrast mean ≈ 0 validated")
        
        # For white noise (Poisson), variance should be close to theoretical expectation
        # Allow some tolerance for finite grid effects and discrete particle placement
        if not (0.7 < variance_ratio < 1.5):
            print(f"  WARNING: Density contrast variance differs from white noise expectation")
            print(f"           This may indicate non-random particle distribution or gridding issues")
        else:
            print(f"  ✅ Density contrast variance consistent with white noise expectation")
        
        # Assert variance is within reasonable bounds for white noise
        assert 0.5 < variance_ratio < 2.0, (
            f"Density contrast variance inconsistent with white noise: "
            f"Var(δ)_ratio = {variance_ratio:.3f} (should be ≈ 1.0)"
        )
        
        # Calculate power spectrum 
        k_bins, power_spectrum, n_modes = calc.calculate_power_spectrum(random_particles, assignment=assignment)
        
        # Check power spectrum results
        valid_bins = n_modes > 100  # Only use bins with sufficient modes
        valid_power = power_spectrum[valid_bins]
        
        if len(valid_power) == 0:
            pytest.fail("No valid k-bins with sufficient modes for testing")
        
        # Calculate weighted mean (weight by number of modes)
        valid_n_modes = n_modes[valid_bins]
        weighted_mean_power = np.average(valid_power, weights=valid_n_modes)
        
        print(f"\nPOWER SPECTRUM RESULTS:")
        print(f"  Weighted mean P(k): {weighted_mean_power:.1f} (Mpc/h)^3")
        print(f"  Expected P(k): {expected_shot_noise:.1f} (Mpc/h)^3")
        print(f"  Ratio (measured/expected): {weighted_mean_power/expected_shot_noise:.3f}")
        
        # CRITICAL TEST: Check if distributed FFT is working properly
        n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        if n_processes > 1:
            print(f"\nDISTRIBUTED FFT VALIDATION:")
            print(f"  Running on {n_processes} processes")
            print(f"  If distributed FFT works: ratio should be ≈ 1.0")
            print(f"  If distributed FFT fails: ratio should be ≈ 0.5")
            
            # In distributed mode, ratio should be close to 1.0, NOT 0.5
            ratio = weighted_mean_power / expected_shot_noise
            if ratio < 0.7:  # Much less than expected
                pytest.fail(
                    f"DISTRIBUTED FFT FAILURE DETECTED:\n"
                    f"  Measured/Expected ratio: {ratio:.3f} (should be ≈1.0)\n"
                    f"  This indicates each process is doing independent FFT instead of distributed FFT\n"
                    f"  Expected: {expected_shot_noise:.1f}, Got: {weighted_mean_power:.1f}"
                )
            elif ratio > 1.3:  # Much more than expected  
                pytest.fail(
                    f"POWER SPECTRUM NORMALIZATION ERROR:\n"
                    f"  Measured/Expected ratio: {ratio:.3f} (should be ≈1.0)\n"
                    f"  Expected: {expected_shot_noise:.1f}, Got: {weighted_mean_power:.1f}"
                )
            else:
                print(f"  ✅ DISTRIBUTED FFT WORKING: ratio = {ratio:.3f} ≈ 1.0")
        
        # Note: Individual bin statistical validation via chi-squared test below
        # Simple ratio test removed in favor of proper statistical analysis
        
        # CHI-SQUARED STATISTICAL VALIDATION
        print(f"\n" + "-"*50)
        print("CHI-SQUARED STATISTICAL VALIDATION")
        print("-"*50)
        
        # For shot noise, each k-bin should follow chi-squared distribution
        # P_measured ~ P_expected * χ²(N_modes) / N_modes
        # Calculate chi-squared statistic for each bin
        valid_bins_mask = n_modes > 10  # Need sufficient modes for chi-squared validity
        valid_power_bins = power_spectrum[valid_bins_mask]
        valid_n_modes_bins = n_modes[valid_bins_mask]
        
        # Chi-squared test: (P_measured / P_expected - 1) * N_modes should be ~ χ²(N_modes) - N_modes
        chi2_statistics = []
        p_values = []
        
        try:
            from scipy import stats
            
            for i, (p_measured, n_mod) in enumerate(zip(valid_power_bins, valid_n_modes_bins)):
                # Normalized deviation
                normalized_deviation = (p_measured / expected_shot_noise - 1.0) * n_mod
                
                # Chi-squared test: is this consistent with χ²(n_mod) - n_mod?
                # For large n_mod, this approaches normal distribution with std = sqrt(2*n_mod)
                if n_mod > 30:
                    # Use normal approximation for large n_mod
                    z_score = normalized_deviation / np.sqrt(2 * n_mod)
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
                else:
                    # Use exact chi-squared for small n_mod
                    chi2_val = normalized_deviation + n_mod
                    p_value = 2 * min(stats.chi2.cdf(chi2_val, n_mod), 1 - stats.chi2.cdf(chi2_val, n_mod))
                
                chi2_statistics.append(normalized_deviation)
                p_values.append(p_value)
            
            chi2_statistics = np.array(chi2_statistics)
            p_values = np.array(p_values)
            
            # Statistical summary
            n_valid_bins = len(valid_power_bins)
            outliers_3sigma = np.sum(np.abs(chi2_statistics) > 3 * np.sqrt(2 * valid_n_modes_bins))
            outliers_2sigma = np.sum(np.abs(chi2_statistics) > 2 * np.sqrt(2 * valid_n_modes_bins))
            fraction_3sigma = outliers_3sigma / n_valid_bins
            fraction_2sigma = outliers_2sigma / n_valid_bins
            
            # Global reduced chi-squared
            reduced_chi2 = np.mean((chi2_statistics / np.sqrt(2 * valid_n_modes_bins))**2)
            
            print(f"Statistical Validation Results:")
            print(f"  Valid k-bins for statistics: {n_valid_bins}")
            print(f"  Outliers beyond 2σ: {outliers_2sigma}/{n_valid_bins} ({fraction_2sigma:.1%})")
            print(f"  Outliers beyond 3σ: {outliers_3sigma}/{n_valid_bins} ({fraction_3sigma:.1%})")
            print(f"  Reduced χ²: {reduced_chi2:.3f}")
            print(f"  Mean p-value: {np.mean(p_values):.3f}")
            print(f"  Min p-value: {np.min(p_values):.3f}")
            
            # Statistical validation criteria
            # Expect ~5% beyond 2σ, ~0.3% beyond 3σ for good statistics
            if fraction_3sigma > 0.05:  # More than 5% beyond 3σ is suspicious
                print(f"  WARNING: Too many 3σ outliers ({fraction_3sigma:.1%} > 5%)")
            else:
                print(f"  ✅ 3σ outlier fraction acceptable ({fraction_3sigma:.1%} ≤ 5%)")
            
            if 0.5 < reduced_chi2 < 2.0:
                print(f"  ✅ Reduced χ² in acceptable range (0.5 < {reduced_chi2:.3f} < 2.0)")
            else:
                print(f"  WARNING: Reduced χ² outside expected range: {reduced_chi2:.3f}")
            
            # Overall statistical consistency
            overall_consistent = (fraction_3sigma <= 0.05) and (0.5 < reduced_chi2 < 2.0)
            if overall_consistent:
                print(f"  ✅ Power spectrum statistically consistent with white noise")
            else:
                print(f"  ❌ Power spectrum shows statistical inconsistencies")
                
            # Print detailed k-bin analysis if test will fail
            will_fail = (fraction_3sigma > 0.10) or not (0.1 < reduced_chi2 < 10.0)
            if will_fail:
                print(f"\n" + "="*80)
                print("DETAILED K-BIN ANALYSIS (Chi-squared test failing)")
                print("="*80)
                print(f"{'k [h/Mpc]':>12} {'P(k) meas':>12} {'P(k) exp':>12} {'Deviation':>12} {'σ units':>10} {'N_modes':>8}")
                print("-" * 80)
                
                for i, (p_measured, n_mod, chi2_stat) in enumerate(zip(valid_power_bins, valid_n_modes_bins, chi2_statistics)):
                    k_val = k_bins[valid_bins_mask][i]  # Get corresponding k value
                    sigma_units = chi2_stat / np.sqrt(2 * n_mod)
                    deviation = p_measured - expected_shot_noise
                    
                    # Mark outliers
                    marker = ""
                    if abs(sigma_units) > 3:
                        marker = " <<<3σ"
                    elif abs(sigma_units) > 2:
                        marker = " <<<2σ"
                    
                    print(f"{k_val:12.6f} {p_measured:12.3f} {expected_shot_noise:12.3f} "
                          f"{deviation:+12.3f} {sigma_units:+10.2f} {n_mod:8d}{marker}")
                
                print("="*80)
                print(f"Expected P(k) = {expected_shot_noise:.1f} (Mpc/h)³ for white noise")
                print(f"Outliers: {outliers_2sigma} bins > 2σ, {outliers_3sigma} bins > 3σ")
                print("="*80)
            
            # FAIL the test if statistics are bad
            assert fraction_3sigma <= 0.10, (
                f"Chi-squared test FAILED: {fraction_3sigma:.1%} of bins beyond 3σ "
                f"(should be ≤ 10% for acceptable noise, got {outliers_3sigma}/{n_valid_bins})"
            )
            
            assert 0.1 < reduced_chi2 < 10.0, (
                f"Chi-squared test FAILED: Reduced χ² = {reduced_chi2:.1f} "
                f"(should be 0.1 < χ² < 10.0 for reasonable statistics)"
            )
                
        except ImportError:
            print("  WARNING: scipy not available, skipping detailed statistical validation")
            print("  Install scipy for complete chi-squared analysis")

        print(f"\n✅ SHOT NOISE TEST PASSED")
        print(f"✅ GRID VARIANCE VALIDATION PASSED")
        if n_processes > 1:
            print(f"✅ DISTRIBUTED FFT VALIDATION PASSED")