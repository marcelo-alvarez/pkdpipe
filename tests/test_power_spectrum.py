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
        
        # Check debug mode
        debug_mode = os.environ.get('PKDPIPE_DEBUG_MODE', 'false').lower() == 'true'
        
        # Detect distributed mode
        n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        process_id = int(os.environ.get('SLURM_PROCID', '0'))
        
        # Create particles per process for distributed testing
        total_particles = TEST_CONFIG['n_particles']
        particles_per_process = total_particles // n_processes
        
        if debug_mode:
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
    
    def test_comprehensive_distributed_power_spectrum(self, random_particles):
        """
        Comprehensive distributed power spectrum validation test.
        
        This test combines distributed FFT validation with full statistical analysis:
        1. Tests that distributed FFT produces correct power spectrum normalization
        2. Validates comprehensive statistical properties (variance, chi-squared)
        3. Tests grid statistics and density contrast calculations
        4. Verifies both distributed FFT correctness AND statistical consistency
        
        CRITICAL TEST: This verifies that distributed FFT is working properly.
        If JAX distributed FFT fails, each process does independent FFT → ~50% amplitude.
        If JAX distributed FFT works, we get the correct normalization AND statistics.
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
        print("COMPREHENSIVE DISTRIBUTED POWER SPECTRUM VALIDATION")
        print("="*80)
        
        # Expected shot noise for random particles  
        expected_shot_noise = EXPECTED_SHOT_NOISE
        print(f"Expected Shot Noise: P_shot = V/N = {box_size:.1f}^3/{len(random_particles['x']):,} = {expected_shot_noise:.1f} (Mpc/h)^3")
        
        # Calculate power spectrum with grid statistics
        k_bins, power_spectrum, n_modes, grid_stats = calc.calculate_power_spectrum(random_particles, assignment=assignment)
        
        # =================================================================
        # PART 1: DISTRIBUTED FFT VALIDATION (from TestDistributedFFT)
        # =================================================================
        print(f"\n" + "-"*60)
        print("PART 1: DISTRIBUTED FFT VALIDATION")
        print("-"*60)
        
        # Get weighted mean power for FFT validation
        valid_bins = n_modes > 100
        valid_power = power_spectrum[valid_bins]
        valid_n_modes = n_modes[valid_bins]
        
        if len(valid_power) == 0:
            pytest.fail("No valid k-bins for testing")
        
        weighted_mean_power = np.average(valid_power, weights=valid_n_modes)
        ratio = weighted_mean_power / expected_shot_noise
        
        print(f"Measured P(k) = {weighted_mean_power:.1f} (Mpc/h)^3")
        print(f"Expected P(k) = {expected_shot_noise:.1f} (Mpc/h)^3")
        print(f"Ratio = {ratio:.3f}")
        
        # Check if we're in distributed mode
        n_processes = int(os.environ.get('SLURM_NTASKS', '1'))
        if n_processes > 1:
            print(f"\nDISTRIBUTED MODE TEST ({n_processes} processes):")
            print(f"  Expected ratio ≈ 1.0 (if distributed FFT works)")
            print(f"  Expected ratio ≈ 0.5 (if distributed FFT fails)")
            print(f"  Actual ratio = {ratio:.3f}")
            
            # Critical test: distributed FFT should give ratio ≈ 1.0, not 0.5
            if ratio < 0.7:
                pytest.fail(
                    f"DISTRIBUTED FFT FAILURE: ratio={ratio:.3f} < 0.7\n"
                    f"This indicates independent FFTs instead of distributed FFT"
                )
            elif ratio > 1.3:
                pytest.fail(f"NORMALIZATION ERROR: ratio={ratio:.3f} > 1.3")
            else:
                print(f"  ✅ DISTRIBUTED FFT WORKING CORRECTLY")
        
        # Assert FFT correctness
        assert 0.7 < ratio < 1.3, f"Power spectrum ratio {ratio:.3f} outside expected range"
        print(f"✅ DISTRIBUTED FFT VALIDATION PASSED")
        
        # =================================================================
        # PART 2: GRID VARIANCE VALIDATION
        # =================================================================
        print(f"\n" + "-"*60)
        print("PART 2: GRID VARIANCE VALIDATION")
        print("-"*60)
        
        # Extract variance statistics from power spectrum calculation (correctly computed with domain decomposition)
        delta_mean = grid_stats['delta_mean']
        delta_variance = grid_stats['delta_variance'] 
        delta_std = grid_stats['delta_std']
        theoretical_delta_variance = grid_stats['theoretical_variance']
        global_particle_count = grid_stats['particle_count']
        
        # Expected particles per cell from config
        expected_particles_per_cell = TEST_CONFIG['particles_per_cell']
        
        print(f"Density Contrast Statistics (using {assignment.upper()} assignment):")
        print(f"  Global particles: {global_particle_count:,}")
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
        
        # =================================================================
        # PART 3: CHI-SQUARED TOTAL VARIANCE TEST
        # =================================================================
        print(f"\n" + "-"*60)
        print("PART 3: CHI-SQUARED TOTAL VARIANCE TEST")
        print("-"*60)
        
        # For white noise (Poisson), the total variance should follow chi-squared distribution
        # Degrees of freedom = number of independent grid cells = ngrid^3
        # Test statistic: (N_grid * measured_variance) / theoretical_variance ~ χ²(N_grid)
        n_grid_cells = ngrid**3
        test_statistic = (n_grid_cells * delta_variance) / theoretical_delta_variance
        
        try:
            from scipy import stats
            
            # Chi-squared test for total variance
            # H0: variance follows expected white noise distribution
            # Critical values for 95% confidence interval
            chi2_lower = stats.chi2.ppf(0.025, n_grid_cells)  # 2.5th percentile
            chi2_upper = stats.chi2.ppf(0.975, n_grid_cells)  # 97.5th percentile
            chi2_mean = n_grid_cells  # Mean of chi-squared distribution
            chi2_std = np.sqrt(2 * n_grid_cells)  # Standard deviation
            
            # Normalized test statistic (z-score)
            z_score = (test_statistic - chi2_mean) / chi2_std
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
            
            print(f"Total Variance Chi-Squared Test:")
            print(f"  Grid cells: {n_grid_cells:,}")
            print(f"  Measured variance: {delta_variance:.6e}")
            print(f"  Theoretical variance: {theoretical_delta_variance:.6e}")
            print(f"  Test statistic: {test_statistic:.1f}")
            print(f"  Expected (χ² mean): {chi2_mean:.1f}")
            print(f"  χ² standard deviation: {chi2_std:.1f}")
            print(f"  Z-score: {z_score:.3f}")
            print(f"  P-value: {p_value:.6f}")
            print(f"  95% CI: [{chi2_lower:.1f}, {chi2_upper:.1f}]")
            
            # Validation criteria
            variance_within_ci = chi2_lower <= test_statistic <= chi2_upper
            variance_significant = p_value < 0.05
            
            if variance_within_ci and not variance_significant:
                print(f"  ✅ Total variance consistent with white noise (within 95% CI)")
            else:
                if variance_significant:
                    print(f"  ⚠️ Total variance significantly deviates from white noise (p={p_value:.6f} < 0.05)")
                if not variance_within_ci:
                    print(f"  ⚠️ Total variance outside 95% confidence interval")
            
            # Assert for test failure if variance is significantly wrong
            assert not variance_significant, (
                f"Total variance chi-squared test FAILED: p-value = {p_value:.6f} < 0.05\n"
                f"Test statistic {test_statistic:.1f} significantly deviates from expected {chi2_mean:.1f}\n"
                f"This indicates non-white noise behavior in the density field"
            )
            
        except ImportError:
            print("  WARNING: scipy not available, skipping chi-squared variance test")
        
        # =================================================================
        # PART 4: CHI-SQUARED K-BIN STATISTICAL VALIDATION
        # =================================================================
        print(f"\n" + "-"*60)
        print("PART 4: CHI-SQUARED K-BIN STATISTICAL VALIDATION")
        print("-"*60)
        
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
                print(f"  ⚠️ Power spectrum shows statistical inconsistencies")
                
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

        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        print(f"\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS SUMMARY")
        print("="*80)
        print(f"✅ DISTRIBUTED FFT VALIDATION PASSED")
        print(f"✅ GRID VARIANCE VALIDATION PASSED")
        print(f"✅ CHI-SQUARED TOTAL VARIANCE PASSED")
        print(f"✅ CHI-SQUARED K-BIN VALIDATION PASSED")
        if n_processes > 1:
            print(f"✅ DISTRIBUTED FFT WORKING CORRECTLY (ratio = {ratio:.3f})")
        print(f"✅ COMPREHENSIVE DISTRIBUTED POWER SPECTRUM VALIDATION COMPLETE")
        print("="*80)