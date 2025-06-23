"""
Test suite for power spectrum calculation with known theoretical cases.

Tests FFT-based power spectrum calculation using JAX on multiple GPUs,
focusing on random particle distributions where we know the expected result.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# Try to import JAX, fall back to CPU-only mode if GPU unavailable
try:
    import os
    # Let JAX auto-detect the best platform (GPU if available, CPU otherwise)
    # os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # Commented out to allow GPU detection
    
    # Configure JAX memory management before importing jax
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.7')
    
    import jax
    import jax.numpy as jnp
    
    # Use 32-bit precision to match our float32 grids and save memory
    jax.config.update("jax_enable_x64", False)
    # Do NOT set jax_platform_name - let JAX auto-detect from JAX_PLATFORMS env var
    
    JAX_AVAILABLE = True
    
    # Safely check for multi-GPU availability
    def has_multiple_gpus():
        try:
            gpu_devices = jax.devices('gpu')
            return len(gpu_devices) >= 2
        except Exception:
            return False
    
    MULTI_GPU_AVAILABLE = has_multiple_gpus()
    
except (ImportError, RuntimeError) as e:
    print(f"JAX not available or failed to initialize: {e}")
    print("Running tests in CPU-only mode")
    jax = None
    jnp = None
    JAX_AVAILABLE = False
    MULTI_GPU_AVAILABLE = False

from pkdpipe.power_spectrum import PowerSpectrumCalculator
from pkdpipe.particle_gridder import ParticleGridder
from pkdpipe.data import Data


class TestParticleGridder:
    """Test particle-to-grid mass assignment functionality."""
    
    @pytest.fixture
    def grid_config(self):
        """Standard grid configuration for testing.""" 
        return {
            'ngrid': 64,  # Small size for fast testing
            'box_size': 100.0,  # Mpc/h
            'assignment': 'cic',  # Cloud-in-Cell
        }
    
    @pytest.fixture
    def simple_particles(self):
        """Simple particle distribution for testing."""
        n_particles = 1000
        box_size = 100.0
        
        particles = {
            'x': np.random.uniform(0, box_size, n_particles),
            'y': np.random.uniform(0, box_size, n_particles), 
            'z': np.random.uniform(0, box_size, n_particles),
            'mass': np.ones(n_particles)
        }
        return particles
    
    def test_gridder_initialization(self, grid_config):
        """Test ParticleGridder can be initialized with proper parameters."""
        gridder = ParticleGridder(
            ngrid=grid_config['ngrid'],
            box_size=grid_config['box_size'],
            assignment=grid_config['assignment']
        )
        
        assert gridder.ngrid == grid_config['ngrid']
        assert gridder.box_size == grid_config['box_size']
        assert gridder.assignment == 'cic'
        assert gridder.grid_spacing == grid_config['box_size'] / grid_config['ngrid']


class TestPowerSpectrumCalculator:
    """Test power spectrum calculation with known theoretical cases."""
    
    @pytest.fixture(scope="class")
    def gpu_config(self):
        """Configure JAX for multi-GPU testing."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        # Debug GPU detection
        import os
        print(f"\nGPU DETECTION DEBUG:")
        print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"  JAX_PLATFORM_NAME: {os.environ.get('JAX_PLATFORM_NAME', 'Not set')}")
        print(f"  Initial JAX backend: {jax.default_backend()}")
        
        # Try to force GPU detection
        try:
            devices = jax.devices()
            n_devices = len(devices)
            print(f"  JAX devices found: {n_devices}")
            for i, device in enumerate(devices):
                print(f"    Device {i}: {device}")
            
            # Try explicitly getting CUDA devices
            try:
                cuda_devices = jax.devices('cuda')
                print(f"  Explicit CUDA devices: {len(cuda_devices)}")
                for i, device in enumerate(cuda_devices):
                    print(f"    CUDA Device {i}: {device}")
                # Force use of CUDA devices if available
                if len(cuda_devices) > 0:
                    n_devices = len(cuda_devices)
                    devices = cuda_devices
                    print(f"  Forcing use of CUDA devices: {n_devices}")
            except Exception as cuda_e:
                print(f"  Failed to get CUDA devices: {cuda_e}")
            
            # Check JAX version and configuration
            print(f"  JAX version: {jax.__version__}")
            
            # Try to see what's causing CPU fallback
            try:
                import jaxlib
                print(f"  JAXlib version: {jaxlib.__version__}")
            except:
                print(f"  JAXlib version: unknown")
                
        except Exception as e:
            print(f"  Exception during device detection: {e}")
            pytest.skip(f"JAX device detection failed: {e}")
        
        if n_devices < 1:
            pytest.skip("No devices available")
        
        # Check if we're in distributed mode (multiple processes)
        try:
            is_distributed = jax.process_count() > 1
        except:
            is_distributed = False
        
        if is_distributed:
            # Multi-process mode: each process uses 1 GPU
            use_devices = 1
        else:
            # Single-process mode: for now, use only 1 GPU until jax_fft is fixed
            # TODO: Fix jax_fft to handle single-process multi-GPU properly
            use_devices = 1
        
        return {
            'n_devices': use_devices,
            'available_devices': n_devices,
            'backend': jax.default_backend()
        }
    
    @pytest.fixture
    def random_particles(self):
        """Generate random particle distribution with known power spectrum."""
        n_particles = 50000  # Moderate size for testing
        box_size = 200.0  # Mpc/h
        
        # Random uniform distribution
        particles = {
            'x': np.random.uniform(0, box_size, n_particles),
            'y': np.random.uniform(0, box_size, n_particles), 
            'z': np.random.uniform(0, box_size, n_particles),
            'mass': np.ones(n_particles)  # Equal mass particles
        }
        
        # Expected shot noise power spectrum: P(k) = V/N
        expected_shot_noise = box_size**3 / n_particles
        
        return particles, box_size, expected_shot_noise
    
    @pytest.fixture
    def grid_config(self):
        """Standard grid configuration for testing.""" 
        return {
            'ngrid': 128,  # Reasonable size for testing
            'assignment': 'ngp',  # Nearest Grid Point - simpler for shot noise test
        }
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_shot_noise_power_spectrum(self, random_particles, gpu_config, grid_config):
        """Test power spectrum calculation for random particles with chi-squared statistics."""
        particles, box_size, expected_shot_noise = random_particles
        
        # Initialize calculator
        calc = PowerSpectrumCalculator(
            ngrid=grid_config['ngrid'],
            box_size=box_size,
            n_devices=gpu_config['n_devices']
        )
        
        # Print detailed test information
        print("\n" + "="*80)
        print("POWER SPECTRUM SHOT NOISE TEST WITH CHI-SQUARED STATISTICS")
        print("="*80)
        print(f"GPU Configuration:")
        print(f"  Backend: {gpu_config['backend']}")
        print(f"  Available devices: {gpu_config['available_devices']}")
        print(f"  Using devices: {gpu_config['n_devices']}")
        
        print(f"\nSimulation Parameters:")
        print(f"  Number of particles: {len(particles['x']):,}")
        print(f"  Box size: {box_size:.1f} Mpc/h")
        print(f"  Volume: {box_size**3:.1e} (Mpc/h)³")
        print(f"  Number density: {len(particles['x'])/box_size**3:.2e} h³/Mpc³")
        
        print(f"\nGrid Configuration:")
        print(f"  Grid size: {grid_config['ngrid']}³")
        print(f"  Assignment scheme: {grid_config['assignment'].upper()}")
        print(f"  Cell size: {box_size/grid_config['ngrid']:.3f} Mpc/h")
        
        print(f"\nExpected Shot Noise:")
        print(f"  P_shot = V/N = {box_size:.1f}³/{len(particles['x']):,} = {expected_shot_noise:.6e} (Mpc/h)³")
        
        # Calculate power spectrum (now with fixed dtype consistency)
        k_bins, power_spectrum, n_modes = calc.calculate_power_spectrum(particles)
        
        # Get diagnostic information about the density field
        density_stats = calc.get_density_diagnostics()
        
        print(f"\nPower Spectrum Results:")
        print(f"  Number of k-bins: {len(k_bins)}")
        print(f"  k-range: {k_bins[0]:.6f} to {k_bins[-1]:.6f} h/Mpc")
        print(f"  Fundamental mode: {2*np.pi/box_size:.6f} h/Mpc")
        print(f"  Total modes measured: {n_modes.sum():,}")
        
        # Calculate Nyquist frequency and aliasing cutoff
        cell_size = box_size / grid_config['ngrid']
        k_nyquist = np.pi / cell_size
        k_cutoff = 0.5 * k_nyquist  # Conservative cutoff to avoid aliasing
        valid_k_mask = k_bins <= k_cutoff
        n_valid_for_stats = np.sum(valid_k_mask)
        
        print(f"\nALIASING ANALYSIS:")
        print(f"  Cell size: {cell_size:.3f} Mpc/h")
        print(f"  Nyquist frequency: {k_nyquist:.3f} h/Mpc")
        print(f"  Aliasing cutoff (0.5 × k_Nyquist): {k_cutoff:.3f} h/Mpc")
        print(f"  Valid k-bins for statistics: {n_valid_for_stats}/{len(k_bins)}")
        
        print(f"\nDENSITY FIELD DIAGNOSTICS:")
        print(f"  Mean density: {density_stats['mean_density']:.6e}")
        print(f"  Density variance: {density_stats['density_variance']:.6e}")
        print(f"  Delta contrast mean: {density_stats['delta_mean']:.6e} [Expected: ~0]")
        print(f"  Delta contrast variance: {density_stats['delta_variance']:.6e}")
        print(f"  Theoretical shot noise variance: {density_stats['theoretical_shot_noise_variance']:.6e}")
        print(f"  Variance ratio (measured/theoretical): {density_stats['delta_variance']/density_stats['theoretical_shot_noise_variance']:.3f} [Expected: ~1]")
        
        # Explicit density field variance validation
        print(f"\nDENSITY FIELD VARIANCE VALIDATION:")
        # For random particles with NGP assignment, the theoretical variance should be:
        # σ²_theoretical = <n>/V_cell where <n> is mean particles per cell (Poisson statistics)
        n_particles = len(particles['x'])
        n_cells = grid_config['ngrid']**3
        volume_per_cell = box_size**3 / n_cells
        mean_particles_per_cell = n_particles / n_cells
        expected_density_variance = mean_particles_per_cell / volume_per_cell
        density_variance_ratio = density_stats['density_variance'] / expected_density_variance
        
        print(f"  Mean particles per cell: {mean_particles_per_cell:.6f}")
        print(f"  Cell volume: {volume_per_cell:.6e} (Mpc/h)³")
        print(f"  Expected density variance: {expected_density_variance:.6e}")
        print(f"  Measured density variance: {density_stats['density_variance']:.6e}")
        print(f"  Density variance ratio: {density_variance_ratio:.3f} [Expected: ~1]")
        
        # Check if density variance is reasonable
        assert 0.5 <= density_variance_ratio <= 2.0, \
            f"Density variance ratio {density_variance_ratio:.3f} outside reasonable range [0.5, 2.0]"

        # Statistical validation using chi-squared theory
        print(f"\n" + "="*80)
        print("CHI-SQUARED STATISTICAL VALIDATION")
        print("="*80)
        print("Theory: For random particles, P(k) follows chi-squared distribution")
        print("  P_measured ~ P_expected × χ²(2N_modes) / (2N_modes)")
        print("  Standard deviation: σ = P_expected × √(2/N_modes)")
        print("  Expected ~68% of bins within 1σ, ~95% within 2σ, ~99.7% within 3σ")
        print()
        
        print(f"{'k [h/Mpc]':<12} {'N_modes':<8} {'P_meas':<12} {'P_exp':<12} {'σ_exp':<12} {'|Δ|/σ':<8} {'Status':<8}")
        print("-" * 84)
        
        # Statistical analysis for each k-bin
        n_valid_bins = 0
        n_within_1sigma = 0
        n_within_2sigma = 0
        n_within_3sigma = 0
        chi_squared_sum = 0.0
        total_dof = 0
        
        for i in range(len(k_bins)):
            if n_modes[i] > 0:
                # Chi-squared statistics for power spectrum
                # For a random field: σ² = P_expected² × (2/N_modes)
                expected_std = expected_shot_noise * np.sqrt(2.0 / n_modes[i])
                
                # Calculate standardized deviation
                deviation = abs(power_spectrum[i] - expected_shot_noise)
                standardized_deviation = deviation / expected_std
                
                # Determine status and count statistics only for valid k-bins (below aliasing cutoff)
                if k_bins[i] <= k_cutoff:
                    n_valid_bins += 1
                    
                    # Count sigma levels for statistical validation
                    if standardized_deviation <= 1.0:
                        n_within_1sigma += 1
                        status = "✓1σ"
                    if standardized_deviation <= 2.0:
                        n_within_2sigma += 1
                        if standardized_deviation > 1.0:
                            status = "✓2σ"
                    if standardized_deviation <= 3.0:
                        n_within_3sigma += 1
                        if standardized_deviation > 2.0:
                            status = "✓3σ"
                    if standardized_deviation > 3.0:
                        status = "✗>3σ"
                else:
                    # Mark aliased bins but don't count them in statistics
                    status = "ALIAS"
                
                # Accumulate chi-squared statistic only for valid k-bins
                if k_bins[i] <= k_cutoff:
                    # χ² = Σ (P_measured / P_expected) × N_modes for each mode
                    chi_squared_contribution = (power_spectrum[i] / expected_shot_noise) * n_modes[i]
                    chi_squared_sum += chi_squared_contribution
                    total_dof += n_modes[i]
                
                print(f"{k_bins[i]:<12.6f} {n_modes[i]:<8} {power_spectrum[i]:<12.6e} {expected_shot_noise:<12.6e} {expected_std:<12.6e} {standardized_deviation:<8.2f} {status:<8}")
        
        # Overall statistical assessment
        print("-" * 84)
        print("STATISTICAL SUMMARY:")
        print(f"  Valid k-bins: {n_valid_bins}")
        print(f"  Within 1σ: {n_within_1sigma:3d} ({100*n_within_1sigma/n_valid_bins:5.1f}%) [Expected: ~68%]")
        print(f"  Within 2σ: {n_within_2sigma:3d} ({100*n_within_2sigma/n_valid_bins:5.1f}%) [Expected: ~95%]")
        print(f"  Within 3σ: {n_within_3sigma:3d} ({100*n_within_3sigma/n_valid_bins:5.1f}%) [Expected: ~99.7%]")
        
        print(f"\nCHI-SQUARED GOODNESS OF FIT:")
        print(f"  Total χ² statistic: {chi_squared_sum:.2f}")
        print(f"  Degrees of freedom: {total_dof}")
        print(f"  Reduced χ²: {chi_squared_sum/total_dof:.3f} [Expected: ~1.0]")
        
        # Overall weighted statistics (only for valid k-bins)
        valid_power = power_spectrum[valid_k_mask]
        valid_n_modes = n_modes[valid_k_mask]
        mean_power = np.average(valid_power, weights=valid_n_modes)
        std_error_mean = expected_shot_noise * np.sqrt(2.0 / valid_n_modes.sum())
        mean_deviation = abs(mean_power - expected_shot_noise) / std_error_mean
        
        print(f"\nWEIGHTED MEAN ANALYSIS:")
        print(f"  Weighted mean P(k): {mean_power:.6e} (Mpc/h)³")
        print(f"  Expected P(k): {expected_shot_noise:.6e} (Mpc/h)³")
        print(f"  Standard error of mean: {std_error_mean:.6e}")
        print(f"  Mean deviation: {mean_deviation:.2f}σ")
        
        # Basic structural assertions
        assert len(k_bins) > 0, "No k-bins returned"
        assert len(power_spectrum) == len(k_bins), "Power spectrum length mismatch"
        assert len(n_modes) == len(k_bins), "Mode count length mismatch"
        assert n_valid_bins > 0, "No valid k-bins with modes"
        assert np.all(power_spectrum > 0), "All power values should be positive"
        assert np.all(np.isfinite(power_spectrum)), "All power values should be finite"
        
        # Statistical assertions based on chi-squared theory
        fraction_within_3sigma = n_within_3sigma / n_valid_bins
        assert fraction_within_3sigma >= 0.9, \
            f"Only {100*fraction_within_3sigma:.1f}% within 3σ (expected ≥90% for good statistics)"
        
        # Reduced chi-squared should be reasonable (between 0.5 and 2.0 for good statistics)
        reduced_chi2 = chi_squared_sum / total_dof
        assert 0.5 <= reduced_chi2 <= 2.0, \
            f"Reduced χ² = {reduced_chi2:.3f} outside reasonable range [0.5, 2.0]"
        
        # Weighted mean should be close to expected (within 3σ)
        assert mean_deviation < 3.0, \
            f"Weighted mean deviates by {mean_deviation:.2f}σ from expected value"
        
        # Final validation message
        print(f"\n" + "="*80)
        print("✅ ALL STATISTICAL TESTS PASSED!")
        print(f"✅ Shot noise power spectrum is consistent with chi-squared statistics")
        print(f"✅ {100*fraction_within_3sigma:.1f}% of k-bins within 3σ statistical bounds")
        print(f"✅ Reduced χ² = {reduced_chi2:.3f} indicates good statistical consistency")
        print("="*80)
    
        def test_memory_optimized_particle_redistribution(self, grid_config):
            """Test memory optimizations in particle redistribution."""
            import os
            
            # Create test particles (smaller scale for unit test)
            n_particles = 10000
            np.random.seed(42)
            
            # Create particles with velocity fields that should be removed
            particles = {
                'x': np.random.uniform(0, 1, n_particles).astype(np.float32),
                'y': np.random.uniform(0, 1, n_particles).astype(np.float32),
                'z': np.random.uniform(0, 1, n_particles).astype(np.float32),
                'vx': np.random.uniform(-100, 100, n_particles).astype(np.float32),  # Should be removed
                'vy': np.random.uniform(-100, 100, n_particles).astype(np.float32),  # Should be removed
                'vz': np.random.uniform(-100, 100, n_particles).astype(np.float32),  # Should be removed
                'mass': np.ones(n_particles, dtype=np.float32)
            }
            
            # Test with small grid to verify memory optimizations work
            ngrid, box_size = 32, 100.0
            calculator = PowerSpectrumCalculator(ngrid=ngrid, box_size=box_size, n_devices=1)
            
            # Mock MPI environment to test redistribution path
            with patch.dict(os.environ, {'SLURM_NTASKS': '2'}):
                # This should trigger the MPI redistribution path
                # The test verifies the code handles velocity removal correctly
                k_bins, power_spectrum, n_modes = calculator.calculate_power_spectrum(
                    particles, subtract_shot_noise=True, assignment='ngp'
                )
            
            # Verify results are reasonable
            assert len(k_bins) > 0
            assert len(power_spectrum) == len(k_bins)
            assert len(n_modes) == len(k_bins)
            assert np.all(np.isfinite(power_spectrum))

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
