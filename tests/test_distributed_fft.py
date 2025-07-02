"""
Distributed FFT functionality has been consolidated.

The distributed FFT validation that was previously in this file has been 
consolidated into TestPowerSpectrumCalculator.test_comprehensive_distributed_power_spectrum
in tests/test_power_spectrum.py.

This consolidation was necessary to avoid JAX reinitialization issues where
both tests tried to call jax.distributed.initialize() in the same Python process,
which is not supported by JAX.

The comprehensive test now validates:
1. Distributed FFT correctness (ratio ≈ 1.0 vs independent FFTs ≈ 0.5)
2. Grid variance validation 
3. Chi-squared total variance testing
4. Chi-squared k-bin statistical validation

This provides complete distributed FFT validation while avoiding test runner
limitations with JAX distributed mode reinitialization.
"""

# This file is intentionally left with just documentation.
# All distributed FFT tests are now in test_power_spectrum.py