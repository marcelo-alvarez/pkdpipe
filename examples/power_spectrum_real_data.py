#!/usr/bin/env python3
"""
Power Spectrum Analysis of Real Simulation Data

This example demonstrates how to analyze simulation data using the 
PowerSpectrumAnalysis high-level API. It shows both real simulation
data analysis and synthetic data generation for testing.

Example usage:
    python power_spectrum_real_data.py
    python power_spectrum_real_data.py --variant wcdm-validation
    python power_spectrum_real_data.py --ngrid 256 --assignment ngp
    python power_spectrum_real_data.py --debug-synthetic
"""

import argparse
import sys
import os
from pathlib import Path

# Configure environment for optimal performance
os.environ.setdefault('OMP_NUM_THREADS', '32')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '32')
os.environ.setdefault('MKL_NUM_THREADS', '32')
os.environ.setdefault('NUMEXPR_MAX_THREADS', '32')
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

# Suppress JAX warnings
import warnings
import logging
warnings.filterwarnings("ignore", message=".*libtpu.so.*")
warnings.filterwarnings("ignore", message=".*Failed to open libtpu.so.*")
logging.getLogger('jax._src.xla_bridge').setLevel(logging.ERROR)

from pkdpipe.power_spectrum_analysis import PowerSpectrumAnalysis, PowerSpectrumConfig


def run_power_spectrum_analysis(args):
    """Run power spectrum analysis using the PowerSpectrumAnalysis API."""
    
    # Get process ID for clean output
    process_id = int(os.environ.get('SLURM_PROCID', '0'))
    
    if process_id == 0:
        print("="*60)
        print("POWER SPECTRUM ANALYSIS")
        print("="*60)
        print(f"Configuration:")
        print(f"  Campaign: {Path(args.campaign_dir).name}")
        print(f"  Variant: {args.variant}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Grid size: {args.ngrid}³")
        print(f"  Assignment: {args.assignment}")
        print(f"  Debug mode: {args.debug_synthetic}")
    
    try:
        if args.debug_synthetic:
            # Use synthetic data
            config = PowerSpectrumConfig(
                ngrid=args.ngrid,
                assignment=args.assignment,
                dataset=args.dataset,
                n_devices=args.n_devices,
                debug_synthetic=True
            )
            
            analysis = PowerSpectrumAnalysis(config=config)
            
        else:
            # Use real simulation data
            config = PowerSpectrumConfig(
                campaign_dir=args.campaign_dir,
                variant=args.variant,
                dataset=args.dataset,
                ngrid=args.ngrid,
                assignment=args.assignment,
                n_devices=args.n_devices,
                debug_synthetic=False
            )
            
            analysis = PowerSpectrumAnalysis(config=config)
        
        # Run the analysis
        if process_id == 0:
            print(f"\nRunning power spectrum analysis...")
        
        results = analysis.run()
        
        # Display results
        if process_id == 0:
            print(f"\n✅ Analysis completed successfully!")
            print(f"  Number of k-bins: {len(results.k_bins)}")
            print(f"  k-range: {results.k_bins[0]:.6f} to {results.k_bins[-1]:.6f} h/Mpc")
            print(f"  Total modes: {results.n_modes.sum():,}")
            
            # Save results
            output_file = f"power_spectrum_ngrid{args.ngrid}_{args.assignment}.txt"
            results.save(output_file)
            print(f"  Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        if process_id == 0:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main function to run the power spectrum analysis."""
    
    parser = argparse.ArgumentParser(
        description="Analyze power spectrum using PowerSpectrumAnalysis API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--campaign-dir", default="/global/cfs/cdirs/cosmosim/slac/malvarez/pkdgrav3/campaigns/cosmosim-mocks-2025",
                       help="Path to campaign directory")
    parser.add_argument("--variant", default="lcdm-validation",
                       help="Simulation variant name")
    parser.add_argument("--dataset", default="xp", choices=["xp", "xvp", "xvh"],
                       help="Dataset type to read")
    parser.add_argument("--ngrid", type=int, default=256,
                       help="Grid size for power spectrum calculation")
    parser.add_argument("--assignment", default="cic", choices=["ngp", "cic", "tsc"],
                       help="Particle assignment scheme")
    parser.add_argument("--n-devices", type=int, default=None,
                       help="Number of GPU devices to use (auto-detects from SLURM if not specified)")
    parser.add_argument("--debug-synthetic", action="store_true",
                       help="Use synthetic random particle data for testing")
    
    args = parser.parse_args()
    
    # Auto-detect n_devices from SLURM environment if not specified
    if args.n_devices is None:
        ntasks = int(os.environ.get('SLURM_NTASKS', '1'))
        args.n_devices = 1 if ntasks > 1 else 1
    
    # Run analysis using the high-level API
    results = run_power_spectrum_analysis(args)
    
    # Example of using results
    process_id = int(os.environ.get('SLURM_PROCID', '0'))
    if process_id == 0:
        print(f"\nExample: Accessing results programmatically:")
        print(f"  k_bins shape: {results.k_bins.shape}")
        print(f"  power_spectrum shape: {results.power_spectrum.shape}")
        print(f"  metadata keys: {list(results.metadata.keys())}")
        print(f"  density_stats keys: {list(results.density_stats.keys())}")
    
    # Clean up
    import gc
    gc.collect()


if __name__ == "__main__":
    main()