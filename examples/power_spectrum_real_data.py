#!/usr/bin/env python3
"""
Power Spectrum Analysis of Real Simulation Data

This example demonstrates how to read actual simulation data from the 
cosmosim-mocks-2025 campaign and compute its power spectrum using the 
pkdpipe power spectrum analysis tools.

Example usage:
    python power_spectrum_real_data.py
    python power_spectrum_real_data.py --variant wcdm-validation
    python power_spectrum_real_data.py --ngrid 256 --assignment ngp
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

# Configure NumPy threading FIRST to use all CPU cores for gridding operations
# This is critical for scaling particle gridding across all 32 cores per GPU
os.environ.setdefault('OMP_NUM_THREADS', '32')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '32')
os.environ.setdefault('MKL_NUM_THREADS', '32')
os.environ.setdefault('NUMEXPR_MAX_THREADS', '32')

# Configure JAX to use CUDA backend on Perlmutter
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
import warnings
import logging
warnings.filterwarnings("ignore", message=".*libtpu.so.*")
warnings.filterwarnings("ignore", message=".*Failed to open libtpu.so.*")
logging.getLogger('jax._src.xla_bridge').setLevel(logging.ERROR)

# JAX will be initialized automatically when FFT operations are needed

# JAX will be imported only when FFT operations are needed
JAX_AVAILABLE = True  # Assume JAX is available - will be verified when needed

from pkdpipe.data import Data
from pkdpipe.power_spectrum import PowerSpectrumCalculator


def find_simulation_data(campaign_dir, variant_name):
    """Find the final snapshot file for a given simulation variant."""
    variant_dir = Path(campaign_dir) / "runs" / variant_name
    
    if not variant_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found: {variant_dir}")
    
    print(f"Searching for simulation data in: {variant_dir}")
    
    # Look for different types of snapshot files
    snapshot_patterns = ["*.tps", "*.lcp", "*.fof"]  # Tipsy, lightcone particles, halos
    snapshot_files = []
    
    # First try with extensions
    for pattern in snapshot_patterns:
        files = list(variant_dir.glob(pattern))
        if files:
            snapshot_files.extend(files)
            print(f"Found {len(files)} {pattern} files")
    
    # If no files with extensions found, look in output directory for numbered files
    if not snapshot_files:
        output_dir = variant_dir / "output"
        if output_dir.exists():
            print(f"Looking for snapshot files in: {output_dir}")
            # Look for files like variant.00001, variant.00002, etc.
            # Exclude auxiliary files (.fofstats, .hpb, .lcp, etc.)
            all_files = list(output_dir.glob(f"{variant_name}.[0-9]*"))
            files = [f for f in all_files if not any(suffix in f.name for suffix in ['.fofstats', '.hpb', '.lcp'])]
            if files:
                snapshot_files.extend(files)
                print(f"Found {len(files)} snapshot files without extensions")
    
    if not snapshot_files:
        raise FileNotFoundError(f"No simulation snapshot files found in {variant_dir}")
    
    # Sort by modification time to get the most recent (likely final) snapshot
    snapshot_files.sort(key=lambda x: x.stat().st_mtime)
    final_snapshot = snapshot_files[-1]
    
    print(f"Selected final snapshot: {final_snapshot.name}")
    print(f"File size: {final_snapshot.stat().st_size / (1024**3):.2f} GB")
    
    return final_snapshot


def load_particle_data(snapshot_file, dataset_type='xvp'):
    """Load particle data from simulation snapshot."""
    
    # During I/O phase, only process 0 prints (based on SLURM_PROCID)
    process_id = int(os.environ.get('SLURM_PROCID', '0'))
    
    # Only print from master process to avoid cluttered output
    if process_id == 0:
        print(f"\n" + "="*60)
        print("PARTICLE DATA LOADING")
        print("="*60)
        print(f"Snapshot file: {snapshot_file}")
        print(f"File size: {snapshot_file.stat().st_size / (1024**3):.2f} GB")
    
    try:
        # Find the parameter file - it should be in the parent directory
        variant_dir = snapshot_file.parent.parent
        param_files = list(variant_dir.glob("*.par"))
        if not param_files:
            raise FileNotFoundError(f"No parameter file found in {variant_dir}")
        param_file = param_files[0]
        
        if process_id == 0:
            print(f"Parameter file: {param_file}")
        
        # Initialize data reader with parameter file
        # Use all available CPU cores for I/O (no JAX conflicts during I/O phase)
        n_io_procs = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
        slurm_ntasks = int(os.environ.get('SLURM_NTASKS', '1'))
        
        if process_id == 0:
            if slurm_ntasks > 1:
                total_io_procs = slurm_ntasks * n_io_procs
                print(f"Multi-process I/O: {slurm_ntasks} processes, {n_io_procs} cores each ({total_io_procs} total)")
            else:
                print(f"Single process I/O: using {n_io_procs} cores")
        
        if process_id == 0:
            print(f"Initializing data reader...")
        
        # Only enable verbose logging on process 0 to avoid repeated output
        # Set verbose=False to test clean progress reporting
        data_reader = Data(param_file=str(param_file), verbose=False)
        
        # Read particle positions (needed for power spectrum)
        # Use a large bounding box to get all particles
        bbox = [[-1000, 1000], [-1000, 1000], [-1000, 1000]]
        
        if process_id == 0:
            print(f"Reading particles with bbox: {bbox}")
            print(f"Dataset: {dataset_type}, Format: tps")
            print("Starting I/O operation... (this may take several minutes)")
            print("Note: Progress updates will appear below (may be slow to start)")
        
        import time
        start_time = time.time()
        
        # fetch_data will automatically decide streaming vs standard mode
        result = data_reader.fetch_data(
            bbox=bbox,
            dataset=dataset_type,
            filetype='tps',
            lightcone=False,
            redshifts=[0.0]
        )
        
        # Get simulation parameters
        sim_params = data_reader.get_simulation_parameters()
        box_size = sim_params.get('dBoxSize', sim_params.get('box_size', 1000.0))
        
        if process_id == 0:
            elapsed_time = time.time() - start_time
            print(f"Data operation completed in {elapsed_time:.1f} seconds")
            print(f"Box size: {box_size:.1f} Mpc/h")
        
        return result, box_size, sim_params
        
    except Exception as e:
        raise RuntimeError(f"Failed to load particle data: {e}")


def calculate_power_spectrum(particles_or_data, box_size, ngrid=512, assignment='cic', n_devices=1):
    """Calculate the power spectrum of the particle distribution."""
    
    print(f"\n" + "="*60)
    print("POWER SPECTRUM CALCULATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  Grid size: {ngrid}³")
    print(f"  Assignment scheme: {assignment.upper()}")
    print(f"  Box size: {box_size:.1f} Mpc/h")
    print(f"  Cell size: {box_size/ngrid:.3f} Mpc/h")
    print(f"  Requested n_devices: {n_devices}")
    
    # Note: JAX distributed mode detection and initialization is handled by PowerSpectrumCalculator
    
    # Initialize power spectrum calculator
    print(f"\nInitializing PowerSpectrumCalculator...")
    calc = PowerSpectrumCalculator(
        ngrid=ngrid,
        box_size=box_size,
        n_devices=n_devices
    )
    
    # Handle different data formats from fetch_data
    if hasattr(particles_or_data, 'fetch_data_chunked'):
        # Data object returned for streaming mode
        data_input = particles_or_data
        print(f"Using streaming data processing")
    elif isinstance(particles_or_data, dict):
        # Dictionary format from snapshot mode
        # Convert first box to particle dictionary format
        particles_rec = list(particles_or_data.values())[0]
        
        # Handle mass field - tps format has mass, xvp dataset might not need it for power spectrum
        print(f"DEBUG: particles_rec type: {type(particles_rec)}")
        print(f"DEBUG: particles_rec dtype: {particles_rec.dtype}")
        print(f"DEBUG: particles_rec shape: {particles_rec.shape}")
        
        try:
            if 'mass' in particles_rec.dtype.names:
                print(f"DEBUG: Extracting mass field...")
                mass_data = particles_rec['mass']
                print(f"DEBUG: mass_data type: {type(mass_data)}, shape: {mass_data.shape}")
            else:
                # Default to unit mass for datasets without explicit mass
                print(f"DEBUG: Creating unit mass array...")
                mass_data = np.ones(len(particles_rec['x']))
            
            print(f"DEBUG: Converting x field...")
            x_data = np.array(particles_rec['x'])
            print(f"DEBUG: Converting y field...")
            y_data = np.array(particles_rec['y'])
            print(f"DEBUG: Converting z field...")
            z_data = np.array(particles_rec['z'])
            print(f"DEBUG: Converting mass field...")
            mass_array = np.array(mass_data)
            
            data_input = {
                'x': x_data,
                'y': y_data, 
                'z': z_data,
                'mass': mass_array
            }
            print(f"Converting snapshot data: {len(data_input['x']):,} particles")
            print(f"DEBUG: data_input keys: {list(data_input.keys())}")
            print(f"DEBUG: data_input['x'] type: {type(data_input['x'])}")
        except Exception as e:
            print(f"ERROR in particle conversion: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        raise ValueError(f"Unexpected data format: {type(particles_or_data)}")
    
    print(f"DEBUG: About to call calculate_power_spectrum...")
    # Calculate power spectrum
    print(f"Starting power spectrum calculation...")
    print(f"  Step 1: Particle gridding ({assignment.upper()} assignment)")
    print(f"  Step 2: FFT computation")
    print(f"  Step 3: Power spectrum binning")
    print(f"  Step 4: Cross-process reduction (if distributed mode)")
    
    # MPI barrier before timing particle gridding
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        comm.Barrier()
        process_id = comm.Get_rank()
        if process_id == 0:
            print(f"\n⏱️  Starting particle gridding timing...")
    except:
        process_id = 0
    
    import time
    gridding_start = time.time()
    
    k_bins, power_spectrum, n_modes = calc.calculate_power_spectrum(data_input, assignment=assignment)
    
    # MPI barrier after gridding to get accurate timing
    try:
        comm.Barrier()
        gridding_end = time.time()
        gridding_time = gridding_end - gridding_start
        if process_id == 0:
            print(f"⏱️  Particle gridding completed in {gridding_time:.2f} seconds")
            print(f"⏱️  Throughput: {len(data_input.get('x', [0])) / gridding_time / 1e6:.1f} M particles/sec")
    except:
        gridding_end = time.time()
        gridding_time = gridding_end - gridding_start
        print(f"⏱️  Particle gridding completed in {gridding_time:.2f} seconds")
    
    print(f"\n✅ Power spectrum calculation completed!")
    print(f"  Number of k-bins: {len(k_bins)}")
    print(f"  k-range: {k_bins[0]:.6f} to {k_bins[-1]:.6f} h/Mpc")
    print(f"  Total modes: {n_modes.sum():,}")
    
    # Get density field diagnostics
    print(f"Getting density field diagnostics...")
    density_stats = calc.get_density_diagnostics()
    
    return k_bins, power_spectrum, n_modes, density_stats


def analyze_results(k_bins, power_spectrum, n_modes, density_stats, box_size, n_particles, ngrid, assignment):
    """Analyze and display power spectrum results."""
    
    print(f"\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    # Frequency analysis
    cell_size = box_size / ngrid
    k_fundamental = 2 * np.pi / box_size
    k_nyquist = np.pi / cell_size
    k_cutoff = 0.5 * k_nyquist
    
    print(f"Frequency Analysis:")
    print(f"  Fundamental mode: {k_fundamental:.6f} h/Mpc")
    print(f"  Nyquist frequency: {k_nyquist:.3f} h/Mpc")
    print(f"  Recommended cutoff: {k_cutoff:.3f} h/Mpc")
    
    # Basic statistics - handle missing density stats gracefully
    print(f"\nDensity Field:")
    if 'mean_density' in density_stats and not np.isnan(density_stats['mean_density']):
        print(f"  Mean density: {density_stats['mean_density']:.6e}")
        
        # Calculate global mean density for comparison if we have particle count
        global_mean_density = n_particles / (box_size ** 3)
        print(f"  Global mean density (from particle count): {global_mean_density:.6e}")
        
        # Check consistency
        density_ratio = density_stats['mean_density'] / global_mean_density
        print(f"  Density ratio (gridded/global): {density_ratio:.4f}")
        if abs(density_ratio - 1.0) > 0.1:
            print(f"  WARNING: Large discrepancy in density calculation!")
    else:
        # Fallback: calculate from total particles and box volume
        global_mean_density = n_particles / (box_size ** 3)
        print(f"  Global mean density (from particle count): {global_mean_density:.6e}")
        print(f"  Note: Detailed density statistics not available in distributed mode")
    
    # Other density statistics if available
    if 'density_variance' in density_stats and not np.isnan(density_stats['density_variance']):
        print(f"  Density variance: {density_stats['density_variance']:.6e}")
    if 'delta_variance' in density_stats and not np.isnan(density_stats['delta_variance']):
        print(f"  Delta contrast variance: {density_stats['delta_variance']:.6e}")
    
    # Display power spectrum summary
    valid_mask = (k_bins <= k_cutoff) & (n_modes > 100)
    valid_bins = np.sum(valid_mask)
    
    print(f"\nPower Spectrum Results:")
    print(f"  Total k-bins: {len(k_bins)}")
    print(f"  Valid k-bins (within cutoff): {valid_bins}")
    print(f"  Power spectrum range: {power_spectrum.min():.2e} to {power_spectrum.max():.2e} (Mpc/h)³")
    print(f"  Total modes measured: {n_modes.sum():,}")
    
    # Save power spectrum to file
    output_file = f"power_spectrum_ngrid{ngrid}_{assignment}.txt"
    print(f"\nSaving power spectrum to: {output_file}")
    
    # Create header with metadata
    header = f"""# Power Spectrum Analysis Results
# Grid size: {ngrid}³
# Box size: {box_size:.1f} Mpc/h
# Assignment: {assignment.upper()}
# Cell size: {cell_size:.6f} Mpc/h
# Fundamental mode: {k_fundamental:.6f} h/Mpc
# Nyquist frequency: {k_nyquist:.6f} h/Mpc
# Cutoff frequency: {k_cutoff:.6f} h/Mpc
# Total particles: {n_particles:,}
# Total modes: {n_modes.sum():,}
# Valid k-bins: {valid_bins}
#
# Columns: k[h/Mpc] P(k)[(Mpc/h)³] N_modes Status
"""
    
    with open(output_file, 'w') as f:
        f.write(header)
        for i in range(len(k_bins)):
            if n_modes[i] > 0:
                status = "valid" if k_bins[i] <= k_cutoff else "aliased"
                f.write(f"{k_bins[i]:.6f} {power_spectrum[i]:.6e} {n_modes[i]} {status}\n")
    
    print(f"✅ Power spectrum analysis completed successfully!")
    print(f"✅ Results saved to {output_file}")
    
    return k_bins, power_spectrum, n_modes


def main():
    """Main function to run the power spectrum analysis."""
    
    parser = argparse.ArgumentParser(
        description="Analyze power spectrum of real simulation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--campaign-dir", default="/global/cfs/cdirs/cosmosim/slac/malvarez/pkdgrav3/campaigns/cosmosim-mocks-2025",
                       help="Path to campaign directory")
    parser.add_argument("--variant", default="lcdm-validation",
                       help="Simulation variant name")
    parser.add_argument("--dataset", default="xvp", choices=["xvp", "xv", "xvh"],
                       help="Dataset type to read")
    parser.add_argument("--ngrid", type=int, default=512,
                       help="Grid size for power spectrum calculation")
    parser.add_argument("--assignment", default="cic", choices=["ngp", "cic", "tsc"],
                       help="Particle assignment scheme")
    parser.add_argument("--n-devices", type=int, default=None,
                       help="Number of GPU devices to use (auto-detects from SLURM if not specified)")
    
    args = parser.parse_args()
    
    # Auto-detect n_devices from SLURM environment if not specified
    if args.n_devices is None:
        ntasks = int(os.environ.get('SLURM_NTASKS', '1'))
        if ntasks > 1:
            # In distributed SLURM mode, each process should use 1 device
            args.n_devices = 1
            print(f"Auto-detected n_devices=1 for distributed mode (SLURM_NTASKS={ntasks})")
        else:
            # Single process mode - use all available devices
            args.n_devices = 1
    
    # Check JAX availability (simple check - initialization handled by PowerSpectrumCalculator)
    if not JAX_AVAILABLE:
        print("Error: JAX is required for power spectrum calculation")
        sys.exit(1)
    
    # Use SLURM_PROCID for clean output during I/O phase (before JAX distributed mode)
    process_id = int(os.environ.get('SLURM_PROCID', '0'))
    
    if process_id == 0:
        print("="*60)
        print("POWER SPECTRUM ANALYSIS OF REAL SIMULATION DATA")
        print("="*60)
        print(f"Campaign: {Path(args.campaign_dir).name}")
        print(f"Variant: {args.variant}")
        print(f"Dataset: {args.dataset}")
    
    try:
        # Find simulation data
        snapshot_file = find_simulation_data(args.campaign_dir, args.variant)
        
        # Load particle data - Data class automatically handles streaming optimization
        particles_or_data, box_size, sim_params = load_particle_data(snapshot_file, args.dataset)
        
        # DEBUG: Check if we made it past data loading
        if process_id == 0:
            print(f"Data loading completed. Box size: {box_size}")
            if isinstance(particles_or_data, dict):
                # particles_or_data.values() gives recarrays, not dicts
                total_particles = sum(len(box) for box in particles_or_data.values())
                print(f"Total particles loaded: {total_particles:,}")
            else:
                print(f"Data object type: {type(particles_or_data)}")
        
        # Calculate power spectrum - PowerSpectrumCalculator handles both particle data and streaming
        if process_id == 0:
            print(f"About to call calculate_power_spectrum...")
            sys.stdout.flush()
        
        print(f"Process {process_id}: Entering calculate_power_spectrum...")
        k_bins, power_spectrum, n_modes, density_stats = calculate_power_spectrum(
            particles_or_data, box_size, args.ngrid, args.assignment, args.n_devices
        )
        print(f"Process {process_id}: Returned from calculate_power_spectrum...")
        
        # Get particle count for analysis (estimate from file size)
        file_size = snapshot_file.stat().st_size
        record_size = 36  # TPS format: 9 fields × 4 bytes
        header_size = 32
        n_particles = (file_size - header_size) // record_size
        
        # Analyze results
        analyze_results(k_bins, power_spectrum, n_modes, density_stats, 
                       box_size, n_particles, args.ngrid, args.assignment)
        
        print(f"\n" + "="*60)
        print("✅ POWER SPECTRUM ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Explicit cleanup to prevent OOM during Python exit
        if process_id == 0:
            print("\n🧹 Performing explicit cleanup to prevent OOM kills...")
        
        # Clear large variables
        del k_bins, power_spectrum, n_modes, density_stats
        if 'particles_or_data' in locals():
            del particles_or_data
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # JAX cleanup if available
        try:
            import jax
            # Clear JAX compilation cache
            jax.clear_backends()
            jax.clear_caches()
        except:
            pass
        
        # MPI barrier to ensure all processes finish before cleanup
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            comm.Barrier()
            if process_id == 0:
                print("✅ All MPI processes synchronized")
        except:
            pass
        
        # Final garbage collection
        gc.collect()
        
        if process_id == 0:
            print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()