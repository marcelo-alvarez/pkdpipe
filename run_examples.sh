#!/bin/bash
# Example Runner for pkdpipe
#
# This script runs the power spectrum real data example with proper SLURM configuration
# and environment setup. It automatically handles both local execution and SLURM job
# submission based on the environment.
#
# Usage:
#   ./run_examples.sh                           # Run with default parameters (lcdm-validation)
#   ./run_examples.sh --variant wcdm-validation # Run specific simulation variant
#   ./run_examples.sh --ngrid 512               # Custom grid size
#   ./run_examples.sh --debug-synthetic         # Use synthetic data for testing
#   ./run_examples.sh --time=30 --ntasks=8      # Custom SLURM parameters
#
# SLURM Options:
#   --nodes=N        Number of nodes (default: 1)
#   --time=N         Time limit in minutes (default: 60)
#   --ntasks=N       Number of MPI tasks (default: 4)
#   --cpus-per-task=N CPUs per task (default: 32)
#
# Example Script Options (passed through):
#   --variant NAME   Simulation variant name (default: lcdm-validation)
#   --dataset TYPE   Dataset type: xvp, xv, xvh (default: xvp)
#   --ngrid N        Grid size for FFT (default: 512)
#   --assignment SCHEME Assignment: ngp, cic, tsc (default: cic)
#   --debug-synthetic Use synthetic data instead of real simulation data

set -e  # Exit on any error

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments
SLURM_NODES=1
SLURM_TIME=15
SLURM_NTASKS=4
SLURM_CPUS_PER_TASK=32
SLURM_GPUS_PER_NODE=4
EXAMPLE_ARGS=()

# Function to show help
show_help() {
    echo "Example Runner for pkdpipe Power Spectrum Analysis"
    echo ""
    echo "Usage: $0 [SLURM_OPTIONS] [EXAMPLE_OPTIONS]"
    echo ""
    echo "SLURM Options:"
    echo "  --nodes=N         Number of nodes (default: 1)"
    echo "  --time=N          Time limit in minutes (default: 15)" 
    echo "  --ntasks=N        Number of MPI tasks (default: 4)"
    echo "  --cpus-per-task=N CPUs per task (default: 32)"
    echo "  --help            Show this help message"
    echo ""
    echo "Example Script Options (passed through):"
    echo "  --variant NAME    Simulation variant (default: lcdm-validation)"
    echo "  --dataset TYPE    Dataset type: xvp, xv, xvh (default: xvp)"
    echo "  --ngrid N         Grid size for FFT (default: 512)"
    echo "  --assignment TYPE Assignment: ngp, cic, tsc (default: cic)"
    echo "  --debug-synthetic Use synthetic data for testing"
    echo ""
    echo "Examples:"
    echo "  $0                                 # Default run"
    echo "  $0 --variant wcdm-validation       # Different simulation variant"
    echo "  $0 --ngrid 256 --debug-synthetic   # Custom grid + synthetic data"
    echo "  $0 --time=30 --ntasks=8            # Custom SLURM parameters"
}

# Parse arguments - pass most through to the example script
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --nodes=*)
            SLURM_NODES="${1#*=}"
            shift
            ;;
        --time=*)
            SLURM_TIME="${1#*=}"
            shift
            ;;
        --ntasks=*)
            SLURM_NTASKS="${1#*=}"
            shift
            ;;
        --cpus-per-task=*)
            SLURM_CPUS_PER_TASK="${1#*=}"
            shift
            ;;
        --gpus-per-node=*)
            SLURM_GPUS_PER_NODE="${1#*=}"
            shift
            ;;
        *)
            # Pass all other arguments to the example script
            EXAMPLE_ARGS+=("$1")
            shift
            ;;
    esac
done

# Load Python environment (only show message for main process)
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "🔧 Loading Python environment..."
fi
source ./load_env.sh

# Check if we're already running under SLURM
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # We're already in a SLURM job - load environment and run example directly
    source /global/cfs/cdirs/cosmosim/slac/malvarez/pkdgrav3/env/loadenv.sh
    source ./load_env.sh
    
    NTASKS=${SLURM_NTASKS:-1}
    PROCID=${SLURM_PROCID:-0}
    
    if [[ "${NTASKS}" -gt 1 ]]; then
        echo "🚀 Running power spectrum example in distributed mode: ${NTASKS} processes, rank ${PROCID}"
    else
        echo "🖥️  Running power spectrum example in single-process mode"
    fi
    
    # Only rank 0 prints the header
    if [[ "${PROCID}" -eq 0 ]]; then
        echo "==============================================="
        echo "PKDPIPE POWER SPECTRUM REAL DATA EXAMPLE"
        echo "==============================================="
        echo "Environment:"
        echo "  Processes: ${NTASKS}"
        echo "  Python: $(python --version)"
        echo "  JAX Available: $(python -c 'try: import jax; print(True, f"({len(jax.devices())} devices)")
except: print(False)' 2>/dev/null || echo "False")"
        echo "  Current Directory: ${SCRIPT_DIR}"
        echo ""
        
        if [[ "${NTASKS}" -gt 1 ]]; then
            echo "📋 Execution Plan:"
            echo "  🔄 Distributed power spectrum calculation on all ${NTASKS} ranks"
            echo "  💾 Real simulation data I/O with MPI coordination"
            echo "  🧮 Multi-GPU JAX FFT acceleration"
        else
            echo "📋 Execution Plan:"
            echo "  🖥️  Single-process power spectrum calculation"
        fi
        echo ""
        
        echo "🧪 Starting example with args: ${EXAMPLE_ARGS[*]}"
        echo ""
    fi

    # Set up Python environment
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

    # Run the example
    python examples/power_spectrum_real_data.py "${EXAMPLE_ARGS[@]}"

    # Only rank 0 prints completion message
    if [[ "${PROCID}" -eq 0 ]]; then
        echo ""
        echo "✅ Power spectrum example completed!"
        
        if [[ "${NTASKS}" -gt 1 ]]; then
            echo "🔍 Summary:"
            echo "  - Real simulation data processed across ${NTASKS} ranks"
            echo "  - Distributed FFT calculation completed"
            echo "  - MPI synchronization completed"
        fi
    fi

else
    # We're not in SLURM - launch the SLURM job
    echo "🚀 Launching distributed SLURM job for power spectrum example..."
    echo "Parameters:"
    echo "  Nodes: ${SLURM_NODES}"
    echo "  Tasks: ${SLURM_NTASKS}"  
    echo "  CPUs per task: ${SLURM_CPUS_PER_TASK}"
    echo "  GPUs per node: ${SLURM_GPUS_PER_NODE}"
    echo "  Time limit: ${SLURM_TIME} minutes"
    
    if [[ ${#EXAMPLE_ARGS[@]} -gt 0 ]]; then
        echo "  Example args: ${EXAMPLE_ARGS[*]}"
    fi
    echo ""
    
    exec srun \
        -n "${SLURM_NTASKS}" \
        -c "${SLURM_CPUS_PER_TASK}" \
        --qos=interactive \
        -N "${SLURM_NODES}" \
        --time="${SLURM_TIME}" \
        -C gpu \
        -A cosmosim \
        --gpus-per-node=4 \
        --exclusive \
        "$0" "${EXAMPLE_ARGS[@]}" 2>&1 | tee examplelog
fi