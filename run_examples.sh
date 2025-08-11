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

# Function to generate metadata log filename
generate_log_filename() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local git_hash=$(git rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
    
    # Check for uncommitted changes to tracked files only (same as Python script)
    if ! git diff-index --quiet HEAD 2>/dev/null; then
        git_hash="${git_hash}-dirty"
    fi
    
    # Extract variant from example args
    local variant="default"
    for i in "${!EXAMPLE_ARGS[@]}"; do
        if [[ "${EXAMPLE_ARGS[i]}" == "--variant" ]] && [[ $((i+1)) -lt ${#EXAMPLE_ARGS[@]} ]]; then
            variant="${EXAMPLE_ARGS[$((i+1))]}"
            break
        fi
    done
    
    # Extract assignment from example args
    local assignment="cic"
    for i in "${!EXAMPLE_ARGS[@]}"; do
        if [[ "${EXAMPLE_ARGS[i]}" == "--assignment" ]] && [[ $((i+1)) -lt ${#EXAMPLE_ARGS[@]} ]]; then
            assignment="${EXAMPLE_ARGS[$((i+1))]}"
            break
        fi
    done
    
    # Extract ngrid from example args
    local ngrid="512"
    for i in "${!EXAMPLE_ARGS[@]}"; do
        if [[ "${EXAMPLE_ARGS[i]}" == "--ngrid" ]] && [[ $((i+1)) -lt ${#EXAMPLE_ARGS[@]} ]]; then
            ngrid="${EXAMPLE_ARGS[$((i+1))]}"
            break
        fi
    done
    
    echo "examplelog_ntasks${NTASKS_ARG}_${variant//-/_}_ngrid${ngrid}_${assignment}_${git_hash}_${timestamp}"
}

# Parse command line arguments  
NODES=1
TIME=15
NTASKS_ARG=4
CPUS_PER_TASK=32
GPUS_PER_NODE=4
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
            NODES="${1#*=}"
            shift
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --time=*)
            TIME="${1#*=}"
            shift
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --ntasks=*)
            NTASKS_ARG="${1#*=}"
            shift
            ;;
        --ntasks)
            NTASKS_ARG="$2"
            shift 2
            ;;
        --cpus-per-task=*)
            CPUS_PER_TASK="${1#*=}"
            shift
            ;;
        --cpus-per-task)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --gpus-per-node=*)
            GPUS_PER_NODE="${1#*=}"
            shift
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
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

# Launch the SLURM job
echo "🚀 Launching distributed SLURM job for power spectrum example..."
echo "Parameters:"
echo "  Nodes: ${NODES}"
echo "  Tasks: ${NTASKS_ARG}"  
echo "  CPUs per task: ${CPUS_PER_TASK}"
echo "  GPUs per node: ${GPUS_PER_NODE}"
echo "  Time limit: ${TIME} minutes"

if [[ ${#EXAMPLE_ARGS[@]} -gt 0 ]]; then
    echo "  Example args: ${EXAMPLE_ARGS[*]}"
fi
echo ""

# Debug: Print exact srun command
echo "DEBUG: Executing srun with:"
echo "  -n ${NTASKS_ARG} (number of tasks)"
echo "  -N ${NODES} (number of nodes)"
echo "  -c ${CPUS_PER_TASK} (CPUs per task)"
echo "  --time=${TIME} (time limit in minutes)"
echo "  --gpus-per-node=4"
echo "  --qos=interactive -C gpu -A cosmosim --exclusive"
echo ""

# Set up Python path for the job
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# Generate unique log filename
LOGFILE=$(generate_log_filename)
echo "📝 Log file: ${LOGFILE}"
echo ""

exec srun \
    -n "${NTASKS_ARG}" \
    -c "${CPUS_PER_TASK}" \
    --qos=interactive \
    -N "${NODES}" \
    --time="${TIME}" \
    -C gpu \
    -A cosmosim \
    --gpus-per-node=4 \
    --exclusive \
    bash -c "source /global/cfs/cdirs/cosmosim/slac/malvarez/pkdgrav3/env/loadenv.sh && source ${SCRIPT_DIR}/load_env.sh && cd ${SCRIPT_DIR} && python examples/power_spectrum_real_data.py ${EXAMPLE_ARGS[*]}" 2>&1 | tee "${LOGFILE}"