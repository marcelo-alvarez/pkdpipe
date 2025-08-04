#!/bin/bash
# Comprehensive Test Suite for pkdpipe
#
# This script automatically handles both serial and distributed test execution:
# - If run normally: Launches distributed SLURM job automatically
# - If already in SLURM: Runs MPI-aware pytest directly 
# - Tests are intelligently routed based on their requirements
#
# Usage:
#   ./run_comprehensive_tests.sh                    # Automatic distributed testing (REQUIRED for validation)
#   ./run_comprehensive_tests.sh --serial           # Force serial mode only
#   ./run_comprehensive_tests.sh --debug            # Enable verbose DEBUG output
#   ./run_comprehensive_tests.sh --time=30          # Custom SLURM parameters
#   ./run_comprehensive_tests.sh --test=test_name   # Run specific test only (for debugging)

set -e  # Exit on any error

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments
FORCE_SERIAL=false
DEBUG_MODE=false
SLURM_NODES=1
SLURM_TIME=10
SLURM_NTASKS=2
SLURM_CPUS_PER_TASK=32
SLURM_GPUS_PER_TASK=2
SPECIFIC_TEST=""

# Common pytest arguments used by both serial and distributed modes
COMMON_PYTEST_ARGS=(
    "tests/"
    "-v"                   # Verbose output
    "-s"                   # Allow print statements / no capture
    "--tb=short"           # Shorter traceback format
    "--strict-markers"     # Enforce marker validation
    "--color=yes"          # Colored output
)

while [[ $# -gt 0 ]]; do
    case $1 in
        --serial)
            FORCE_SERIAL=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
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
        --test=*)
            SPECIFIC_TEST="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
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
    # We're already in a SLURM job - load environment and run tests directly
    source /global/cfs/cdirs/cosmosim/slac/malvarez/pkdgrav3/env/loadenv.sh
    source ./load_env.sh
    
    NTASKS=${SLURM_NTASKS:-1}
    PROCID=${SLURM_PROCID:-0}
    
    if [[ "${NTASKS}" -gt 1 ]]; then
        echo "🚀 Running comprehensive tests in MPI mode: ${NTASKS} processes, rank ${PROCID}"
    else
        echo "🖥️  Running comprehensive tests in single-process mode"
    fi
    
    # Only rank 0 prints the header
    if [[ "${PROCID}" -eq 0 ]]; then
        echo "==============================================="
        echo "PKDPIPE COMPREHENSIVE TEST SUITE"
        echo "==============================================="
        echo "Environment:"
        echo "  Processes: ${NTASKS}"
        echo "  Python: $(python --version)"
        echo "  JAX Available: $(python -c 'try: import jax; print(True, f"({len(jax.devices())} devices)")
except: print(False)' 2>/dev/null || echo "False")"
        echo "  Current Directory: ${SCRIPT_DIR}"
        echo ""
        
        if [[ "${NTASKS}" -gt 1 ]]; then
            echo "📋 Test Execution Plan:"
            echo "  ✅ Serial tests (campaign, config): Rank 0 only"
            echo "  🔄 Distributed tests (power spectrum, JAX): All ranks"  
            echo "  🎯 Default tests: Rank 0 only"
        else
            echo "📋 Test Execution Plan:"
            echo "  🖥️  All tests will run in single-process mode"
        fi
        echo ""
    fi

    # Set up Python environment
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
    
    # Export debug mode for tests to use
    export PKDPIPE_DEBUG_MODE="${DEBUG_MODE}"

    # Use common pytest args (defined below)
    PYTEST_ARGS=("${COMMON_PYTEST_ARGS[@]}")
    
    # Add specific test if provided
    if [[ -n "${SPECIFIC_TEST}" ]]; then
        PYTEST_ARGS=("-k" "${SPECIFIC_TEST}" "${PYTEST_ARGS[@]}")
    fi

    # Add distributed-specific options in MPI mode
    if [[ "${NTASKS}" -gt 1 ]]; then
        PYTEST_ARGS+=(
            "--capture=no"     # Don't capture output (better for distributed debugging)
            "-s"               # Don't capture stdout/stderr
        )
    fi

    # Only rank 0 prints test start message
    if [[ "${PROCID}" -eq 0 ]]; then
        echo "🧪 Starting pytest with args: ${PYTEST_ARGS[*]}"
        echo ""
    fi

    # Run the tests
    python -m pytest "${PYTEST_ARGS[@]}"

    # Only rank 0 prints completion message
    if [[ "${PROCID}" -eq 0 ]]; then
        echo ""
        echo "✅ Comprehensive test suite completed!"
        
        if [[ "${NTASKS}" -gt 1 ]]; then
            echo "🔍 Test Summary:"
            echo "  - Serial tests executed on rank 0"
            echo "  - Distributed tests executed on all ${NTASKS} ranks"
            echo "  - MPI synchronization completed"
        fi
    fi

else
    # We're not in SLURM - launch the SLURM job
    if [[ "${FORCE_SERIAL}" == "true" ]]; then
        echo "🖥️  Running in forced serial mode..."
        
        # Load Python environment for serial mode
        source ./load_env.sh
        
        echo "==============================================="
        echo "PKDPIPE COMPREHENSIVE TEST SUITE (SERIAL)"
        echo "==============================================="
        echo "Environment:"
        echo "  Python: $(python --version)"
        echo "  JAX Available: $(python -c 'try: import jax; print(True, f"({len(jax.devices())} devices)")
except: print(False)' 2>/dev/null || echo "False")"
        echo ""
        
        # Set up Python environment
        export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
        
        # Export debug mode for tests to use
        export PKDPIPE_DEBUG_MODE="${DEBUG_MODE}"
        
        # Use common pytest args (defined below)
        PYTEST_ARGS=("${COMMON_PYTEST_ARGS[@]}")
        
        # Add specific test if provided
        if [[ -n "${SPECIFIC_TEST}" ]]; then
            PYTEST_ARGS=("-k" "${SPECIFIC_TEST}" "${PYTEST_ARGS[@]}")
        fi
        
        # Run tests in serial mode
        python -m pytest "${PYTEST_ARGS[@]}"
        
        echo ""
        echo "✅ Serial test suite completed!"
    else
        echo "🚀 Launching distributed SLURM job for comprehensive testing..."
        echo "Parameters:"
        echo "  Nodes: ${SLURM_NODES}"
        echo "  Tasks: ${SLURM_NTASKS}"  
        echo "  CPUs per task: ${SLURM_CPUS_PER_TASK}"
        echo "  Time limit: ${SLURM_TIME} minutes"
        echo ""
        
        # Launch SLURM job with this same script
        # Pass through any arguments
        SLURM_ARGS=()
        if [[ -n "${SPECIFIC_TEST}" ]]; then
            SLURM_ARGS+=("--test=${SPECIFIC_TEST}")
        fi
        if [[ "${DEBUG_MODE}" == "true" ]]; then
            SLURM_ARGS+=("--debug")
        fi
        
        exec srun \
            -n "${SLURM_NTASKS}" \
            -c "${SLURM_CPUS_PER_TASK}" \
            --qos=interactive \
            -N "${SLURM_NODES}" \
            --time="${SLURM_TIME}" \
            -C gpu \
            -A cosmosim \
            --gpus-per-node="${SLURM_GPUS_PER_NODE}" \
            --exclusive \
            "$0" "${SLURM_ARGS[@]}" 2>&1 | tee testlog
    fi
fi