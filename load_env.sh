#!/bin/bash
# Environment loading script for pkdpipe project
# Usage: source ./load_env.sh

echo "Loading pkdgrav3 environment..."

# Load python module
module load python

# Activate mamba environment
mamba activate pkdgrav

echo "Environment loaded successfully."
echo "Python: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
