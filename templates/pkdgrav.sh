#!/bin/bash
#SBATCH -t ${tlimit}
#SBATCH --qos regular
#SBATCH -A cosmosim
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${email}
#SBATCH -L SCRATCH
#SBATCH -C gpu
#SBATCH --gpus-per-node=4

#SBATCH -N ${nodes}
#SBATCH -J ${jobname}
#SBATCH --output=log-${jobname}.%j.oe

source ${runscript} ${nodes} ${parfile}
