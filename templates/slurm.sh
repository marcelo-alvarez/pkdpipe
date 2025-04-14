#!/bin/bash
#SBATCH -t ${tlimit}
#SBATCH --qos regular
#SBATCH -A cosmosim
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${email}
#SBATCH -L SCRATCH
#SBATCH -C gpu
#SBATCH --gpus-per-node=${gpupern}
#SBATCH --cpus-per-task=${cpupert}
#SBATCH -N ${nodes}
#SBATCH -J ${jobname}
#SBATCH --output=log-${jobname}.%j.oe

${runcmd}s
