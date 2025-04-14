#!/bin/bash

parfile=$1
logfile=$2
sargs=""
interactive=0
if [ ! -z $3 ] ; then
    N=1   ; if [ ! -z $4 ] ; then N=$4 ; fi
    g=4   ; if [ ! -z $5 ] ; then g=$5 ; fi
    c=128 ; if [ ! -z $6 ] ; then c=$6 ; fi
    sargs="--qos=interactive -N $N --time=120 -C gpu -A cosmosim --gpus-per-node $g -c $c"
    interactive=1
fi

exe=/pscratch/sd/j/jderose/pkdgrav3/build/pkdgrav3
condalib=/global/homes/m/malvarez/.conda/envs/pkdgrav/lib

module load python
mamba activate pkdgrav
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$condalib
module unload python
export OMP_NUM_THREADS=64
cmd="stdbuf -oL srun $sargs $exe $parfile"
if [ $interactive == 0 ] ; then
    $cmd | tee $logfile &
    wait
else
    $cmd > $logfile &
fi

