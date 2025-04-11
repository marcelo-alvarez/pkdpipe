exe=/pscratch/sd/j/jderose/pkdgrav3/build/pkdgrav3
condalib=/global/homes/m/malvarez/.conda/envs/pkdgrav/lib

nodes=$1
parfile=$2
args=""
if [ ! -z $3 ] ; then 
    args="--qos=interactive -N $nodes --time=120 -C gpu -A cosmosim --gpus-per-node=4 --exclusive" 
fi

module load python
mamba activate pkdgrav
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$condalib
module unload python
stdbuf -oL nohup srun $args $exe $parfile
