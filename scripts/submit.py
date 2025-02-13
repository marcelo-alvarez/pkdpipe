import numpy as np
import sys
import pathlib
import subprocess
import os
import shutil
from string import Template

dpath    = f"{os.getenv('SCRATCH')}/pkdgrav3/scaling-tests"
djobname = "N{ngrid}-L{lbox}-{nodes*4}gpus"
demail   = "${pkdgravemail}"

templatedir = f"{str(pathlib.Path(__file__).parent.resolve())}/../templates"
scriptdir   = f"{str(pathlib.Path(__file__).parent.resolve())}/../scripts"
datadir     = f"{str(pathlib.Path(__file__).parent.resolve())}/../data"

partemp   = f"{templatedir}/lightcone.par"
slurmtemp = f"{templatedir}/pkdgrav.sh"
transfer  = f"{datadir}/euclid_z0_transfer_combined.dat"
runtemp   = f"{scriptdir}/run.sh"

# command line parameters
cparams = {
    'ngrid'  : {'val' :       1400, 'type' :   int, 'desc' : 'cube root particle number'},
    'lbox'   : {'val' :       1050, 'type' : float, 'desc' : 'boxsize [Mpc/h]'},
    'zmax'   : {'val' :       0.38, 'type' : float, 'desc' : 'zmax of lightcone'},
    'nodes'  : {'val' :          2, 'type' :   int, 'desc' : 'number of nodes'},
    'rundir' : {'val' :      dpath, 'type' :   str, 'desc' : 'path for runs'},
    'jobname': {'val' :   djobname, 'type' :   str, 'desc' : 'directory name for run'},
    'email'  : {'val' :     demail, 'type' :   str, 'desc' : 'email for slurm notifications'},
    'tlimit' : {'val' : '48:00:00', 'type' :   str, 'desc' : 'time limit'}}

def copytemplate(templatefile,outfile,data):
    with open(templatefile, "r") as file:
        templines = file.read()
    template = Template(templines)
    lines = template.substitute(data)
    with open(outfile, 'w') as par:
        par.write(lines)

def parsecommandline():
    import argparse

    parser   = argparse.ArgumentParser(description='Commandline interface to pkdpipe')

    for param in cparams:
        pdval = cparams[param]['val']
        ptype = cparams[param]['type']
        pdesc = cparams[param]['desc']
        parser.add_argument('--'+param, default=pdval, help=f'{pdesc} [{pdval}]', type=ptype)

    return vars(parser.parse_args())

params = parsecommandline()

ngrid   = params['ngrid']
lbox    = params['lbox']
zmax    = params['zmax']
nodes   = params['nodes']
rundir  = params['rundir']
jobname = params['jobname']
tlimit  = params['tlimit']
email   = params['email']

gpus = nodes * 4

if jobname == djobname:
    jobname = f"N{ngrid}-L{lbox}-{gpus}gpus"
    
if email == demail:
    email = os.getenv('pkdgravemail')

jobdir  = f"{rundir}/{jobname}"
outdir  = f"{jobdir}/output"

if os.path.isdir(jobdir):
    print(f"{jobdir} already exists; exiting...")
    sys.exit(1)

os.makedirs(outdir)

shutil.copy( transfer, jobdir)

runscript = f"{jobdir}/run.sh"
shutil.copy(runtemp, runscript)

# parameter file
parfile = f"{jobdir}/{jobname}.par"
copytemplate(partemp,parfile,{
            "jobname" : jobname,
            "jobdir"  : jobdir,
            "outdir"  : outdir,
            "zmax"    : zmax,
            "ngrid"   : ngrid,
            "lbox"    : lbox
    })

# slurm batch script
slurmfile = f"{jobdir}/launch.sh"
copytemplate(slurmtemp,slurmfile,{
            "tlimit"    : tlimit,
            "email"     : email,
            "nodes"     : nodes,
            "jobname"   : jobname,
            "runscript" : runscript,
            "lbox"      : lbox,
            "parfile"   : parfile
    })

subprocess.call(f"sbatch {slurmfile}", shell=True)

# cat $templtdir/lightcone.par | sed -e "s,_NGRID_REPLACE_,$ngrid,g" \
#                              | sed -e "s,_LBOX_REPLACE_,$lbox,g" \
#                              | sed -e "s,_ZMAX_REPLACE_,$zmax,g" \
#                              | sed -e "s,_JOBNAME_REPLACE_,$jobname,g" > $parfile

# cd $jobdir
# if [ $int -eq 0 ] ; then
#     cat $templtdir/pkdgrav.sh | sed -e "s,_RUNSCRIPT_REPLACE_,$runscript,g" \
#                               | sed -e "s,_NODES_REPLACE_,$nodes,g" \
#                               | sed -e "s,_TLIMIT_REPLACE_,$tlimit,g" \
#                               | sed -e "s,_JOBNAME_REPLACE_,$jobname,g" \
#                               | sed -e "s,_PARFILE_REPLACE_,$parfile,g" > pkdgrav.sh
#     jobid=$(sbatch pkdgrav.sh | awk '{print $4}')
#     echo "submitted $jobid to queue"
#     cp pkdgrav.sh pkdgrav-$jobid.sh
# else
#     source $runscript $nodes $parfile interactive | tee log-$jobname
# fi
