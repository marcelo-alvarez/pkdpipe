import numpy as np
import sys
import pathlib
import subprocess
import os
import shutil
from string import Template
from pkdpipe.cosmology import Cosmology
from pkdpipe.cli import parsecommandline

"""
pkdpipe simulation
"""

djobname = "N{ngrid}-L{dBoxSize}-{nodes*4}gpus"
demail   = "${pkdgravemail}"
dsimname = "lcone-small"

templatedir = f"{str(pathlib.Path(__file__).parent.resolve())}/../templates"
scriptdir   = f"{str(pathlib.Path(__file__).parent.resolve())}/../scripts"

partemp   = f"{templatedir}/lightcone.par"
slurmtemp = f"{templatedir}/pkdgrav.sh"
runtemp   = f"{scriptdir}/run.sh"

def copytemplate(templatefile,outfile,data):
    with open(templatefile, "r") as file:
        templines = file.read()
    template = Template(templines)
    lines = template.substitute(data)
    with open(outfile, 'w') as par:
        par.write(lines)

def get_simulation(simname):
    dpath    = f"{os.getenv('SCRATCH')}/pkdgrav3/runs"
    dcosmo = 'desi-dr2-planck-act-mnufree'
    redshiftlist = sorted([3. , 2.81388253, 2.6350418 , 2.46500347, 2.30360093,
       2.1496063 , 2.003003  , 1.86204923, 1.72851296, 1.60145682,
       1.48015873, 1.36406619, 1.25377507, 1.1486893 , 1.04834084,
       0.9527436 , 0.8615041 , 0.77462289, 0.69176112, 0.61290323,
       0.53751538, 0.46584579, 0.39742873, 0.33226752, 0.27000254,
       0.21065375, 0.15420129, 0.10035211, 0.04898773, 0.        ] + 
       [0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.33, 0.922, 0.955],reverse=True)
    ddRedTo = f'{[f'{z:0.4f}' for z in redshiftlist]}'.replace("'", "")
    dnSteps = f'{[f'{i}' for i in range(len(redshiftlist))]}'.replace("'", "")

    pkp_params = {
        'sbatch'  : {'val' :      False, 'type' :  bool, 'desc' : 'submit with sbatch; otherwise only create dir & files'},
        'simname' : {'val' :   dsimname, 'type' :   str, 'desc' : 'simulation parameter set name'},
        'cosmo'   : {'val' :     dcosmo, 'type' :   str, 'desc' : 'cosmology'}
    }
    job_params = {
        'nodes'   : {'val' :          2, 'type' :   int, 'desc' : 'number of nodes'},
        'gpuper'  : {'val' :          4, 'type' :   int, 'desc' : 'GPUs per node'},
        'rundir'  : {'val' :      dpath, 'type' :   str, 'desc' : 'path for runs'},
        'jobname' : {'val' :   djobname, 'type' :   str, 'desc' : 'directory name for run'},
        'email'   : {'val' :     demail, 'type' :   str, 'desc' : 'email for slurm notifications'},
        'tlimit'  : {'val' : '48:00:00', 'type' :   str, 'desc' : 'time limit'}
    }
    pkd_params = {
        'nGrid'        : {'val' :       1400, 'type' :   int, 'desc' : 'cube root particle number'},
        'dBoxSize'     : {'val' :       1050, 'type' : float, 'desc' : 'boxsize [Mpc/h]'},
        'dRedFrom'     : {'val' :         12, 'type' : float, 'desc' : 'starting redshift'},
        'iLPT'         : {'val' :          3, 'type' :   int, 'desc' : 'LPT order for ICs'},
        'dRedTo'       : {'val' :    ddRedTo, 'type' :   str, 'desc' : 'output redshifts'},
        'nSteps'       : {'val' :    dnSteps, 'type' :   str, 'desc' : 'output steps'},
        'iOutInterval' : {'val' :          1, 'type' :   int, 'desc' : 'snapshot interval'}
    }
    params = pkp_params | job_params | pkd_params

    # only lightcone tests are implemented so far
    if simname == 'lcone-medium':
        pass
    elif simname == 'lcone-small':
        params['dBoxSize']['val']  = 525
        params['nGrid']['val'] = 700
        params['nodes']['val'] = 1
    elif simname == 'lcone-large':
        params['dBoxSize']['val']  = 2100
        params['nGrid']['val'] = 2800
        params['nodes']['val'] = 16
    return params

class Simulation:

    '''Simulation'''
    def __init__(self, **kwargs):

        self.simname = kwargs.get('simname', dsimname)

        # initialize parameters using parser
        self.params = parsecommandline(get_simulation(self.simname))

        if kwargs.get('parse', False):
            # if requested parse again to retrieve updated simulation name from command-line
            self.simname = self.params['simname']
            self.params = parsecommandline(get_simulation(self.simname))

        # update parameters via kwargs
        for key in self.params:
            self.params[key] = kwargs.get(key,self.params[key])

    def update_params(self, params):
        self.params = get_simulation(params['simname'])
        print(self.params['simname'])
        for key in params:
            self.params[key] = params[key]

    def submit(self):

        nGrid    = self.params['nGrid']
        dBoxSize = self.params['dBoxSize']
        nodes    = self.params['nodes']
        rundir   = self.params['rundir']
        jobname  = self.params['jobname']
        tlimit   = self.params['tlimit']
        email    = self.params['email']
        sbatch   = self.params['sbatch']

        gpus = nodes * self.params['gpuper']

        if jobname == djobname:
            jobname = f"N{ngrid:04d}-L{dBoxSize}-{gpus:03d}gpus"
            
        if email == demail:
            email = os.getenv('pkdgravemail')

        jobdir  = f"{rundir}/{jobname}"
        outdir  = f"{jobdir}/output"

        achTfFile = f"{jobdir}/{jobname}-transfer.dat"

        if os.path.isdir(jobdir):
            response = input(f"{jobdir} already exists; enter REMOVE if you want to delete it\n")
            if response == "REMOVE":
                nsleep=10
                for i in range(nsleep):
                    print(f"will delete {jobdir} in {nsleep-i} seconds...",end='\r')
                    subprocess.call(f"sleep 1", shell=True)
                print()
                if len(jobdir) > 10:
                    subprocess.call(f"rm -rf {jobdir}", shell=True)
                else:
                    print(f"exiting...")
                    sys.exit(1)
            else:
                print(f"exiting...")
                sys.exit(1)

        os.makedirs(outdir)

        runscript = f"{jobdir}/run.sh"
        shutil.copy(runtemp, runscript)

        # get cosmology
        cosmo  = Cosmology(cosmology='desi-dr2-planck-act-mnufree')

        # get maximum redshift
        dRedshiftLCP = f"{cosmo.chi2z(dBoxSize/cosmo.params['h'])+0.01:0.2f}"

        # write transfer file
        cosmo.writetransfer(achTfFile)

        # parameter file
        parfile = f"{jobdir}/{jobname}.par"
        copytemplate(partemp,parfile,{
                    "achOutName"  : jobdir,
                    "achTfFile"   : achTfFile,
                    "dRedshiftLCP": dRedshiftLCP,
                    "dSpectral"   : f"{cosmo.params['ns']:0.6f}",
                    "h"           : f"{cosmo.params['h']:0.6f}",
                    "dOmega0"     : f"{cosmo.params['omegam']:0.6f}",
                    "dLambda"     : f"{cosmo.params['omegal']:0.6f}",
                    "dSigma8"     : f"{cosmo.params['sigma8']:0.6f}",
                    "dRedFrom"    : f"{self.params['dRedFrom']}",
                    "ngrid"       : f"{self.params['nGrid']:<4}",
                    "dBoxSize"    : f"{self.params['dBoxSize']:<4}",
                    "iLPT"        : f"{self.params['iLPT']}",
                    "dRedTo"      : f"{self.params['dRedTo']}",
                    "nSteps"      : f"{self.params['nSteps']}",
                    "iOutInterval": f"{self.params['iOutInterval']}"
        })

        # slurm batch script
        slurmfile = f"{jobdir}/launch.sh"
        copytemplate(slurmtemp,slurmfile,{
                    "tlimit"    : tlimit,
                    "email"     : email,
                    "nodes"     : nodes,
                    "jobname"   : jobname,
                    "runscript" : runscript,
                    "dBoxSize"      : dBoxSize,
                    "parfile"   : parfile
            })
        print(f"created {slurmfile}")

        if sbatch:
            subprocess.call(f"sbatch {slurmfile}", shell=True)
