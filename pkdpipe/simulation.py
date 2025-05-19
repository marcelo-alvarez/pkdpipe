import os
import time
import shutil
import subprocess
from string import Template
from typing import Dict, Any, Union
import pathlib

from .cosmology import Cosmology
from .cli import parsecommandline
from .config import (
    DEFAULT_RUN_DIR_BASE,
    DEFAULT_SCRATCH_DIR_BASE,
    DEFAULT_JOB_NAME_TEMPLATE,
    DEFAULT_SIMULATION_NAME,
    DEFAULT_COSMOLOGY_NAME,
    DEFAULT_REDSHIFT_TARGETS_STR,
    DEFAULT_NSTEPS_STR,
    SIMULATION_PRESETS,
    TEMPLATE_LIGHTCONE_PAR,
    TEMPLATE_SLURM_SH,
    SCRIPT_RUN_SH,
    DEFAULT_TIME_LIMIT,
    DEFAULT_NODES,
    DEFAULT_CPUS_PER_TASK,
    DEFAULT_GPUS_PER_NODE,
    DEFAULT_NGRID,
    DEFAULT_BOXSIZE_MPC_H,
    DEFAULT_REDSHIFT_FROM,
    DEFAULT_LPT_ORDER,
    DEFAULT_OUTPUT_INTERVAL,
    DirectoryError,
    TemplateError,
    JobSubmissionError,
    PKDGRAVEMAIL
)

def safemkdir(dir_path: str, force_remove_delay: int = 10) -> None:
    """
    Safely creates a directory. If the directory exists, prompts the user
    for removal.

    Args:
        dir_path: The path to the directory to be created.
        force_remove_delay: Seconds to wait before removing if user confirms.

    Raises:
        DirectoryError: If directory creation or removal fails.
    """
    if os.path.isdir(dir_path):
        response = input(
            f"'{dir_path}' already exists. Enter 'REMOVE' to delete it and continue, "
            f"or anything else to abort: "
        )
        if response == "REMOVE":
            print(f"Attempting to remove '{dir_path}' in {force_remove_delay} seconds...")
            try:
                for i in range(force_remove_delay, 0, -1):
                    print(f"Removing in {i}...", end='\r')
                    time.sleep(1)
                print("Removing now...          ")
                
                if len(pathlib.Path(dir_path).parts) < 3 and "*" not in dir_path:
                    raise DirectoryError(f"Path '{dir_path}' is too shallow for automatic removal for safety reasons.")
                
                shutil.rmtree(dir_path)
                print(f"Directory '{dir_path}' removed successfully.")
            except Exception as e:
                raise DirectoryError(f"Failed to remove existing directory '{dir_path}': {e}")
        else:
            raise DirectoryError(f"Directory creation aborted by user as '{dir_path}' already exists.")
    
    try:
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created successfully.")
    except Exception as e:
        raise DirectoryError(f"Failed to create directory '{dir_path}': {e}")

def copy_template_with_substitution(template_file_path: Union[str, pathlib.Path], 
                                    output_file_path: Union[str, pathlib.Path], 
                                    substitutions: Dict[str, Any]) -> None:
    """
    Reads a template file, performs substitutions, and writes to an output file.

    Args:
        template_file_path: Path to the template file.
        output_file_path: Path to the output file.
        substitutions: A dictionary of {placeholder: value} for substitution.

    Raises:
        TemplateError: If template processing or file operations fail.
    """
    try:
        with open(template_file_path, "r") as f_template:
            template_content = f_template.read()
        
        template = Template(template_content)
        rendered_content = template.substitute(substitutions)
        
        with open(output_file_path, 'w') as f_output:
            f_output.write(rendered_content)
        print(f"Successfully wrote rendered template to '{output_file_path}'")
    except FileNotFoundError as e:
        raise TemplateError(f"Template file not found: {template_file_path}. Error: {e}")
    except IOError as e:
        raise TemplateError(f"IO error processing template '{template_file_path}' to '{output_file_path}': {e}")
    except KeyError as e:
        raise TemplateError(f"Missing substitution key {e} for template '{template_file_path}'. Available keys: {list(substitutions.keys())}")
    except Exception as e:
        raise TemplateError(f"An unexpected error occurred during template processing for '{template_file_path}': {e}")

def get_default_simulation_parameters(sim_preset_name: str = DEFAULT_SIMULATION_NAME) -> Dict[str, Any]:
    """
    Constructs a dictionary of default simulation parameters, applying a preset if specified.

    Args:
        sim_preset_name: The name of the simulation preset to apply from config.SIMULATION_PRESETS.

    Returns:
        A dictionary containing all parameters for a simulation.
    """
    base_params = {
        'sbatch': False,
        'interact': False,
        'nodes': DEFAULT_NODES,
        'cpupert': DEFAULT_CPUS_PER_TASK,
        'gpupern': DEFAULT_GPUS_PER_NODE,
        'rundir': DEFAULT_RUN_DIR_BASE,
        'scrdir': DEFAULT_SCRATCH_DIR_BASE,
        'jobname_template': DEFAULT_JOB_NAME_TEMPLATE,
        'email': PKDGRAVEMAIL,
        'tlimit': DEFAULT_TIME_LIMIT,
        'simname': sim_preset_name,
        'cosmo': DEFAULT_COSMOLOGY_NAME,
        'scratch': False,
        'nGrid': DEFAULT_NGRID,
        'dBoxSize': DEFAULT_BOXSIZE_MPC_H,
        'dRedFrom': DEFAULT_REDSHIFT_FROM,
        'iLPT': DEFAULT_LPT_ORDER,
        'dRedTo': DEFAULT_REDSHIFT_TARGETS_STR,
        'nSteps': DEFAULT_NSTEPS_STR,
        'iOutInterval': DEFAULT_OUTPUT_INTERVAL,
    }

    if sim_preset_name in SIMULATION_PRESETS:
        preset_overrides = SIMULATION_PRESETS[sim_preset_name]
        for key, value in preset_overrides.items():
            if key in base_params:
                base_params[key] = value
            else:
                print(f"Warning: Preset '{sim_preset_name}' contains key '{key}' not in base_params.")
    elif sim_preset_name != DEFAULT_SIMULATION_NAME:
        print(f"Warning: Simulation preset '{sim_preset_name}' not found. Using defaults for '{DEFAULT_SIMULATION_NAME}'.")
        if DEFAULT_SIMULATION_NAME in SIMULATION_PRESETS:
             base_params.update(SIMULATION_PRESETS[DEFAULT_SIMULATION_NAME])

    cli_structured_params = {}
    for param_name, default_value in base_params.items():
        param_type = type(default_value)
        if param_type == str and default_value.isdigit():
            param_type = int
        elif param_type == str and default_value.replace('.', '', 1).isdigit():
             param_type = float
        elif isinstance(default_value, bool):  # Changed from 'default_value in [True, False]'
            param_type = bool
        
        cli_structured_params[param_name] = {
            'val': default_value,
            'type': param_type,
            'desc': f"{param_name} (default: {default_value})"
        }
    return cli_structured_params

class Simulation:
    """
    Manages the setup, configuration, and creation of pkdgrav3 simulation runs.
    """
    def __init__(self, params: Dict[str, Any] | None = None, parse_cli_args: bool = False, **kwargs):
        """
        Initializes the Simulation instance.

        Args:
            params: A dictionary of parameters to override defaults. 
                    If parse_cli_args is True, these are ignored initially.
            parse_cli_args: If True, parameters will be parsed from the command line
                            using `pkdpipe.cli.parsecommandline`. Defaults are fetched
                            using `get_default_simulation_parameters`.
            **kwargs: Additional keyword arguments to override parameters if `params` is None
                      and `parse_cli_args` is False.
        """
        if parse_cli_args:
            from .cli import parsecommandline
            default_params_for_cli = get_default_simulation_parameters(DEFAULT_SIMULATION_NAME)
            self.params = parsecommandline(default_params_for_cli, description='Create and manage pkdgrav3 simulations.')
        elif params is not None:
            self.params = params
        else:
            default_params_values = {k: v['val'] for k, v in get_default_simulation_parameters(kwargs.get('simname', DEFAULT_SIMULATION_NAME)).items()}
            default_params_values.update(kwargs)
            self.params = default_params_values
        
        required_keys = ['rundir', 'jobname_template', 'nGrid', 'dBoxSize', 'gpupern', 'simname', 'cosmo']
        if not all(key in self.params for key in required_keys):
            print("Warning: Initial parameters seem incomplete. Re-fetching defaults.")
            fresh_defaults = {k: v['val'] for k, v in get_default_simulation_parameters(self.params.get('simname', DEFAULT_SIMULATION_NAME)).items()}
            for req_key in required_keys:
                if req_key not in self.params:
                    self.params[req_key] = fresh_defaults[req_key]
            for k, v_dict in get_default_simulation_parameters(self.params.get('simname', DEFAULT_SIMULATION_NAME)).items():
                if k not in self.params:
                    self.params[k] = v_dict['val']

        self._resolve_job_name()

    def _resolve_job_name(self) -> None:
        """
        Resolves the job name using the jobname_template and current parameters.
        Stores the result in self.params['jobname_actual'].
        """
        template_str = self.params.get('jobname_template', DEFAULT_JOB_NAME_TEMPLATE)
        try:
            template_keys = {
                'nGrid': self.params.get('nGrid'),
                'dBoxSize': self.params.get('dBoxSize'),
                'gpupern': self.params.get('gpupern')
            }
            valid_template_keys = {k: v for k, v in template_keys.items() if v is not None}
            
            if not all(k in valid_template_keys for k in ['nGrid', 'dBoxSize', 'gpupern']):
                 print(f"Warning: Not all keys for jobname template '{template_str}' are available. Using fallback name.")
                 self.params['jobname_actual'] = f"pkdpipe_run_{self.params.get('simname', 'default')}"

            else:
                jobname_template = Template(template_str)
                self.params['jobname_actual'] = jobname_template.substitute(valid_template_keys)

        except KeyError as e:
            print(f"Warning: Missing key {e} for job name template '{template_str}'. Using fallback name.")
            self.params['jobname_actual'] = f"pkdpipe_run_{self.params.get('simname', 'default')}"
        except TypeError as e:
            print(f"Warning: Type error during job name templating (likely a None value): {e}. Using fallback name.")
            self.params['jobname_actual'] = f"pkdpipe_run_{self.params.get('simname', 'default')}"

    def _setup_directories_and_paths(self) -> Dict[str, pathlib.Path]:
        """
        Sets up job directories and defines key file paths.

        Returns:
            A dictionary of important paths (job_dir, job_scr_dir, etc.).
        
        Raises:
            DirectoryError: If directory creation fails.
        """
        actual_job_name = self.params.get('jobname_actual', f"pkdpipe_fallback_jobname_{time.time():.0f}")
        base_run_dir = pathlib.Path(self.params['rundir'])
        job_dir = base_run_dir / actual_job_name

        paths = {
            "job_dir": job_dir,
            "parfile": job_dir / f"{actual_job_name}.par",
            "slurmfile": job_dir / f"{actual_job_name}.sbatch",
            "logfile": job_dir / f"{actual_job_name}.log",
            "transferfile": job_dir / f"{actual_job_name}.transfer",
            "runscript_dest": job_dir / SCRIPT_RUN_SH.name,
        }

        safemkdir(str(job_dir))

        if self.params.get('scratch', False):
            base_scratch_dir = pathlib.Path(self.params['scrdir'])
            job_scr_dir = base_scratch_dir / actual_job_name
            safemkdir(str(job_scr_dir))
            paths["ach_out_name_effective"] = str(job_scr_dir)
            link_path = job_dir / "scratch_output"
            if not link_path.exists():
                 os.symlink(job_scr_dir, link_path, target_is_directory=True)
            print(f"Using scratch directory: {job_scr_dir}. Symlinked at {link_path}")
        else:
            paths["ach_out_name_effective"] = str(job_dir)

        return paths

    def _generate_cosmology_transfer_file(self, cosmo_params: Dict[str, Any], transfer_file_path: pathlib.Path) -> None:
        """
        Initializes Cosmology and writes the transfer function file.
        """
        cosmo_instance = Cosmology(
            cosmology=cosmo_params.get('cosmo', DEFAULT_COSMOLOGY_NAME), 
            **cosmo_params
        )
        cosmo_instance.writetransfer(str(transfer_file_path))
        print(f"Cosmology transfer file written to '{transfer_file_path}'")
        
        self.params['effective_h'] = cosmo_instance.params.get('h')
        self.params['effective_omegam'] = cosmo_instance.params.get('omegam')
        self.params['effective_omegal'] = cosmo_instance.params.get('omegal')
        self.params['effective_sigma8'] = cosmo_instance.params.get('sigma8')
        self.params['effective_ns'] = cosmo_instance.params.get('ns')

    def _generate_parameter_file(self, paths: Dict[str, pathlib.Path]) -> None:
        """
        Generates the pkdgrav3 .par file.
        """
        h_val = self.params.get('effective_h')
        if h_val is None or h_val == 0:
            raise ValueError(
                f"Critical parameter 'effective_h' is missing, None, or zero ({h_val}). "
                "Cannot proceed with LCP calculations for .par file generation. "
                "Please check cosmology setup."
            )

        temp_cosmo_for_chi = Cosmology(cosmology=self.params.get('cosmo', DEFAULT_COSMOLOGY_NAME), h=h_val)
        box_size_mpc_h = float(self.params['dBoxSize'])
        chi_for_lcp_mpc = box_size_mpc_h / h_val
        dRedshiftLCP_val = temp_cosmo_for_chi.chi2z(chi_for_lcp_mpc) + 0.01

        # Retrieve effective parameters, providing defaults if they are None
        effective_ns = self.params.get('effective_ns')
        effective_h_par = self.params.get('effective_h') # h_val is already validated for calculation
        effective_omegam = self.params.get('effective_omegam')
        effective_omegal = self.params.get('effective_omegal')
        effective_sigma8 = self.params.get('effective_sigma8')

        par_substitutions = {
            "achOutName": paths["ach_out_name_effective"],
            "achTfFile": str(paths["transferfile"]),
            "dRedshiftLCP": f"{dRedshiftLCP_val:.2f}",
            "dSpectral": f"{effective_ns if effective_ns is not None else 0.96:0.6f}",
            "h": f"{effective_h_par if effective_h_par is not None else 0.7:0.6f}",
            "dOmega0": f"{effective_omegam if effective_omegam is not None else 0.3:0.6f}",
            "dLambda": f"{effective_omegal if effective_omegal is not None else 0.7:0.6f}",
            "dSigma8": f"{effective_sigma8 if effective_sigma8 is not None else 0.8:0.6f}",
            "dRedFrom": f"{self.params['dRedFrom']}",
            "nGrid": f"{self.params['nGrid']:<4}",
            "dBoxSize": f"{self.params['dBoxSize']:<4}",
            "iLPT": f"{self.params['iLPT']}",
            "dRedTo": self.params['dRedTo'],
            "nSteps": self.params['nSteps'],
            "iOutInterval": f"{self.params['iOutInterval']}"
        }
        copy_template_with_substitution(TEMPLATE_LIGHTCONE_PAR, paths["parfile"], par_substitutions)

    def _generate_slurm_script(self, paths: Dict[str, pathlib.Path]) -> None:
        """
        Generates the SLURM batch script.
        """
        run_command = f'{paths["runscript_dest"]} {paths["parfile"]} {paths["logfile"]}'
        slurm_substitutions = {
            "tlimit": self.params['tlimit'],
            "email": self.params['email'],
            "nodes": self.params['nodes'],
            "cpupert": self.params['cpupert'],
            "gpupern": self.params['gpupern'],
            "jobname": self.params.get('jobname_actual', 'pkdpipe_job'),
            "runcmd": run_command
        }
        copy_template_with_substitution(TEMPLATE_SLURM_SH, paths["slurmfile"], slurm_substitutions)

    def _copy_run_script(self, paths: Dict[str, pathlib.Path]) -> None:
        """Copies the master run.sh script to the job directory."""
        try:
            shutil.copy(SCRIPT_RUN_SH, paths["runscript_dest"])
            os.chmod(paths["runscript_dest"], 0o755)
            print(f"Run script copied to '{paths['runscript_dest']}' and made executable.")
        except Exception as e:
            raise FileNotFoundError(f"Failed to copy run script from '{SCRIPT_RUN_SH}' to '{paths['runscript_dest']}': {e}")

    def _handle_job_submission(self, paths: Dict[str, pathlib.Path]) -> None:
        """
        Handles job submission (sbatch or interactive message).
        """
        run_command_display = f'{paths["runscript_dest"]} {paths["parfile"]} {paths["logfile"]}'
        if self.params.get('sbatch', False):
            try:
                subprocess.run(["sbatch", str(paths["slurmfile"])], check=True, cwd=paths["job_dir"])
                print(f"Job submitted via sbatch: {paths['slurmfile']}")
            except subprocess.CalledProcessError as e:
                raise JobSubmissionError(f"sbatch submission failed for '{paths['slurmfile']}': {e}")
            except FileNotFoundError:
                raise JobSubmissionError("sbatch command not found. Ensure SLURM is installed and in PATH.")
        elif self.params.get('interact', False):
            print("Interactive mode: Job files created.")
            print(f"To run interactively, navigate to '{paths['job_dir']}' and execute:")
            print(f"  {run_command_display}")
        else:
            print("Simulation files created. No submission requested.")
            print(f"To submit manually: sbatch {paths['slurmfile']}")
            print(f"Or run interactively in '{paths['job_dir']}': {run_command_display}")

    def create(self) -> None:
        """
        Orchestrates the creation of the simulation directory, configuration files,
        and optionally submits the job.
        """
        try:
            paths = self._setup_directories_and_paths()
            
            self._generate_cosmology_transfer_file(self.params, paths["transferfile"])
            
            self._generate_parameter_file(paths)
            self._generate_slurm_script(paths)
            self._copy_run_script(paths)
            
            self._handle_job_submission(paths)
            
            print(f"\nSimulation '{self.params.get('jobname_actual')}' setup complete in '{paths['job_dir']}'.")

        except (DirectoryError, TemplateError, JobSubmissionError) as e:
            print(f"Error during simulation creation: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during simulation creation: {e}")