import os
import re
import sys
import logging
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, List, Callable, Any
import configparser

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# pyABC imports
from pyabc import sampler, LocalTransition, AdaptiveAggregatedDistance, AggregatedDistance, QuantileEpsilon, History, RV, Distribution
from pyabc.populationstrategy import AdaptivePopulationSize
from pyabc.storage import load_dict_from_json
from dask.distributed import Client, get_worker

# UQ PhysiCell imports
from uq_physicell import PhysiCell_Model
from uq_physicell.abc.utils import (
    insert_adaptive_weights_db,
    insert_metadata_db,
    insert_models_db,
)
from uq_physicell.abc.safeguards import CheckedABCSMC
from uq_physicell.abc.transitions import ScaledMultivariateNormalTransition
from uq_physicell.utils import run_replicate_serializable
from ..utils.sumstats import _convert_qoi_function_to_string


@dataclass
class ModelSpec:
    """One candidate model for ABC-SMC calibration / model selection.

    Attributes:
        name (str): Unique, filesystem-safe label. Drives the pyABC model-function
            name, the per-model IO subfolder (multi-model runs) and the DB keys.
        model_config (dict): PhysiCell model configuration. Requires ``ini_path`` and
            ``struc_name``; may also carry ``numReplicates``, ``input_folder`` and
            ``output_folder``.
        prior (Distribution): Prior over this model's parameters. Models may have
            different parameters / dimensionality.
        fixed_params (dict): Parameters held fixed for this model.
        num_replicates (int): Replicates per parameter set. Resolved from
            ``model_config`` / the ini file during context construction.
    """
    name: str
    model_config: dict
    prior: "Distribution"
    fixed_params: dict = field(default_factory=dict)
    num_replicates: Optional[int] = None


class CalibrationContext:
    """
    Context for Approximate Bayesian Computation (ABC) calibration using pyABC.

    This class encapsulates all necessary parameters and configurations for model calibration
    using ABC-SMC with sophisticated handling of multiple models, parallel computation,
    and adaptive strategies.

    Args:
        db_path (str): Path to the database file for storing and retrieving calibration results.
        obsData (str or dict): Path to observed data CSV file or dictionary containing observed data.
        obsData_columns (dict): Dictionary mapping QoI names to their corresponding columns in the observed data.
        model_config (dict): Configuration dictionary for the PhysiCell model, including paths and structure names.
        qoi_functions (dict): Dictionary of functions to compute quantities of interest (QoIs) from model outputs.
        qoi_def (dict): first-class object, that can be used in qoi_functions lambda string, mapped to their name.
        distance_functions (dict): Dictionary of distance functions with their weights for comparing model outputs to observed data.
        prior (Distribution): Distribution defining the prior distributions for parameters
        abc_options (dict): Options for ABC-SMC including population parameters, sampling strategies, and convergence criteria.
        logger (logging.Logger): Logger instance for logging messages during the calibration process.
    """
    
    def __init__(
        self,
        db_path: str,
        obsData: Union[str, dict],
        obsData_columns: dict,
        model_config: Optional[dict] = None,
        qoi_functions: Optional[dict] = None,
        distance_functions: Optional[dict] = None,
        prior: Optional[Distribution] = None,
        abc_options: Optional[dict] = None,
        qoi_def:dict={},
        logger: Optional[logging.Logger] = None
    ):
        """Initialize CalibrationContext with comprehensive validation and setup.

        Model selection: pass ``abc_options["models"]`` as a list of dicts, each
        REQUIRING ``name``, ``model_config`` and ``prior`` (optional: ``fixed_params``,
        ``num_replicates``). When present, the top-level ``model_config`` / ``prior``
        arguments are ignored. Otherwise the single-model form (top-level
        ``model_config`` + ``prior``, optional ``abc_options["model_name"]`` /
        ``abc_options["fixed_params"]``) is used.
        """
        # Core configuration
        self.db_path = db_path
        abc_options = dict(abc_options) if abc_options else {}
        if qoi_functions is None or distance_functions is None:
            raise ValueError("qoi_functions and distance_functions are required")
        # QOI_FUNCTIONS MUST BE STRINGS, BECAUSE THEY NEED TO BE SERIALIZABLE TO BE SAVED IN THE DATABASE AND USED IN THE DEFAULT AGGREGATION FUNCTION.
        self.qoi_functions = {key: _convert_qoi_function_to_string(value, key) if not isinstance(value, str) else value for key, value in qoi_functions.items()}
        self.qoi_def = qoi_def
        self.distance_functions = distance_functions
        self.abc_options = abc_options
        
        # Setup logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
        else:
            self.logger = logger

        # Load and validate observed data
        if isinstance(obsData, dict):
            self.dic_obsData = obsData
            self.obsData_path = None
        else:  # obsData is a path
            try:
                self.obsData_path = obsData
                self.dic_obsData = pd.read_csv(obsData).to_dict('list')
                # Replace column names according to obsData_columns mapping
                for qoi, column_name in obsData_columns.items():
                    if column_name in self.dic_obsData:
                        self.dic_obsData[qoi] = np.array(self.dic_obsData.pop(column_name), dtype=np.float64)
                    else:
                        raise ValueError(f"Column '{column_name}' not found in observed data.")
                self.logger.debug(f"Successfully loaded observed data from {obsData}")
            except Exception as e:
                self.logger.error(f"Error reading observed data from {obsData}: {e}")
                raise

        # ABC-SMC configuration
        self.max_populations = abc_options.get("max_populations", 20)
        self.max_simulations = abc_options.get("max_simulations", 1000)
        self.population_strategy = abc_options.get("population_strategy", "adaptive")
        self.min_population_size = abc_options.get("min_population_size", 100)
        self.max_population_size = abc_options.get("max_population_size", 500)
        self.epsilon_strategy = abc_options.get("epsilon_strategy", "quantile")
        self.epsilon_alpha = abc_options.get("epsilon_alpha", 0.5) # Quantile for epsilon threshold (e.g., 0.5 for median distance)
        
        # Transition and distance configuration
        self.transition_strategy = abc_options.get("transition_strategy", "multivariate")  # "local" or "multivariate"
        # Largest share of one model's particles that may get a zeroed importance weight in a generation
        self.max_zero_weight_fraction = abc_options.get("max_zero_weight_fraction", 0.05)
        self.adaptive_distance = abc_options.get("adaptive_distance", False)
        self.adaptive_distance_file = abc_options.get("adaptive_distance_file", None if not self.adaptive_distance else "adaptive_distance_log.json")
        self.convergence_check_func = abc_options.get("convergence_check_func", None)
        
        # Sampling configuration
        self.sampler_type = abc_options.get("sampler", "multicore")  # "dask" or "multicore"
        self.cluster_setup_func = abc_options.get("cluster_setup_func", None) # Function to setup Dask cluster
        self.num_workers = abc_options.get("num_workers", os.cpu_count())
        
        # Model configuration — normalize into a list of candidate ModelSpec objects
        self.models = self._build_model_specs(abc_options, model_config, prior)
        self.num_models = len(self.models)
        self.model_selection = self.num_models > 1
        # Worker math below uses the largest replicate count across candidate models
        self.num_replicates = max(spec.num_replicates for spec in self.models)
        self.summary_function = abc_options.get("summary_function", None)
        self.aggregation_func = abc_options.get("custom_aggregation_func", self._default_aggregation_func)
        self.custom_run_single_replicate_func = abc_options.get("custom_run_single_replicate_func", None)


        # Parallelization setup
        self._setup_parallelization()
        
        # Validate configuration
        self._validate_configuration()
        
        self.logger.info(f"🔧 CalibrationContext initialized for ABC-SMC calibration")
        self.logger.info(f"📊 Database: {self.db_path}")
        self.logger.info(f"🎯 QoIs: {list(self.qoi_functions.keys())}")
        for spec in self.models:
            self.logger.info(
                f"🔍 Model '{spec.name}': struc={spec.model_config['struc_name']}, "
                f"params={list(spec.prior.get_parameter_names())}, replicates={spec.num_replicates}"
            )
        if self.model_selection:
            self.logger.info(f"🔀 ABC-SMC model selection enabled over {self.num_models} models")
        self.logger.info(f"⚙️ Sampler: {self.sampler_type} with {self.num_workers} workers")

    # ------------------------------------------------------------------
    # Back-compat single-model accessors (delegate to self.models[0])
    # ------------------------------------------------------------------
    @property
    def model_config(self) -> dict:
        return self.models[0].model_config

    @model_config.setter
    def model_config(self, value: dict):
        self.models[0].model_config = value

    @property
    def prior(self) -> Distribution:
        return self.models[0].prior

    @prior.setter
    def prior(self, value: Distribution):
        self.models[0].prior = value

    @property
    def fixed_params(self) -> dict:
        return self.models[0].fixed_params

    @fixed_params.setter
    def fixed_params(self, value: dict):
        self.models[0].fixed_params = value

    def _build_model_specs(self, abc_options: dict, model_config: Optional[dict],
                           prior: Optional[Distribution]) -> List[ModelSpec]:
        """Normalize the model configuration into a list of ModelSpec objects."""
        for deprecated in ("num_models", "model_selection"):
            if deprecated in abc_options:
                self.logger.warning(
                    f"abc_options['{deprecated}'] is deprecated and ignored; "
                    f"use abc_options['models'] to enable model selection."
                )

        raw_models = abc_options.get("models", None)
        specs: List[ModelSpec] = []
        if raw_models:
            if model_config is not None or prior is not None:
                self.logger.warning(
                    "abc_options['models'] is set; top-level model_config/prior are ignored."
                )
            for i, entry in enumerate(raw_models):
                missing = [k for k in ("name", "model_config", "prior") if entry.get(k) is None]
                if missing:
                    raise ValueError(f"abc_options['models'][{i}] is missing required field(s): {missing}")
                specs.append(ModelSpec(
                    name=str(entry["name"]),
                    model_config=dict(entry["model_config"]),
                    prior=entry["prior"],
                    fixed_params=dict(entry.get("fixed_params") or {}),
                    num_replicates=entry.get("num_replicates", None),
                ))
        else:
            if model_config is None or prior is None:
                raise ValueError("Provide either abc_options['models'] or both model_config and prior.")
            specs.append(ModelSpec(
                name=str(abc_options.get("model_name", "Model_0")),
                model_config=dict(model_config),
                prior=prior,
                fixed_params=dict(abc_options.get("fixed_params") or {}),
                num_replicates=model_config.get("numReplicates", None),
            ))

        # Names must be unique and filesystem-safe (used for IO subfolders + DB keys)
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Model names must be unique, got: {names}")
        for n in names:
            if not re.match(r"^[A-Za-z0-9_.-]+$", n):
                raise ValueError(
                    f"Model name '{n}' is invalid: use only letters, digits, '_', '.', '-'."
                )

        # Resolve replicate counts; isolate IO folders per model for multi-model runs
        for spec in specs:
            spec.num_replicates = self._resolve_num_replicates(spec)
            if len(specs) > 1:
                spec.model_config.setdefault("input_folder", f"{spec.name}/")
                spec.model_config.setdefault("output_folder", f"{spec.name}/")
        return specs

    def _resolve_num_replicates(self, spec: ModelSpec) -> int:
        """Resolve a spec's replicate count from the spec, model_config, or ini file."""
        if spec.num_replicates is not None:
            return int(spec.num_replicates)
        n = spec.model_config.get("numReplicates", None)
        if n is not None:
            return int(n)
        configFile = configparser.ConfigParser()
        configFile.read_file(open(spec.model_config["ini_path"]))
        return int(configFile[spec.model_config["struc_name"]]["numReplicates"])

    def _instantiate_model(self, spec: ModelSpec) -> "PhysiCell_Model":
        """Build a PhysiCell_Model for a spec, applying any per-model IO subfolders."""
        model = PhysiCell_Model(spec.model_config["ini_path"], spec.model_config["struc_name"])
        if spec.model_config.get("input_folder"):
            model.input_folder += spec.model_config["input_folder"]
        if spec.model_config.get("output_folder"):
            model.output_folder += spec.model_config["output_folder"]
        return model

    def _setup_parallelization(self):
        """Setup parallelization strategy based on configuration."""
        if self.sampler_type == "dask":
            self.workers_inner = None  # Dask handles its own parallelization
            self.workers_outer = self.num_workers
        elif self.sampler_type == "multicore":
            # Calculate nested parallelization for multicore
            self.workers_inner = min(self.num_workers, self.num_replicates)  # workers for replicates
            self.workers_outer = max(1, self.num_workers // self.workers_inner)  # workers for parameter sets
        else:
            raise ValueError(f"Sampler {self.sampler_type} is not supported. Use 'dask' or 'multicore'.")

    def _validate_configuration(self):
        """Validate the configuration parameters."""
        required_model_keys = ['ini_path', 'struc_name']
        for spec in self.models:
            for key in required_model_keys:
                if key not in spec.model_config:
                    raise ValueError(f"Model '{spec.name}': missing required model_config key: {key}")
            if not spec.prior:
                raise ValueError(f"Model '{spec.name}': prior cannot be empty")

        if not self.qoi_functions:
            raise ValueError("qoi_functions cannot be empty")

        if not self.distance_functions:
            raise ValueError("distance_functions cannot be empty")

        # Validate QoI consistency
        for qoi in self.qoi_functions.keys():
            if qoi not in self.distance_functions:
                raise ValueError(f"Distance function not defined for QoI: {qoi}")

    def setup_sampler(self, cluster_setup_func=None):
        """Setup the pyABC sampler based on configuration."""
        if self.sampler_type == "dask":
            if cluster_setup_func is None:
                raise ValueError("cluster_setup_func must be provided for cluster mode with Dask")
            cluster = cluster_setup_func()
            my_sampler = sampler.DaskDistributedSampler(Client(cluster))
            self.logger.info(f"Using Dask sampler with {self.num_workers} workers")
        elif self.sampler_type == "multicore":
            my_sampler = sampler.MulticoreParticleParallelSampler(n_procs=self.workers_outer)
            self.logger.info(f"Using nested multicore parallelization: {self.workers_outer} outer processes × {self.workers_inner} inner threads = {self.workers_outer * self.workers_inner} total")
        else:
            raise ValueError(f"Sampler {self.sampler_type} is not supported.")
        
        return my_sampler

    def setup_population_strategy(self):
        """Setup population size strategy."""
        if self.population_strategy == "adaptive":
            return AdaptivePopulationSize(
                start_nr_particles=self.max_population_size,
                mean_cv=0.2,
                min_population_size=self.min_population_size,
                max_population_size=self.max_population_size
            )
        else:
            # Fixed population size - use max_population_size as the fixed size
            return self.max_population_size

    def setup_distance_function(self, distances_dict):
        qois = list(self.qoi_functions.keys())
        distance_funcs = [distance_func["function"] for distance_func in distances_dict.values()]
        distance_weights = [distance_func.get("weight", 1.0) for distance_func in distances_dict.values()]
        """Setup distance function with optional adaptive weighting."""
        if len(qois) > 1:
            if self.adaptive_distance:
                # If adaptive distance, try to load previous weights if database exists
                if os.path.exists(self.db_path) and hasattr(self, '_load_adaptive_weights'):
                    try:
                        distance_weights = self._load_adaptive_weights(self.db_path)
                    except Exception as e:
                        raise ValueError(f"Could not load adaptive weights: {e}")
                    # Starting with loaded adaptive weights.
                    distance_func = AdaptiveAggregatedDistance(distance_funcs, initial_weights=distance_weights, log_file=self.adaptive_distance_file)
                else:
                    # Start with adaptive weighting
                    distance_func = AdaptiveAggregatedDistance(distance_funcs, log_file=self.adaptive_distance_file)
                    self.logger.info("Starting with adaptive distance weighting")
            else:
                # Use provided weights or equal weights
                distance_func = AggregatedDistance(distance_funcs, weights=distance_weights)
                self.logger.info(f"Using fixed distance weights: {distance_weights}")
        else:
            distance_func = distance_funcs[0]
            self.logger.info(f"Using single distance function: {distances_dict[qois[0]]['function']}")

        return distance_func

    def setup_transition_function(self):
        """Setup transition functions for ABC-SMC, one kernel per candidate model.

        "multivariate" (default) uses ScaledMultivariateNormalTransition, a multivariate
        normal kernel fitted on standardized parameters so that parameters on very
        different scales are all perturbed. "local" uses pyABC's LocalTransition.
        """
        if self.transition_strategy == "local":
            return [LocalTransition() for _ in self.models]
        elif self.transition_strategy == "multivariate":
            return [ScaledMultivariateNormalTransition() for _ in self.models]
        else:
            raise ValueError(f"Transition strategy '{self.transition_strategy}' is not supported. Use 'multivariate' or 'local'.")

    def setup_epsilon_function(self):
        """Setup epsilon (tolerance) function for ABC-SMC."""
        if self.epsilon_strategy == "quantile":
            return QuantileEpsilon(alpha=self.epsilon_alpha)
        else:
            raise ValueError(f"Epsilon strategy '{self.epsilon_strategy}' not supported")

    def create_model_wrapper(self, model_spec: ModelSpec, workers_inner=None):
        """Create a pyABC model function for one candidate model."""
        def model_wrapper(pars):
            return self._run_physicell_model(pars, model_spec, workers_inner)
        model_wrapper.__name__ = f"run_physicell_{model_spec.name}"
        return model_wrapper

    def _run_physicell_model(self, pars, model_spec: ModelSpec, workers_inner=None):
        """Run a candidate PhysiCell model with given parameters.

        This is the function pyABC calls directly for every proposed particle. A
        PhysiCell crash (segfault, malformed output, a transient filesystem
        hiccup, ...) is a property of one parameter draw, not of the whole
        calibration — letting the exception propagate kills the pyABC worker
        process outright ("At least one worker is dead"), aborting the entire
        (possibly multi-day) run over a single bad sample. Instead, log it and
        return None: pyABC treats that as this particle's raw data, so the
        distance function decides its fate. Distance functions must therefore
        handle `sim is None` (and shape mismatches) by returning np.inf, the
        same convention used for a malformed/short simulation output — see
        ex11_ABC_ModelSelection.ipynb's `_relative_rmse`.
        """
        try:
            # Choose parallelization strategy based on sampler
            if self.sampler_type == 'multicore' and workers_inner is not None:
                return self._run_replicates_parallel(workers_inner, pars, model_spec)
            else:
                return self._run_physicell_model_sequential(pars, model_spec)
        except Exception as e:
            self.logger.error(f"Error in model evaluation ({model_spec.name}): {e} -- rejecting this particle")
            return None
        
    def _default_aggregation_func(self, replicate_results):
        """Define function to aggregate the replicates"""
        try:
            results_df = pd.concat(list(replicate_results.values()), ignore_index=True)
            # Take the mean of all columns in the same sampleID and time
            return results_df.pivot_table(index=['sampleID','time'])
        except Exception as e:
            # `.values()` is a dict_values view, not subscriptable -- guard both
            # that and the empty-dict case so this handler can't itself raise a
            # confusing, unrelated TypeError/IndexError that masks `e`.
            sample_ids = (
                list(replicate_results.values())[0]['sampleID'].unique()
                if replicate_results else "unknown (no replicates)"
            )
            raise ValueError(f"Error in _default_aggregation_func for sampleID: {sample_ids}: {e}") from e

    def _run_physicell_model_sequential(self, pars, model_spec: ModelSpec, sample_id=None, replicate_id=None):
        """Run one candidate PhysiCell model sequentially."""
        fixed_params = model_spec.fixed_params
        num_replicates = model_spec.num_replicates

        # Create the PhysiCell model instance for each worker (applies per-model IO subfolders)
        physicell_model = self._instantiate_model(model_spec)
        physicell_model.numReplicates = num_replicates
        physicell_model.timeout = 600
        physicell_model.output_summary_Path = None

        # Get parameter names
        params_xml = [param_name for param_name in physicell_model.XML_parameters_variable.values()]
        params_rules = [param_name for param_name in physicell_model.parameters_rules_variable.values()]

        # Get worker ID
        if sample_id is None:
            sample_id = self._get_worker_id()

        # Prepare parameters
        dic_pars_xml = {par: (pars[par] if par in pars.keys() else None) for par in params_xml}
        dic_pars_rules = {par: (pars[par] if par in pars.keys() else None) for par in params_rules}

        # Include fixed parameters
        for par in fixed_params.keys():
            if par in dic_pars_xml.keys():
                dic_pars_xml[par] = fixed_params[par]
            if par in dic_pars_rules.keys():
                dic_pars_rules[par] = fixed_params[par]

        # Validation
        if None in dic_pars_xml.values() or None in dic_pars_rules.values():
            raise ValueError(f"Some parameters are None: {dic_pars_xml}, {dic_pars_rules}")

        # Run replicates. NOTE: keep the loop variable distinct from the
        # `replicate_id` parameter -- reusing it here would shadow the
        # parameter, so the `replicate_id is None` check below would never be
        # true again after the loop runs, silently skipping aggregation for
        # every caller that passes replicate_id=None (i.e. any non-multicore
        # sampler, such as "dask").
        replicates = range(num_replicates) if replicate_id is None else [replicate_id]

        # Draw one distinct random seed per replicate up front (without
        # replacement) so replicates of the same particle launched in the same
        # clock tick -- e.g. parallel threads in _run_replicates_parallel --
        # don't collide on PhysiCell's default system-clock seed. Mirrors the
        # equivalent fix in the model-analysis runner (ma_context.py).
        replicate_seeds = np.random.default_rng().choice(2**32, size=len(replicates), replace=False).tolist()

        dic_all_replicates = {}
        for seed, rep_id in zip(replicate_seeds, replicates):
            try:
                _, _, result_data = run_replicate_serializable(
                    PhysiCellModel_conf=model_spec.model_config,
                    sample_id=sample_id,
                    replicate_id=rep_id,
                    ParametersXML=dic_pars_xml,
                    ParametersRules=dic_pars_rules,
                    qoi_functions=self.qoi_functions,
                    qoi_def=self.qoi_def,
                    return_binary_output=False,
                    #drop_columns,
                    custom_summary_function=self.summary_function,
                    random_seed=int(seed),
                )
                dic_all_replicates[rep_id] = result_data
            except Exception as e:
                raise RuntimeError(f"Error in RunModel (SampleID: {sample_id}): {e}")

            # Check if RunModel returned valid data
            if not hasattr(result_data, 'columns') or len(result_data) == 0:
                raise RuntimeError(f"RunModel returned empty or invalid DataFrame for SampleID: {sample_id}, ReplicateID: {rep_id}")
        # All replicates done, run aggregation function
        if replicate_id is None:
            return self.aggregation_func(dic_all_replicates)
        else:
            return dic_all_replicates

    def _run_replicates_parallel(self, workers_inner, params, model_spec: ModelSpec):
        """Run replicates in parallel using ThreadPoolExecutor."""

        with ThreadPoolExecutor(max_workers=workers_inner) as executor:
            futures = []
            for replicate_id in range(model_spec.num_replicates):
                future = executor.submit(
                    self._run_physicell_model_sequential,
                    params, model_spec, self._get_worker_id(), replicate_id
                )
                futures.append(future)
            
            dic_all_replicates = {}
            for future_done in as_completed(futures):
                dict_result = future_done.result(timeout=0.5)
                for key, value in dict_result.items():
                    dic_all_replicates[key] = value

        return self.aggregation_func(dic_all_replicates)

    def _get_worker_id(self):
        """Get worker ID for distributed computing."""
        try:
            worker_id = int(get_worker().name)
        except:
            try:
                worker_id = int(get_worker().name.split("-")[-1])
            except:
                worker_id = os.getpid()
        return worker_id

    def setup_abc_smc(self, models_list, priors_list, distance_function, population_size, transitions_func, my_sampler, eps_function):
        """Setup the ABC-SMC object (with between-generation weight and kernel checks)."""
        return CheckedABCSMC(
            max_zero_weight_fraction=self.max_zero_weight_fraction,
            models=models_list,
            parameter_priors=priors_list,
            distance_function=distance_function,
            population_size=population_size,
            transitions=transitions_func,
            sampler=my_sampler,
            eps=eps_function
        )

    def load_or_create_database(self, abc_smc, abc_id=1):
        """Load existing database or create new one."""
        db_file = "sqlite:///" + os.path.join(self.db_path)
        
        if os.path.exists(self.db_path):
            try:
                abc_smc.load(db_file, abc_id=abc_id)
                self.logger.info(f"Loaded existing database: {self.db_path}")
                return True, abc_smc.history.n_populations, abc_smc.history.total_nr_simulations
            except ValueError as e:
                self.logger.error(f"Error loading database {db_file}: {e}")
                raise
        else:
            abc_smc.new(db_file, observed_sum_stat=self.dic_obsData)
            self.logger.info(f"Created new database: {self.db_path}")
            return False, 0, 0

    def run_calibration(self, abc_smc, resume_db=False, current_populations=0, current_simulations=0):
        """Run the ABC-SMC calibration."""
        if resume_db:
            extra_populations = max(0, self.max_populations - current_populations)
            extra_simulations = max(0, self.max_simulations - current_simulations)
            self.logger.info(f"Resuming: extra populations: {extra_populations}, extra simulations: {extra_simulations}")
            
            if extra_populations > 0 and extra_simulations > 0:
                abc_smc.run(max_nr_populations=self.max_populations, max_total_nr_simulations=self.max_simulations)
            else:
                self.logger.info("No additional calibration needed")
        else:
            self.logger.info(f"Starting calibration: max populations: {self.max_populations}, max simulations: {self.max_simulations}")
            abc_smc.run(max_nr_populations=self.max_populations, max_total_nr_simulations=self.max_simulations)
            # Add extra info of adaptive distance to database
            if self.adaptive_distance:
                insert_adaptive_weights_db(self.db_path, dict_distances=self.distance_functions, dict_adaptive_weights=load_dict_from_json(self.adaptive_distance_file))

    def check_convergence(self, abc_smc):
        """Check convergence criteria."""
        # This would need the check_convergence_generic function
        # For now, return False to continue until max iterations
        return False

    def include_additional_metadata(self, **metadata):
        """Include additional metadata in the database."""
        # This would need the include_additional_data_in_db function
        # For now, just log the metadata
        self.logger.info(f"Additional metadata: {metadata}")


def run_abc_calibration( calib_context: CalibrationContext) -> History:
    """
    Execute the complete ABC-SMC calibration process.
    
    This function orchestrates the entire ABC-SMC workflow using the CalibrationContext,
    including sampler setup, distance function configuration, model wrapper creation,
    and calibration execution with convergence checking.
    
    Args:
        calib_context (CalibrationContext): The calibration context containing all configuration
    
    Returns:
        History: The pyABC History object containing calibration results
    """
    logger = calib_context.logger
    
    try:
        logger.info("🚀 Starting ABC-SMC calibration process")
        logger.info(f"📊 Database: {calib_context.db_path}")
        logger.info(f"🎯 Max populations: {calib_context.max_populations}")
        logger.info(f"🔬 Max simulations: {calib_context.max_simulations}")

        
        # Setup sampler
        logger.info("⚙️ Setting up sampler...")
        my_sampler = calib_context.setup_sampler(calib_context.cluster_setup_func)
        
        # Setup population strategy
        logger.info("👥 Setting up population strategy...")
        population_size = calib_context.setup_population_strategy()
        
        # Setup distance function
        logger.info("📏 Setting up distance function...")
        distance_function = calib_context.setup_distance_function(calib_context.distance_functions)
        
        # Setup transition function
        logger.info("🔄 Setting up transition function...")
        transitions_func = calib_context.setup_transition_function()
        
        # Setup epsilon function
        logger.info("🎯 Setting up epsilon function...")
        eps_function = calib_context.setup_epsilon_function()
        
        # Setup model wrappers — one pyABC model function + prior per candidate model
        logger.info("🧬 Setting up model wrappers...")
        models_list = []
        priors_list = []
        for spec in calib_context.models:
            # Every sampled or fixed parameter must reach PhysiCell, or it would be silently dropped
            # (a custom run_single_replicate function maps parameters itself)
            if not calib_context.custom_run_single_replicate_func:
                calib_context._instantiate_model(spec).check_parameter_names(
                    list(spec.prior.get_parameter_names()) + list(spec.fixed_params),
                    context=f"Prior/fixed parameters of '{spec.name}'",
                )
            wrapper = calib_context.create_model_wrapper(spec, calib_context.workers_inner)
            # Register by name in module globals so pickle-by-reference samplers (Dask) resolve it
            globals()[wrapper.__name__] = wrapper
            models_list.append(wrapper)
            priors_list.append(spec.prior)
        if calib_context.model_selection:
            logger.info(
                f"🔀 ABC-SMC model selection over {len(models_list)} models: "
                f"{[s.name for s in calib_context.models]}"
            )
        
        # Setup ABC-SMC object
        logger.info("🔧 Setting up ABC-SMC object...")
        abc_smc = calib_context.setup_abc_smc(
            models_list=models_list,
            priors_list=priors_list,
            distance_function=distance_function,
            population_size=population_size,
            transitions_func=transitions_func,
            my_sampler=my_sampler,
            eps_function=eps_function
        )
        
        # Load or create database
        logger.info("💾 Managing database...")
        resume_db, current_populations, current_simulations = calib_context.load_or_create_database(abc_smc)

        # Persist run + model metadata *before* running any simulation, not
        # after. This is pure static configuration (model names, priors, config
        # fingerprint hashes) that doesn't depend on any simulation result, so
        # there's no reason to defer it. run_calibration() below is a single
        # blocking abc_smc.run(max_nr_populations=..., ...) call that does not
        # return until pyABC has completed *all* requested populations (or hit
        # the simulation cap) -- for a multi-hour/day run with no
        # convergence_check_func configured (the common case), that is the
        # entire calibration. Writing metadata only after it returns meant a
        # crash, timeout, or kill anywhere during that single call -- the most
        # likely place for exactly that to happen -- left pyABC's own History
        # file populated but Metadata/CandidateModels completely empty.
        # Idempotent (INSERT OR REPLACE) so it also refreshes on a later resume.
        try:
            insert_metadata_db(calib_context.db_path, calib_context)
            insert_models_db(calib_context.db_path, calib_context)
        except Exception as e:
            logger.warning(f"Could not persist calibration metadata: {e}")

        # Run calibration
        logger.info("🎲 Starting calibration run...")
        calib_context.run_calibration(abc_smc, resume_db, current_populations, current_simulations)

        # Check convergence and run additional populations if needed
        if calib_context.convergence_check_func is not None and getattr(calib_context, 'mode', 'cluster') == 'cluster':
            logger.info("🔍 Checking convergence...")
            while True:
                if calib_context.convergence_check_func(abc_smc.history):
                    logger.info("✅ Convergence achieved!")
                    break
                
                # If max limits reached, extend by one to run one more population
                if abc_smc.history.total_nr_simulations >= calib_context.max_simulations:
                    calib_context.max_simulations = abc_smc.history.total_nr_simulations + 1
                
                if abc_smc.history.n_populations >= calib_context.max_populations:
                    calib_context.max_populations = abc_smc.history.n_populations + 1
                
                logger.info(f"🔄 Continuing calibration: {calib_context.max_simulations} simulations, {calib_context.max_populations} populations")
                
                # Run one more iteration
                abc_smc.run(
                    max_nr_populations=calib_context.max_populations,
                    max_total_nr_simulations=calib_context.max_simulations
                )
                # Add extra info of adaptive distance to database
                if calib_context.adaptive_distance:
                    insert_adaptive_weights_db(calib_context.db_path, dict_distances=calib_context.distance_functions, dict_adaptive_weights=load_dict_from_json(calib_context.adaptive_distance_file))
        
        # Persist results that depend on the final history state (adaptive
        # weights) -- the run + model metadata itself was already persisted
        # right after the first run() call, above. Model probabilities are not
        # persisted separately: they're pure derived data already recoverable
        # from pyABC's own tables via history.get_model_probabilities(), so
        # there is nothing here worth a second, independently-stale copy of.
        try:
            if calib_context.model_selection:
                logger.info(f"🔀 Final model probabilities:\n{abc_smc.history.get_model_probabilities()}")
            if (calib_context.adaptive_distance and calib_context.adaptive_distance_file
                    and os.path.exists(calib_context.adaptive_distance_file)):
                insert_adaptive_weights_db(
                    calib_context.db_path,
                    dict_distances=calib_context.distance_functions,
                    dict_adaptive_weights=load_dict_from_json(calib_context.adaptive_distance_file),
                )
        except Exception as e:
            logger.warning(f"Could not persist calibration metadata: {e}")

        # Remove temporary file and folders (one PhysiCell model per candidate)
        for spec in calib_context.models:
            calib_context._instantiate_model(spec).remove_io_folders()
        
        # Final results
        final_history = abc_smc.history
        logger.info("✅ ABC-SMC calibration completed successfully!")
        logger.info(f"📈 Final statistics:")
        logger.info(f"   - Populations: {final_history.n_populations}")
        logger.info(f"   - Total simulations: {final_history.total_nr_simulations}")
        logger.info(f"   - Database: {calib_context.db_path}")
        
        return final_history
        
    except Exception as e:
        logger.error(f"❌ Error in ABC-SMC calibration: {e}")
        raise
