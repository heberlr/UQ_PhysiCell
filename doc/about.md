# About UQ-PhysiCell

UQ-PhysiCell is a comprehensive framework for performing uncertainty quantification and parameter calibration of PhysiCell models. It provides sophisticated tools for model analysis, calibration, and model selection.

## Describing Your Model

Every workflow — sensitivity analysis, calibration, or a single one-off run — starts from the same thing: an INI configuration file that describes one or more model *structures*. For each structure you specify the PhysiCell executable path, the XML configuration file, the number of replicates, and which parameters the framework is allowed to change during simulations.

The configuration supports the following (examples):

- executable: path to the PhysiCell binary
- configfile_ref: path to the model XML file
- numreplicates: number of replicates to run
- parameters: mapping of XML paths to fixed values or to parameter names used for sampling
- rulesfile_ref: path to a CSV file with rule definitions (optional)
- parameters_rules: mapping of rule parameters to names used for sampling

See an example INI file at [examples/virus-mac-new/uq_pc_struc.ini](examples/virus-mac-new/uq_pc_struc.ini).

UQ-PhysiCell uses Python's ElementTree API to reference XML paths. You can fix some parameters for an experiment (for example, disable SVG output or set the output interval) while allowing others to vary, including parameters defined in rules files.
```ini
[Model_struc]
executable = ./project
configfile_ref = config/PhysiCell_settings.xml
numreplicates = 2
parameters = {
    './/save/SVG/enable': 'false', 
    './/save/full_data/interval': '360', 
    ".//cell_definitions/cell_definition[@name='macrophage']/phenotype/cell_interactions/live_phagocytosis_rates/phagocytosis_rate[@name='epithelial_infected']": [None, 'mac_phag_rate_infected'], 
    ".//cell_definitions/cell_definition[@name='macrophage']/phenotype/motility/migration_bias": [None, 'mac_motility_bias']}
rulesfile_ref = config/cell_rules.csv
parameters_rules = {
    'epithelial,virus,increases,transform to epithelial_infected,saturation': [None, 'epi2infected_sat'], 
    'epithelial,virus,increases,transform to epithelial_infected,half_max': [None, 'epi2infected_hfm']}
```

## Quick Start: Context Objects

Once you have a configuration file, most work happens through one of three high-level *context* objects. Each one takes your `model_config` (the `.ini` path plus the structure name) together with the parameters/QoIs/search space relevant to the task, and handles sampling, running simulations (serially, in multiple processes, or across MPI nodes — see {doc}`model_analysis`), and storing everything in a SQLite database for later analysis and plotting:

- **`ModelAnalysisContext`** (`uq_physicell.model_analysis`) — global/local sensitivity analysis and general parameter-space exploration. See {doc}`model_analysis` for a worked example.
- **`CalibrationContext`** (`uq_physicell.bo`) — calibration via multi-objective Bayesian Optimization. See {ref}`Bayesian Optimization <bayesian-optimization>` for a worked example.
- **`CalibrationContext`** (`uq_physicell.abc`) — calibration via Approximate Bayesian Computation (ABC-SMC). See {ref}`Approximate Bayesian Computation (ABC) <approximate-bayesian-computation-abc>` for a worked example.

For example, running a small Sobol sensitivity analysis:
```python
from uq_physicell.model_analysis import ModelAnalysisContext

model_config = {"ini_path": "uq_pc_struc.ini", "struc_name": "Model_struc"}
params_info = {
    "mac_phag_rate_infected": {"lower_bound": 0.7, "upper_bound": 1.5},
    "mac_motility_bias":      {"lower_bound": 0.0, "upper_bound": 1.0},
}
qoi_funcs = {
    "epithelial_live": lambda df_cell: len(df_cell[df_cell['dead'] == False]),
}

context = ModelAnalysisContext("results.db", model_config, "Sobol", params_info, qoi_funcs, num_workers=8)
context.generate_samples(N=8)
context.run()
```

See {doc}`installation` for the extras (`bo`, `abc`, `mpi`, ...) each of these paths needs beyond the base install.

## Advanced: Running a Single Simulation Directly

The context objects above are built on top of `PhysiCell_Model`, the lower-level class that owns one model structure and knows how to generate its input files and run a single simulation. Most users won't need it directly, but it's there if you want fine-grained control — for example, to drive PhysiCell from your own sampling loop or optimizer rather than one of the built-in contexts.

```python
from uq_physicell import PhysiCell_Model

# Load the config file (.ini)
PC_model = PhysiCell_Model(
    configFilePath="uq_pc_struc.ini",
    keyModel="Model_struc"
)

# Print out the structure
PC_model.info()

# Run a simulation
PC_model.RunModel(
    SampleID = 0, ReplicateID = 0, 
    Parameters = {"mac_phag_rate_infected": 1.0, "mac_motility_bias": 0.5},
    ParametersRules = {"epi2infected_sat": 0.01, "epi2infected_hfm": 0.2}
)
```
