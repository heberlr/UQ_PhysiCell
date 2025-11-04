# About UQ-PhysiCell

UQ-PhysiCell is a comprehensive framework for performing uncertainty quantification and parameter calibration of PhysiCell models. It provides sophisticated tools for model analysis, calibration, and model selection.

## Quick Start
The package uses an INI configuration file to define one or more model structures for experiments. For each model structure you should specify the PhysiCell executable path, the XML configuration file, the number of replicates, and which parameters the framework should change during simulations.

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
    './/save/full_data/interval': '360', ".//cell_definitions/cell_definition[@name='macrophage']/phenotype/cell_interactions/live_phagocytosis_rates/phagocytosis_rate[@name='epithelial_infected']": [None, 'mac_phag_rate_infected'], ".//cell_definitions/cell_definition[@name='macrophage']/phenotype/motility/migration_bias": [None, 'mac_motility_bias']}
rulesfile_ref = config/cell_rules.csv
parameters_rules = {'epithelial,virus,increases,transform to epithelial_infected,saturation': [None, 'epi2infected_sat'], 'epithelial,virus,increases,transform to epithelial_infected,half_max': [None, 'epi2infected_hfm']}
```
You can initialize the model structure, print information, and run a simulation as follows:
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