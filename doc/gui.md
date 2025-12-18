# Graphical User Interface (GUI)

To launch the GUI, run `uq_physicell`.

```{note}
This section of the GUI documentation is under active development. Content and usage instructions may change.
```

## Example Workflow

This example demonstrates how to use the UQ PhysiCell GUI to perform a complete uncertainty quantification workflow: from model configuration to sensitivity analysis. We'll build a simple model, run parameter sampling, and analyze the results.

**Prerequisites:** Navigate to your PhysiCell folder, populate the `template` project by running `make template && make`, then open PhysiCell Studio to customize the initial conditions and cell cycle model.

### Step 1: Modify the Template Project Using PhysiCell Studio
**Steps:**
1. Enable the initial condition checkbox
2. Set the number of cells to initialize the model to $0$
3. Create and save the custom initial condition with a disc where $R1 = 0$ and $R2 = 200$
4. Change the combo box from 'Flow cytometry model (separated)' to 'live' model
5. Save the .xml file

<div align="center">
  <img src="_static/figures/Model_IC01.png" width="45%"/>
  <img src="_static/figures/Model_IC02.png" width="45%"/>
  <br>
  <em>Left: Enable the initial condition checkbox (step 1). Right: Set the number of initial cells to 0 (step 2).</em>
</div>

<div align="center">
  <img src="_static/figures/Model_IC03.png" width="45%"/>
  <img src="_static/figures/Model_CellCycle.png" width="45%"/>
  <br>
  <em>Left: Create a disc with $R1 = 0$ and $R2 = 200$, then save (step 3). Right: Change the cell cycle model to 'live' (step 4).</em>
</div>


### Step 2: Create the UQ PhysiCell Configuration File (.ini)

Now we'll use the GUI to create a configuration file that defines which parameters to explore in our uncertainty quantification analysis.

**Steps:**
1. Launch UQ PhysiCell: run `uq_physicell` in the terminal
2. Load the .xml file from your model
3. Select parameters to be fixed in the model exploration: `max_time` = 1440, `omp_threads` = 1, and `enable SVG` = false
4. Select parameters to add to the analysis: `cell_cycle_entry` and `apoptosis_rate`
5. Provide a structure name (`Model A`), executable path (`project`), and number of replicates (`3`)
6. Save the .ini file (`uq_config.ini`)

<div align="center">
  <img src="_static/figures/UQ_IniFile_01.png" width="45%"/>
  <img src="_static/figures/UQ_IniFile_02.png" width="45%"/>
  <br>
  <em>Left: Load the .xml file (step 2). Right: Set `max_time` = 1440 (step 3).</em>
</div>
<div align="center">
  <img src="_static/figures/UQ_IniFile_03.png" width="45%"/>
  <img src="_static/figures/UQ_IniFile_04.png" width="45%"/>
  <br>
  <em>Left: Set `omp_threads` = 1 (step 3). Right: Set `enable SVG` = false (step 3).</em>
</div>
<div align="center">
  <img src="_static/figures/UQ_IniFile_05.png" width="45%"/>
  <img src="_static/figures/UQ_IniFile_06.png" width="45%"/>
  <br>
  <em>Left: Add the `cell_cycle_entry` parameter to the analysis (step 4). Right: Add the `apoptosis_rate` parameter to the analysis (step 4).</em>
</div>
<div align="center">
  <img src="_static/figures/UQ_IniFile_07.png" width="45%"/>
  <img src="_static/figures/UQ_IniFile_08.png" width="45%"/>
  <br>
  <em>Left: Define the structure name as `ModelA` and set the executable path to `project` (step 5). Right: Set the number of replicates and save the configuration as `uq_config.ini` (step 6).</em>
</div>


### Step 3: Generate the Simulation Database (.db)

Next, we'll sample the parameter space and run simulations to build a database of results.

**Steps:**
1. Define the parameter sampling strategy as `Global` and set the sampler to `Sobol`
2. Change the range of the `apoptosis_rate` parameter to `50%` and press Enter
3. Sample the parameters with `8` samples and click Plot to visualize the parameter space
4. Set the database filename to `Simulations.db` and click the Run Simulations button
5. Set the number of workers to run simulations in parallel using the inter-process strategy, then confirm in the warning message that you want to store the list of MCDS objects
<div align="center">
  <img src="_static/figures/UQ_GenerateDataBase_01.png" width="45%"/>
  <img src="_static/figures/UQ_GenerateDataBase_02.png" width="45%"/>
  <br>
  <em>Left: Define the parameter sampling strategy as `Global` (step 1). Right: Adjust the range of the `apoptosis_rate` parameter to `50%` (step 2).</em>
</div>
<div align="center">
  <img src="_static/figures/UQ_GenerateDataBase_03.png" width="45%"/>
  <img src="_static/figures/UQ_GenerateDataBase_04.png" width="45%"/>
  <br>
  <em>Left: Sample the parameters using Sobol sampling with 8 samples (step 3). Right: Plot the sampled parameters (step 3).</em>
</div>
<div align="center">
  <img src="_static/figures/UQ_GenerateDataBase_05.png" width="45%"/>
  <img src="_static/figures/UQ_GenerateDataBase_06.png" width="45%"/>
  <br>
  <em>Left: Set the database filename as `Simulations.db`, click the Run Simulations button, and set the number of workers (step 4-5). Right: Confirm the storage of the MCDS objects list (step 5).</em>
</div>

### Step 4: Define Quantities of Interest (QoIs) and Perform Sensitivity Analysis

Finally, we'll define the outputs we want to analyze and compute sensitivity indices to understand which parameters most influence the model behavior.

**Steps:**
1. Define the QoIs using the predefined options: `live_cells` and `dead_cells`
2. Visualize the mean values of the QoIs across all simulations
3. Run the sensitivity analysis and plot the results
<div align="center">
  <img src="_static/figures/UQ_QoI_SA_01.png" width="45%"/>
  <img src="_static/figures/UQ_QoI_SA_02.png" width="45%"/>
  <br>
  <em>Left: Select `live_cells` QoI (step 1). Right: Select `dead_cells` QoI (step 1).</em>
</div>
<div align="center">
  <img src="_static/figures/UQ_QoI_SA_03.png" width="45%"/>
  <img src="_static/figures/UQ_QoI_SA_04.png" width="45%"/>
  <br>
  <em>Left: Plot the QoIs (step 2). Right: Visualize the `dead_cells` QoI time series (step 2).</em>
</div>
<div align="center">
  <img src="_static/figures/UQ_QoI_SA_05.png" width="45%"/>
  <img src="_static/figures/UQ_QoI_SA_06.png" width="45%"/>
  <br>
  <em>Left: Run and plot the sensitivity analysis (step 3). Right: Visualize the sensitivity indices for `dead_cells` (step 3).</em>
</div>

```{note}
These QoIs were selected for demonstration purposes. Feel free to explore other predefined QoIs or create your own custom QoI. Note that other sensitivity analysis methods are also compatible with the Sobol sampling strategy, so you can experiment with different SA approaches as well.
```