# Installation
Install the package using pip:
```
pip install uq_physicell
```

# Examples
Here are some examples to help you get started with the package:

## Example 1: Basic Usage
  - Description: Print information about two model structures, [physicell_model_1](examples/SampleModel.ini#l1) and [physicell_model_2](examples/SampleModel.ini#l18), as defined in the config file [examples/SampleModel.ini](examples/SampleModel.ini#l20). This operation does not run a PhysiCell simulation. See [example 1](examples/ex1_print.py).
    ```bash
    python examples/ex1_print.py
    ```

## Example 2: Running PhysiCell Simulations
  - Requirements: A PhysiCell folder is required.
  - Description: Run three PhysiCell simulations associated with the key [physicell_model_2](examples/SampleModel.ini#l18) in the config file [SampleModel.ini](examples/SampleModel.ini). This corresponds to the `virus_macrophage` example in `PhysiCell's sample projects`. See [example 2](examples/ex2_runModel.py).
    - **First simulation:** Demonstrates running a simulation with a predefined summary function that summarizes the final population of live and dead cells, storing results in a new folder `output2`.
    - **Second simulation:** Runs the simulation while preserving the config files and retaining the complete PhysiCell output without summarization.
    - **Third simulation:** Configures the execution to summarize the output and returns a DataFrame with the summary.

    Run script:
    ```bash
    python examples/ex2_runModel.py
    ```

    Alternatively, download the lastest PhysiCell version with:
    ```bash
    bash examples/PhysiCell.sh
    ```
    This will create a folder named `PhysiCell-master` inside `examples`. Populate and compile the project (Step 1 below) without modifying [SampleModel.ini](examples/SampleModel.ini).
    - Step 1: Compile the `virus-macrophage` example in the PhysiCell folder:
      ```bash
      make reset && make virus-macrophage-sample && make
      ```
    - Step 2: Update the `executable` and `configFile_ref` paths in the [physicell_model_2](examples/SampleModel.ini#l20) model in the [examples/SampleModel.ini](examples/SampleModel.ini) section of [SampleModel.ini](examples/SampleModel.ini).
      ```ini
      executable = [new path]
      configFile_ref = [new path]
      ```
    - Step 3: Execute the script:
      ```bash
      python examples/ex2_runModel.py
      ```

## Example 3: Customizable Summary Function
  - Requirements: A PhysiCell folder is required.
  - Description: Run two simulations of [physicell_model_2](examples/SampleModel.ini#l18) using a customizable summary function to generate population time series. See [example 3](examples/ex3_runModelCust.py).
    - **First simulation:** Runs the simulation while preserving the config files definitions and using a custom summary function.
    - **Second simulation:** Similar to the first, but adjust the model for 4 OpenMP threads and returns a DataFrame instead of a summary file.
  - Run script:
    ```bash
    python examples/ex3_runModelCust.py
    ```

## Example 4: Sensitivity Analysis (Single Task)
  - Requirements: A PhysiCell folder and the ``SALib`` Python package.
  - Description: Perform sensitivity analysis using the Sobol method. See [example 4](examples/ex4_runSA_singletask.py).
  - Run script:
    ```bash
    python examples/ex4_runSA_singleTask.py
    ```

## Example 5: Sensitivity Analysis (Parallel Tasks with MPI)
  - Requirements: A PhysiCell folder, and the `SALib` and `mpi4py` Python packages.
  - Description: Perform sensitivity analysis using the Sobol method with MPI. See [example 5](examples/ex5_runSA_MPI.py).
  - Run script:
    ```bash
    mpiexec -n 2 python -m mpi4py examples/ex5_runSA_MPI.py
    ```

## Example 6: Sensitivity Analysis with Constrained Parameters (MPI).
  - Requirements: A PhysiCell folder (SampleModel.ini assumes it is located in the examples folder) and the `SALib` and `mpi4py` Python packages. Compile the asymmetric_division example:
    ```bash
    make reset && make asymmetric-division-sample && make
    ```
  - Description: Perform sensitivity analysis (Sobol method) with MPI, handling constrained parameters. This example uses the `asymmetric_division` model from `PhysiCell's sample_projects` and includes analyzing `parameters of rules`. See [example 6](examples/ex6_runSA_AsymDiv.py).
  - Run script:
    ```bash
    mpiexec -n 2 python -m mpi4py examples/ex6_runSA_AsymDiv.py
    ```

Feel free to explore these examples to understand the package's capabilities and how to use it.
