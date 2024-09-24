# Installation
Install the package using pip:
```
pip install uq_physicell
```

# Examples
Here are some examples to help you get started with using the package:

1. Example 1: Basic Usage
    - Description: Print information of models [physicell_model_1](examples/SampleModel.ini#l1) and [physicell_model_2](examples/SampleModel.ini#l18) as shown in the config file [examples/SampleModel.ini](examples/SampleModel.ini#l20), without running the PhysiCell simulation.
    - Code:
      ```
      python examples/ex1_print.py
      ```

2. Example 2: Running PhysiCell Simulations
    - Description: Run PhysiCell simulations of [physicell_model_2](examples/SampleModel.ini#l18)
    - Step 1: Compile the virus-macrophage example in the PhysiCell folder.
    - Code:
      ```
      make reset && make virus-macrophage-sample && make
      ```
    - Step 2: Change the path of the executable variable in [physicell_model_2](examples/SampleModel.ini#l20) model in the [examples/SampleModel.ini](examples/SampleModel.ini) file.
    - Code:
      ```
      executable = [new path]
      ```
    - Step 3: run the simulations.
    - Code:
      ```
      python examples/ex2_runModel.py
      ```

3. Example 3: Customizable Summary Function
    - Description: This example illustrates how to run PhysiCell simulations of [physicell_model_2](examples/SampleModel.ini#l18) with a customizable summary function that generates population time series.
    - Code:
      ```
      python examples/ex3_runModelCust.py
      ```

4. Example 4: Sensitivity Analysis as an unique task
    - Requirements: install SALib 
    - Description: Run simulations to perfom sensitivity analysis (Sobol method)  [physicell_model_2](examples/SampleModel.ini#l18)
    - Code:
      ```
      python examples/ex4_runSA_singleTask.py
      ```
    - After simulate all samples you can perform the analysis: ``python example/SA_ex4_ex5.py``

5. Example 5: Sensitivity Analysis as parallel tasks using MPI
    - Requirements: install SALib and mpi4py
    - Description: Run simulations to perfom sensitivity analysis (Sobol method) of [physicell_model_2](examples/SampleModel.ini#l18)
    - Code:
      ```
      mpiexec -n 2 python -m mpi4py examples/ex5_runSA_MPI.py
      ```
    - After simulate all samples you can perform the analysis: ``python example/SA_ex4_ex5.py``

Feel free to explore these examples to understand the capabilities of the package and how to use it effectively.
