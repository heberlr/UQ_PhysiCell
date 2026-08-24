# Installation

## Requirements

UQ-PhysiCell requires Python 3.10 or later and the following dependencies:
- [pcdl](https://github.com/elmbeech/physicelldataloader/tree/master)
- [SALib](https://salib.readthedocs.io/en/latest/index.html)

matplotlib, numpy, pandas, scipy, and seaborn are also installed automatically.

## Installation from PyPI

We recommend using a virtual environment, such as [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```bash
pip install uq-physicell
```

## Installation from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/heberlr/UQ_PhysiCell.git
   cd UQ_PhysiCell
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

## Additional Dependencies

Some features depend on packages that are not installed by default. These are grouped into extras that can be installed with `pip install uq-physicell[<extra>]` (or `pip install -e .[<extra>]` from a source checkout):

- `gui` — [PyQt5](https://pypi.org/project/PyQt5/), required to launch the {doc}`GUI <gui>`
- `mpi` — [mpi4py](https://mpi4py.readthedocs.io/en/stable/), required for inter-node MPI execution (see {doc}`model_analysis`)
- `bo` — [torch](https://pytorch.org/), [botorch](https://botorch.org/), and [gpytorch](https://gpytorch.ai/), required for {ref}`Bayesian Optimization <bayesian-optimization>`
- `abc` — [pyabc](https://pyabc.readthedocs.io/en/latest/) and [dask](https://www.dask.org/), required for {ref}`Approximate Bayesian Computation <approximate-bayesian-computation-abc>`; `dask` is only needed if you select the `"dask"` ABC sampler (the default `"multicore"` sampler only needs `pyabc`)
- `utils` — extra packages (e.g. [gudhi](https://gudhi.inria.fr/)) used by a few advanced, opt-in summary/QoI functions

For example:
```bash
pip install uq-physicell[gui]
pip install uq-physicell[bo,abc]
```

Or install everything at once:
```bash
pip install uq-physicell[all]
```

## Verification

To verify the installation, run:
```python
import uq_physicell
print(f"UQ-PhysiCell version: {uq_physicell.__version__}")
```

You should see the current version number displayed. The latest version is {sub-ref}`version`.