# Installation

## Requirements

UQ-PhysiCell requires Python 3.8 or later and the following dependencies:

- numpy
- pandas
- pcdl
- torch (for Bayesian optimization)
- botorch (for Bayesian optimization)
- gpytorch (for Gaussian processes)

## Installation from PyPI

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

For Bayesian optimization features, install the optional dependencies:

```bash
pip install torch botorch gpytorch
```

For documentation building:

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

## Verification

To verify the installation, run:

```python
import uq_physicell
print(f"UQ-PhysiCell version: {uq_physicell.__version__}")
```

You should see the current version number displayed. The latest version is {sub-ref}`version`.
