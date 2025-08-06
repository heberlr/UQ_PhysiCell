# UQ-PhysiCell Documentation

This directory contains the Sphinx-based documentation for UQ-PhysiCell.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r requirements.txt
```

Or using the Makefile:

```bash
make install-deps
```

### Building HTML Documentation

To build the HTML documentation:

```bash
make html
```

The generated documentation will be available in `_build/html/index.html`.

### Live Documentation Development

For live reloading during documentation development:

```bash
make livehtml
```

This will start a local server and automatically rebuild the documentation when files change.

### Cleaning Build Files

To clean the build directory:

```bash
make clean
```

## Documentation Structure

- `index.md` - Main documentation index
- `installation.md` - Installation instructions
- `bayesian_optimization.md` - Detailed Bayesian optimization guide
- `api_reference.md` - API documentation
- `examples.md` - Usage examples
- `conf.py` - Sphinx configuration
- `UQ_PhysiCell_logo.png` - Project logo

## Documentation Format

The documentation uses MyST Markdown format, which allows for rich formatting including:

- Math equations using LaTeX syntax
- Code blocks with syntax highlighting
- Cross-references and links
- Admonitions and callouts
- Automatic API documentation generation

## Contributing to Documentation

When adding new features or modules:

1. Update the relevant documentation files
2. Add new API documentation to `api_reference.md` if needed
3. Include examples in `examples.md`
4. Test the documentation builds correctly with `make html`

## API Documentation

The API documentation is automatically generated from docstrings in the source code. Make sure to follow these conventions:

- Use Google-style or NumPy-style docstrings
- Include parameter types and descriptions
- Add return value descriptions
- Include example usage where appropriate
