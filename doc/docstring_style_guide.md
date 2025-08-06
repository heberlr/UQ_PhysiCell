# Documentation Style Guide

## Docstring Formats for UQ-PhysiCell

This guide shows how to write proper docstrings for better Sphinx documentation generation.

### Google-Style Docstrings (Recommended)

UQ-PhysiCell uses Google-style docstrings for consistency with many scientific Python packages.

```python
def example_function(param1: dict, param2: str, optional_param: float = 1.0) -> tuple:
    """Brief one-line description of what the function does.
    
    Longer description can go here, explaining the algorithm, methodology,
    or providing additional context. This can span multiple paragraphs.
    
    Args:
        param1 (dict): Description of the first parameter. Can include
            details about the expected structure or format.
        param2 (str): Description of the second parameter.
        optional_param (float, optional): Description of optional parameter.
            Defaults to 1.0.
    
    Returns:
        tuple: Description of what is returned. For complex returns:
            - first_element (dict): Description of first element
            - second_element (list): Description of second element
    
    Raises:
        ValueError: Description of when this exception is raised.
        TypeError: Description of when this exception is raised.
    
    Example:
        Basic usage example:
        
        >>> result = example_function({'key': 'value'}, 'method')
        >>> print(result[0])
        {'analysis': 'complete'}
        
        Advanced usage:
        
        >>> result = example_function(
        ...     param1={'complex': 'structure'},
        ...     param2='advanced_method',
        ...     optional_param=2.5
        ... )
    
    Note:
        Important notes about the function behavior, limitations,
        or special considerations.
    
    See Also:
        related_function: Brief description of related function
        AnotherClass.method: Brief description of related method
    """
    pass
```

### NumPy-Style Docstrings (Alternative)

For comparison, here's the NumPy style format:

```python
def numpy_style_function(param1, param2, optional_param=1.0):
    """Brief description of the function.
    
    Longer description here.
    
    Parameters
    ----------
    param1 : dict
        Description of parameter1.
    param2 : str
        Description of parameter2.
    optional_param : float, optional
        Description of optional parameter. Default is 1.0.
    
    Returns
    -------
    result : tuple
        Description of return value.
    
    Raises
    ------
    ValueError
        Description of when this is raised.
    """
    pass
```

### Class Documentation

```python
class ExampleClass:
    """Brief description of the class.
    
    Longer description explaining the purpose and usage of the class.
    
    Args:
        init_param (str): Description of initialization parameter.
        config (dict, optional): Configuration dictionary. Defaults to None.
    
    Attributes:
        public_attr (str): Description of public attribute.
        another_attr (list): Description of another attribute.
    
    Example:
        >>> obj = ExampleClass('initialization_value')
        >>> result = obj.method_name()
    """
    
    def __init__(self, init_param: str, config: dict = None):
        """Initialize the ExampleClass.
        
        Args:
            init_param (str): Initialization parameter.
            config (dict, optional): Configuration dictionary.
        """
        self.public_attr = init_param
        self.another_attr = []
    
    def method_name(self, data: list) -> dict:
        """Process data and return results.
        
        Args:
            data (list): Input data to process.
        
        Returns:
            dict: Processed results with keys 'status' and 'data'.
        """
        return {'status': 'complete', 'data': data}
```

## Best Practices

1. **Always include type hints** in function signatures
2. **Start with a brief one-line summary**
3. **Use proper formatting** for parameters and returns
4. **Include examples** for complex functions
5. **Document exceptions** that might be raised
6. **Be consistent** with the chosen style throughout the project

## Sphinx-Specific Tips

- Use `Args:` instead of `Parameters:` for Google style
- Include type information in parentheses: `param (str):`
- Use proper indentation (4 spaces for continuation)
- Use `Returns:` instead of `Return:`
- Reference other functions with proper syntax: `:func:`function_name``
- Reference classes with: `:class:`ClassName``
