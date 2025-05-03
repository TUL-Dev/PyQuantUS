def dependencies(*deps: list) -> dict:
    """
    A decorator to specify the dependencies of a function.

    Args:
        deps (list): List of dependencies required by the function.

    Returns:
        function: The decorated function with the specified dependencies.
    """
    def decorator(func):
        if type(func) is not dict:
            out_dict = {}
            out_dict['func'] = func
            out_dict['deps'] = deps
            return out_dict
        func['deps'] = deps
        return func
    return decorator

def required_kwargs(*kwarg_names: list) -> dict:
    """
    A decorator to specify the required keyword arguments for a function.

    Args:
        kwarg_names (list): List of required keyword argument names.

    Returns:
        function: The decorated function with the specified keyword arguments.
    """
    def decorator(func):
        if type(func) is not dict:
            out_dict = {}
            out_dict['func'] = func
            out_dict['kwarg_names'] = kwarg_names
            return out_dict
        func['kwarg_names'] = kwarg_names
        return func
    return decorator

def output_vars(*names: list) -> dict:
    """
    A decorator to specify the variable names written to in ResultsClass.

    Args:
        names (list): List of variable names to be written to in ResultsClass.

    Returns:
        function: The decorated function with the specified variable names.
    """
    def decorator(func):
        if type(func) is not dict:
            out_dict = {}
            out_dict['func'] = func
            out_dict['outputs'] = names
            return out_dict
        func['outputs'] = names
        return func
    return decorator