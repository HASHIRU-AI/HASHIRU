def suppress_output(func):
    """
    Decorator to suppress output of a function.
    """
    def wrapper(*args, **kwargs):
        import sys
        import os
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        try:
            result = func(*args, **kwargs)
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        return result


    return wrapper