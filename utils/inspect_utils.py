import inspect

def get_variable_name(text):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return_value = [key for key, value in callers_local_vars if value is text]
    return return_value[0] if len(return_value) == 1 else 'variable name not found'
