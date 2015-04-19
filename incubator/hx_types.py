import numpy as np

INT_TYPE = np.int64

def is_any_int_type(x):
    return type(x) in [int, np.int64, np.int32, np.int16, np.int8], "type %r should be an int type."%(type(x),)
