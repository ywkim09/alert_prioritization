import numpy as np
import errno
import os
from datetime import datetime
from itertools import chain, combinations

def powerSet(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def checkExistType(attType, jsonDict):
    existence = False
    try:
        for k, v in jsonDict.items():
            if attType in jsonDict.keys():
                existence = True
            elif isinstance(v, dict):
                existence = checkExistType(attType, v)
    except:
        pass
    return existence

def checkAttackerType(attType, jsonDict):
    interOutput = {}
    output = {}
    existence = checkExistType(attType, jsonDict)
    if existence:    
        for k, v in jsonDict.items():
            try:
                interOutput[attType] = jsonDict[attType]
            except:
                interOutput[k] = checkAttackerType(attType, v)
        output = interOutput[attType]
    else:
        output = jsonDict
    return output

def convKeyInt(jsonDict):
    output = {}
    try:    
        for k, v in jsonDict.items():
            try:
                if isinstance(v, dict):
                    output[int(k)] = convKeyInt(v)
                else:
                    output[int(k)] = v
            except:
                if isinstance(v, dict):
                    output[k] = convKeyInt(v)
                else:
                    output[k] = v
    except:
        output = jsonDict
    return output

def convertDec2Bin(arr, pad):
    """
    Arguments: 
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    arr = np.array([arr])
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(pad))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [pad], dtype=np.int8)
    for bit_ix in range(0, pad):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret.reshape([-1,pad])[:,::-1]

def folderCreation(folder):
    mydir = os.path.join(
        os.getcwd(), folder,
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return mydir