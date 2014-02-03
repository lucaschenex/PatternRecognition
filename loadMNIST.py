from __future__ import division, print_function
from struct import unpack
import numpy as np

def product(seq):
    p = 1
    for i in seq:
        p *= i
    return p

def loadMINST(fhandle):
    header = unpack('>BBBB', fhandle.read(4))
    magic_number = header[2]
    dimensions   = header[3]
    if magic_number != 8:
        raise ValueError("Wrong datatype!")
    dims = unpack('>'+'I'*dimensions, fhandle.read(dimensions*4))
    n = product(dims)
    data = fhandle.read(n)
    if len(data) != n:
        raise ValueError("File too short!")
    arry = np.fromstring(data, dtype=np.uint8)
    arry = arry.reshape(dims)
    return arry




