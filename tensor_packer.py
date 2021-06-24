import numpy as np


class TensorPacker:
    __slots__ = ('shapes', 'length')

    def __init__(self, shapes):
        self.shapes = shapes
        self.length = sum([np.product(shape) for shape in shapes])

    def pack(self, parts, np):
        """
        @param parts    sequence of tensors to ravel and concatentate into a vector
        @param np   Some implementation of the numpy api, such as jax.numpy or the ordinary numpy.
        """
        return np.concatenate([part.ravel() for part in parts])

    def unpack(self, x):
        res = []
        start = 0
        for shape in self.shapes:
            l = np.product(shape)
            res.append(x[start : start+l].reshape(shape))
            start += l
        return res
        
