import scipy
import numpy as np
from abc import ABCMeta


class Semiring(metaclass=ABCMeta):
    
    def __init__(self, addR, mulR, sumR, prodR, zeroR, oneR):
        self._add = addR
        self._multiply = mulR
        self._sum = sumR
        self._prod = prodR
        self._zero = zeroR
        self._one = oneR
    
    def add(self, a, b, *args, **kwargs):
        return self._add(a, b, *args, **kwargs)
    
    def multiply(self, a, b, *args, **kwargs):
        return self._multiply(a, b, *args, **kwargs)
    
    def sum(self, a, *args, **kwargs):
        return self._sum(a, *args, **kwargs)
    
    def prod(self, a, *args, **kwargs):
        return self._prod(a, *args, **kwargs)
    
    @property
    def one(self):
        return self._one
        
    @property
    def zero(self):
        return self._zero
    
class LogSemiring(Semiring):
    
    def __init__(self):
        Semiring.__init__(self, addR=np.logaddexp,
                          mulR=np.add,
                          sumR=scipy.misc.logsumexp,
                          prodR=np.sum,
                          zeroR=-np.inf,
                          oneR=0)
        
class DefaultSemiring(Semiring):
    
    def __init__(self):
        Semiring.__init__(self, addR=np.add,
                          mulR=np.multiply,
                          sumR=np.sum,
                          prodR=np.prod,
                          zeroR=0,
                          oneR=1)
