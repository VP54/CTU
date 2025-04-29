### Abstract class for models
from abc import ABC
from logging import Logger

class Model(ABC):
    def _encode(self):
        pass
    
    def predict(self):
        raise NotImplementedError("Implement method for predicting")
    
    def score(self):
        raise NotImplementedError("Implement scoring methods")
    