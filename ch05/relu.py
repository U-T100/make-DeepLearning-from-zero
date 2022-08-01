from turtle import forward
import numpy as np

class Relu:
    def __init__(self, x):
        self.mask = None
        self.x = x


    def forward(self):
        self.mask = (self.x <= 0)
        out = self.x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx

relu = Relu(np.array([[1.0, -0.5], [-2.0, 3.0]]))
f = relu.forward()
print(f)
