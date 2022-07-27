import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

t = np.array([4, 8, 7])
t = t.reshape(1, -1)
y = np.array([0, 1, 6])
y = y.reshape(1, -1)


print(y[np.arange(3), t])

