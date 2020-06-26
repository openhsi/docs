import numpy as np


def minmax(it):
    min = max = None
    for val in it:
        if min is None or val < min:
            min = val
        if max is None or val > max:
            max = val
    return min, max


def NGaussFunc(x, *params):  # x0 pk width
    y = np.zeros_like(x)
    for i in range(0, len(params) - 1, 3):
        ctr = params[i]
        amp = params[i + 1]
        wid = params[i + 2]
        y = y + amp * np.exp(-((x - ctr) / wid) ** 2)
    return y + params[-1]

