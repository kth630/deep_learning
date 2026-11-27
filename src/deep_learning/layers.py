import numpy as np

def im2col(x, FH, FW, stride=1, pad=0):
    n, channels, height, width = x.shape
    out_h = (height + 2 * pad - FH) // stride + 1
    out_w = (width + 2 * pad - FW) // stride + 1
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    windows = np.empty((n, out_h, out_w, channels, FH, FW), dtype=x.dtype)

    for row in range(FH):
        for col in range(FW):
            pixels = padded[:, :, row:row + out_h * stride:stride,
                            col:col + out_w * stride:stride]
            windows[..., row, col] = pixels.transpose(0, 2, 3, 1)

    return windows.reshape(n * out_h * out_w, channels * FH * FW)



class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH)// self.stride)
        out_w = int(1 + (W + 2*self.pad - FW)// self.stride )
     
        self.x = x
        self.col = im2col(x, FH, FW, self.stride, self.pad)
        self.col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        return out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        
        col = im2col(x, self.pool_h, self.pool_w, self.stride, 0)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        out = np.max(col, axis=1)

        return out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        self.original_x_shape = x.shape
        N = x.shape[0]
        self.x = x.reshape(N, -1)
        return np.dot(self.x, self.W) + self.b

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return y