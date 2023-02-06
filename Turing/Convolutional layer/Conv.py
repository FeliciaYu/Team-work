import numpy as np

def img2col(x, ksize, stride):
    wx, hx, cx = x.shape
    feature_w = (wx - ksize) // stride + 1
    #返回的特征图尺寸
    image_col = np.zeros((feature_w*feature_w, ksize*ksize*cx))
    num = 0
    for i in range(feature_w):
        for j in range(feature_w):
            image_col[num] =  x[i*stride:i*stride+ksize, j*stride:j*stride+ksize, :].reshape(-1)
            num += 1
    return image_col

#前向传播
def conv_forward(z,K,b,padding=(0,0),strides=(1,1)):
    """
    :param z: 卷积层矩阵,形状为（N,C,H,W)
    :param K: 卷积核，形状为（C,D,K1,K2)
    :param b: 偏置项
    :param padding:padding为0
    :param strides: 步长
    :return: 卷积结果
    """
    z_padding = np.lib.pad(z, ((0, 0), (0, 0),
                               (padding[0], padding[0]), (padding[1], padding[1])),
                           'constant',constant_values=0)
    #目的是获取和z形状一致其余元素为0的矩阵

    N,_,height,width = z_padding.shape
    #目的是获取以上参数作后用，N是batch_size,h为高，w为宽

    C,D,K1,K2 = K.shape
    #C为输入通道数，D为输出通道数，K1为卷积核高度，K2为卷积核宽度

    assert (height - K1) % strides[0] == 0, '步长不为1时，步长必须刚好能够被整除'
    assert (width - K2) % strides[1] == 0, '步长不为1时，步长必须刚好能够被整除'

    z_conv = np.zeros((N, D, 1 + (height - K1) // strides[0], 1 + (width - K2) // strides[1]))\
    #创建一个用于记录卷积结果的矩阵

    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(height - K1 + 1)[::strides[0]]:
                for w in np.arange(width - K2 + 1)[::strides[1]]:
                    z_conv[n, d, h // strides[0], w // strides[1]] = np.sum(
                        z_padding[n, :, h:h + K1, w:w + K2] * K[:, d]) + b[d]
    #卷积过程

    return np.maximum(0,z_conv)
    #用激活函数relu处理最终结果并返回

#反向传播
def backward(self, delta, learning_rate):
    bx, wx, hx, cx = self.x.shape
    #bx为batch_size,wx为宽度,hx为高度,cx为通道数
    wk, hk, ck, nk = self.k.shape
    #wk为宽度,hk为高度,ck为输入通道,nk为输出通道
    bd, wd, hd, cd = delta.shape
    #bd为batch_size,wd为宽度，hd为高度,cd为输出通道

    # 计算self.k_gradient,self.b_gradient
    delta_col = delta.reshape(bd, -1, cd)
    for i in range(bx):
        self.k_gradient += np.dot(self.image_col[i].T, delta_col[i]).reshape(self.k.shape)
    self.k_gradient /= bx
    self.b_gradient += np.sum(delta_col, axis=(0, 1))
    self.b_gradient /= bx

    # 计算delta_backward
    delta_backward = np.zeros(self.x.shape)
    k_180 = np.rot90(self.k, 2, (0, 1))  # numpy矩阵旋转180度
    k_180 = k_180.swapaxes(2, 3)
    k_180_col = k_180.reshape(-1, ck)

    if hd - hk + 1 != hx:
        pad = (hx - hd + hk - 1) // 2
        pad_delta = np.pad(delta, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    else:
        pad_delta = delta

    for i in range(bx):
        pad_delta_col = img2col(pad_delta[i], wk, self.stride)
        delta_backward[i] = np.dot(pad_delta_col, k_180_col).reshape(wx, hx, ck)

    # 反向传播
    self.k -= self.k_gradient * learning_rate
    self.b -= self.b_gradient * learning_rate

    return delta_backward
