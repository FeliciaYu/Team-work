import numpy as np

#前向过程
def max_pooling_forward(z,pooling,strides=(2,2),padding=(0,0)):
    """
    最大池化前向过程
    :param z: 卷积层矩阵
    :param pooling: 池化大小
    :param strides:步长
    :param padding: 用0填充
    :return:
    """

    N,H,W,C = z.shape
    #N为batch_side,C为通道数,H为高,W为宽
    #目的是获取以上参数作后用

    z_padding = np.lib.pad(z,((0,0),(0,0),
                              (padding[0],padding[0]),(padding[1],padding[1])),
                           'constant',constant_values=0)
    #目的是在图像四周边缘填充0，避免因为卷积运算导致输出图像缩小和图像边缘信息丢失

    h = (H + 2 * padding[0] - pooling[0]) // (strides[0] + 1)
    w = (W + 2 * padding[1] - pooling[1]) // (strides[1] + 1)
    #目的是求出输出的高度和宽度

    z_pool = np.zeros((N,C,h,w))
    #目的是创建用于输出的矩阵

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(h):
                for j in np.arange(w):
                    z_pool[n,c,i,j] = np.max(z_padding[n,c,
                                             strides[0]*i:strides[0] * i + pooling[0],
                                             strides[1]*j:strides[1] * j + pooling[1]])
    #目的是选出每个子区域的最大值然后记录到用于输出的矩阵
    return z_pool

#反向过程
def max_pooling_backward(next_dz,z,pooling,strides = (2,2),padding = (0,0)):
    """
    最大池化反向过程
    :param next_dz: 损失函数关于最大池化输出的损失
    :param z: 卷积层矩阵
    :param pooling: 池化大小
    :param strides: 步长
    :param padding: 用0填充
    :return:
    """

    N,H,W,C = z.shape
    _,_,h,w = next_dz.shape
    # N为batch_side,C为通道数,H为高,W为宽
    # 目的是获取以上参数作后用

    z_padding = np.lib.pad(z,((0, 0), (0, 0),
                              (padding[0], padding[0]), (padding[1], padding[1])),
                           'constant',constant_values=0)
    # 目的是在图像四周边缘填充0，避免因为卷积运算导致输出图像缩小和图像边缘信息丢失
    dz_padding = np.zeros_like(z_padding)
    #零填充后的梯度，目的同上

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(h):
                for j in np.arange(w):
                    flat_idx = np.argmax(z_padding[n, c,
                                         strides[0] * i:strides[0] * i + pooling[0],
                                         strides[1] * j:strides[1] * j + pooling[1]])
                    h_idx= (strides[0] * i +flat_idx)//pooling[1]
                    w_idx = (strides[1] * j) + (flat_idx % pooling[1])
                    dz_padding[n,c,h_idx,w_idx] += next_dz[n,c,i,j]
    #目的是找到最大值的那个元素坐标，将梯度传给这个坐标

    result = dz_padding.remove(padding)
    #目的是返回结果时将零填充部分剔除

    return result
