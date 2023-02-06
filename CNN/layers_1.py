import numpy as np
from struct import unpack
import gzip

#对输入图像的预处理
def img2col(x, ksize, stride):
    wx, hx, cx = x.shape
    #wx是宽度,hx是高度,cx是通道数
    feature_w = (wx - ksize) // stride + 1
    #返回的特征图尺寸
    image_col = np.zeros((feature_w*feature_w, ksize*ksize*cx))
    num = 0
    for i in range(feature_w):
        for j in range(feature_w):
            image_col[num] =  x[i*stride:i*stride+ksize, j*stride:j*stride+ksize, :].reshape(-1)
            num += 1
    return image_col

#卷积层代码
class Conv(object):

    #初始化参数
    def __init__(self,kernel_shape,stride=1,padding=0):
        width,height,in_channel,out_channel = kernel_shape
        self.stride = stride
        self.padding = padding
        scale = np.sqrt((3*in_channel*width*height)/out_channel)
        self.k = np.random.standard_normal(kernel_shape) / scale
        self.b = np.random.standard_normal(out_channel) / scale
        self.k_gradient = np.zeros(kernel_shape)
        self.b_gradient = np.zeros(out_channel)

    #前向传播
    def forward(self,x):
        self.x = x
        if self.pad != 0:
            self.x = np.pad(self.x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), 'constant')
        bx, wx, hx, cx = self.x.shape
        wk, hk, ck, nk = self.k.shape
        # kernel的宽、高、通道数、个数
        feature_w = (wx - wk) // self.stride + 1
        # 返回的特征图尺寸
        feature = np.zeros((bx, feature_w, feature_w, nk))

        self.image_col = []
        kernel = self.k.reshape(-1, nk)
        for i in range(bx):
            image_col = img2col(self.x[i], wk, self.stride)
            feature[i] = (np.dot(image_col, kernel) + self.b).reshape(feature_w, feature_w, nk)
            self.image_col.append(image_col)
        return feature

    #反向传播
    def backward(self, delta, learning_rate):
        bx, wx, hx, cx = self.x.shape
        wk, hk, ck, nk = self.k.shape
        bd, wd, hd, cd = delta.shape

        # 计算self.k_gradient,self.b_gradient
        delta_col = delta.reshape(bd, -1, cd)
        for i in range(bx):
            self.k_gradient += np.dot(self.image_col[i].T, delta_col[i]).reshape(self.k.shape)
        self.k_gradient /= bx
        self.b_gradient += np.sum(delta_col, axis=(0, 1))
        self.b_gradient /= bx

        # 计算delta_backward
        delta_backward = np.zeros(self.x.shape)
        k_180 = np.rot90(self.k, 2, (0, 1))
        # numpy矩阵旋转180度
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

#RELU层
class Relu(object):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, delta):
        delta[self.x < 0] = 0
        return delta

#池化层
class Pool(object):
    def forward(self, x):
        b, w, h, c = x.shape
        feature_w = w // 2
        feature = np.zeros((b, feature_w, feature_w, c))
        self.feature_mask = np.zeros((b, w, h, c))
        # 记录最大池化时最大值的位置信息用于反向传播
        for bi in range(b):
            for ci in range(c):
                for i in range(feature_w):
                    for j in range(feature_w):
                        feature[bi, i, j, ci] = np.max(x[bi,i*2:i*2+2,j*2:j*2+2,ci])
                        index = np.argmax(x[bi,i*2:i*2+2,j*2:j*2+2,ci])
                        self.feature_mask[bi, i*2+index//2, j*2+index%2, ci] = 1
        return feature

    def backward(self, delta):
        return np.repeat(np.repeat(delta, 2, axis=1), 2, axis=2) * self.feature_mask

#全连接层
class Linear(object):
    def __init__(self, inChannel, outChannel):
        scale = np.sqrt(inChannel/2)
        self.W = np.random.standard_normal((inChannel, outChannel)) / scale
        self.b = np.random.standard_normal(outChannel) / scale
        self.W_gradient = np.zeros((inChannel, outChannel))
        self.b_gradient = np.zeros(outChannel)

    def forward(self, x):
        self.x = x
        x_forward = np.dot(self.x, self.W) + self.b
        return x_forward

    def backward(self, delta, learning_rate):
        ## 梯度计算
        batch_size = self.x.shape[0]
        self.W_gradient = np.dot(self.x.T, delta) / batch_size
        self.b_gradient = np.sum(delta, axis=0) / batch_size
        delta_backward = np.dot(delta, self.W.T)
        ## 反向传播
        self.W -= self.W_gradient * learning_rate
        self.b -= self.b_gradient * learning_rate

        return delta_backward

#softmax层
class Softmax(object):
    def cal_loss(self, predict, label):
        batchsize, classes = predict.shape
        self.predict(predict)
        loss = 0
        delta = np.zeros(predict.shape)
        for i in range(batchsize):
            delta[i] = self.softmax[i] - label[i]
            loss -= np.sum(np.log(self.softmax[i]) * label[i])
        loss /= batchsize
        return loss, delta

    def predict(self, predict):
        batchsize, classes = predict.shape
        self.softmax = np.zeros(predict.shape)
        for i in range(batchsize):
            predict_tmp = predict[i] - np.max(predict[i])
            predict_tmp = np.exp(predict_tmp)
            self.softmax[i] = predict_tmp / np.sum(predict_tmp)
        return self.softmax

#读取训练集
#读取训练图像
def __read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    return img

#读取标签
def __read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab

#处理读取到的训练图像
def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

#读热编码处理标签
def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def load_mnist(x_train_path, y_train_path, x_test_path, y_test_path, normalize=True, one_hot=True):
    '''读入MNIST数据集
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    Returns
    ----------
    (训练图像, 训练标签), (测试图像, 测试标签)
    '''
    image = {
        'train': __read_image(x_train_path),
        'test': __read_image(x_test_path)
    }

    label = {
        'train': __read_label(y_train_path),
        'test': __read_label(y_test_path)
    }

    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])


