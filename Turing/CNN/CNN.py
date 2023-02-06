import numpy as np
import layers_1
import cv2
import gzip
from struct import unpack

#读取被识别的图像
img = cv2.imread('write.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度化处理图像

#读取数据集
x_train_path = './Mnist/train-images-idx3-ubyte.gz'
y_train_path = './Mnist/train-labels-idx1-ubyte.gz'
x_test_path = './Mnist/t10k-images-idx3-ubyte.gz'
y_test_path = './Mnist/t10k-labels-idx1-ubyte.gz'
(x_train, y_train), (x_test, y_test) = layers_1.load_mnist(x_train_path, y_train_path, x_test_path, y_test_path)

#训练数据集
train_image = (x_train,y_train)[0]
train_label = (x_train,y_train)[1]

conv1 = layers_1.Conv(kernel_shape=(5,5,1,6))
relu1 = layers_1.Relu()
pool1 = layers_1.Pool()
conv2 = layers_1.Conv(kernel_shape=(5,5,6,16))
relu2 = layers_1.Relu()
pool2 = layers_1.Pool()
nn = layers_1.Linear(256,10)
softmax = layers_1.Softmax()

lr = 0.01
batch = 3
for epoch in range(10):
    for i in range(0,60000,batch):
        X = train_image[i:i+batch]
        Y = train_label[i:i+batch]

        predict = conv1.forward(X)
        predict = relu1.forward(predict)
        predict = pool1.forward(predict)
        predict = conv2.forward(predict)
        predict = relu2.forward(predict)
        predict = pool2.forward(predict)
        predict = predict.reshape(batch, -1)
        predict = nn.forward(predict)

        loss, delta = softmax.cal_loss(predict, Y)

        delta = nn.backward(delta, lr)
        delta = delta.reshape(batch, 4, 4, 16)
        delta = pool2.backward(delta)
        delta = relu2.backward(delta)
        delta = conv2.backward(delta, lr)
        delta = pool1.backward(delta)
        delta = relu1.backward(delta)
        conv1.backward(delta, lr)
        print("Epoch-{}-{:05d}".format(str(epoch), i), ":", "loss:{:.4f}".format(loss))

        lr *= 0.95 ** (epoch + 1)
        np.savez("data2.npz", k1=conv1.k, b1=conv1.b, k2=conv2.k, b2=conv2.b, w3=nn.W, b3=nn.b)

#测试数据集
test_image = (x_test, y_test)[0]
test_label = (x_test, y_test)[1]
