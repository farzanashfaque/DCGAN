import numpy as np
import matplotlib.pyplot as plt

def preprocess_img(x):
    x = x/255
    return x


def import_mnist(preprocess=True):
    
    print("Downloading MNIST data ... ", end = "")
    from keras.datasets import mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28,28,1)
    x_test = x_test.reshape(x_test.shape[0], 28,28,1)

    if preprocess == True:
        x_train = preprocess_img(x_train)
        x_test = preprocess_img(x_test)

    print("Done")
    return x_train, y_train, x_test, y_test


def gen_noise(size):
    return np.random.normal(scale=0.02, size=(size,100))


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def plot_large(img):
    fig1 = plt.figure(figsize=(4,4))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.imshow(img, cmap="gray")
    plt.show()

def plot(generator,n=5):
    img = np.zeros((n*28,1))
    for i in range(n):
        col = np.multiply(np.add(generator.predict(gen_noise(n)).reshape(n*28,28),1.0),255.0/2.0)
        img = np.concatenate((img,col), axis=1)
    plot_large(img)