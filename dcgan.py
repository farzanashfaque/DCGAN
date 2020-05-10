import numpy as np
import keras
import tqdm
import utils
from tqdm import tqdm
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model


class DCGAN:

    def __init__(self, learning_rate, batch_size, num_epochs, save_path):
        self.img_shape = (28, 28, 1)
        self.noise_dim = 100

        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_path = save_path

        self.discriminator = self.create_disc()
        self.generator = self.create_gen()
        z = Input(shape=(self.noise_dim,))
        image = self.generator(z)
        validity = self.discriminator(image)
        self.gan_model = Model(z, validity)

        self.discriminator.compile(keras.optimizers.Adam(10*self.lr), "binary_crossentropy")
        self.gan_model.compile(keras.optimizers.Adam(self.lr), "binary_crossentropy")

    def create_gen(self):

        model = Sequential()
        depth = 256
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        model.add(Dense(dim * dim * depth, input_dim=self.noise_dim))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Reshape((dim, dim, depth)))
        model.add(Dropout(0.4))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image
        model.add(Conv2DTranspose(1, 5, padding='same'))
        model.add(Activation('tanh'))
        # Reshape input into 7x7x256 tensor via a fully connected layer

        print("Generator model = ")
        model.summary()
        noise = Input(shape=(self.noise_dim,))
        img = model(noise)
        return Model(noise, img)


    def create_disc(self):

        model = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        model.add(Conv2D(depth * 1, 5, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(depth * 2, 5, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(depth * 4, 5, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(depth * 8, 5, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        print("Discriminator model = ")
        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    # function to pretrain the discriminator model to identify real and fake images
    def pretrain(self, x):
        size = x.shape[0]//200
        images, labels = self.gen_data(x, size)
        self.discriminator.fit(images, labels, batch_size=self.batch_size, epochs=1)
        print("Pretraining done!")


    # function to generate data for the discriminator to train on in every iteration
    def gen_data(self, x, size):
        shuffle = np.random.randint(0, x.shape[0], size)
        realimg = x[shuffle]
        noise = utils.gen_noise(size)
        fakeimg = self.generator.predict(noise)
        images = np.concatenate((realimg, fakeimg))
        label1 = np.ones((size, 1))
        label2 = np.zeros((size, 1))
        labels = np.concatenate((label2, label1))

        return images, labels



    def train(self, x, num_iter):

        for i in range(self.num_epochs):
            print("Epoch no :"+str(i+1)+"/"+str(self.num_epochs))

            for j in tqdm(range(num_iter)):

                x1, y = self.gen_data(x, self.batch_size//2)
                # train the discriminator
                self.discriminator.train_on_batch(x1, y)
                # Freeze the discriminator to train the GAN model
                utils.make_trainable(self.discriminator, False)
                # train the gan model
                inp = utils.gen_noise(self.batch_size//2)
                labels = np.zeros((self.batch_size//2, 1))
                self.gan_model.train_on_batch(inp, labels)

                # make the discriminator params back to trainable for the next iteration
                utils.make_trainable(self.discriminator, True)

            #save the weights and plot the results every 10 epochs
            if i % 10 == 0:
                self.gan_model.save_weights(self.save_path + str(i+1)+".h5")
                utils.plot(self.generator)
