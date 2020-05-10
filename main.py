import utils
import tensorflow as tf
import dcgan
import keras

def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session()

def main(lr, batch_size, num_epochs, save_path):

    x,_,_,_ = utils.import_mnist()

    gan_model = dcgan.DCGAN(learning_rate=1e-4, batch_size=128, num_epochs=100, save_path='D:/Downloads')

    gan_model.pretrain(x)
    gan_model.train(x,num_iter=100)
    print("Training Done!")

if __name__ == "__main__":
    
    keras.backend.tensorflow_backend.set_session(get_session())
    lr = 1e-4
    batch_size = 128
    num_epochs = 100
    save_path = "D:/Downloads"

    main(lr, batch_size, num_epochs, save_path)
