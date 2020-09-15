from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, Flatten, Reshape
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
from matplotlib import pyplot
import os
from PIL import Image
from itertools import cycle

def define_discriminator(input_shape):
    model = Sequential()

    model.add(Conv2D(256,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1,activation="sigmoid"))

    opt = Adam(lr=0.0002,beta_1=0.5)
    model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])
    return model

def define_generator(latent_dim):
    model = Sequential()

    model.add(Dense(256*8*8,input_dim = latent_dim))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((8,8,256)))

    model.add(Conv2DTranspose(256,(4,4),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(256,(4,4),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(256,(4,4),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(3,(8,8),activation="tanh",padding="same"))
    return model

def define_gan(generator,discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(learning_rate=0.0002,beta_1=0.5)
    model.compile(loss="binary_crossentropy",optimizer=opt)
    return model

def load_real_samples(n):
    directory = "D:\Youssief\myProjects\MachineLearning\GANS\AnimeFaces\data\Faces"
    files = np.array(os.listdir(directory))
    ix = np.random.randint(0,files.shape[0],n)
    images = files[ix]
    x = np.array([np.array(Image.open(os.path.join(directory,fname))) for fname in images])
    y = np.ones((n,1))
    return x,y
def generate_fake_samples(generator,latent_dim,n):
    latent = generate_latent_variables(latent_dim,n)
    X = generator.predict(latent)
    Y = np.zeros((n,1))
    return X,Y

def generate_latent_variables(latent_dim,n):
     return np.random.randn(n*latent_dim).reshape(n,latent_dim)

def save_plot(examples, epoch, n=10):
    # plot images
    for i in range(n*n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis("off")
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, :], cmap="gray_r")
    # save plot to file
    filename = "D:\Youssief\myProjects\MachineLearning\GANS\AnimeFaces\generated_plot_e%03d.png" % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()
def summarize_performance(epoch, generator, discriminator,  latent_dim, n_samples=100):
    X_real,Y_real = load_real_samples(n_samples)
    X_fake,Y_fake = generate_fake_samples(generator,latent_dim,n_samples)

    _,acc_real = discriminator.evaluate(X_real,Y_real,verbose=0)
    _,acc_fake = discriminator.evaluate(X_fake,Y_fake,verbose=0)

    print(">Accuracy real: %.0f%%, fake: %.0f%%" % (acc_real*100, acc_fake*100))
    save_plot(X_fake, epoch)

    filename = "D:\Youssief\myProjects\MachineLearning\GANS\AnimeFaces\generator_model_%03d.h5" % (epoch + 1)
    generator.save(filename)

def train(discriminator,generator,gan,latent_dim,data,n_epochs=100,batch_size=256,n_eval=1):
    batches = cycle(iter(data))
    batches_per_epoch = int(21551/batch_size)
    half_batch = int(batch_size/2)
    for i in range(n_epochs):
        for j in range(batches_per_epoch):
            batch = next(batches)
            batch = (batch - 127.5) / 127.5
            ##Generate Samples
            X_real, Y_real = batch,np.ones((batch.shape[0],1))
            X_fake,Y_fake = generate_fake_samples(generator,latent_dim,half_batch)
            X, y = np.vstack((X_real, X_fake)), np.vstack((Y_real, Y_fake))
            ##Train discriminator
            d_loss, _ = discriminator.train_on_batch(X, y)
            ##Train GAN
            X_gan = generate_latent_variables(latent_dim,batch_size)
            Y_gan = np.ones((batch_size,1))
            g_loss = gan.train_on_batch(X_gan,Y_gan)
            print(">%d, %d/%d, d=%.3f, g=%.3f" % (i+1, j+1,batches_per_epoch, d_loss, g_loss))
        if (i+1) % n_eval == 0:
            summarize_performance(i, generator, discriminator, latent_dim)





generator = load_model("D:\Youssief\myProjects\MachineLearning\GANS\AnimeFaces\generator_model_020.h5")

X = generate_latent_variables(100,100)
X_fake = generator.predict(X)
X_fake = (X_fake+1)/2
save_plot(X_fake, 2)