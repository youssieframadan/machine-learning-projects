import numpy as np
from numpy import hstack
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
def generate_real_samples(n):
    X1 = 2 * np.random.rand(n) - 1
    X2 = 2*X1 + 1
    X1 = X1.reshape(n,1)
    X2 = X2.reshape(n,1)
    Y = np.ones((n, 1))
    X = hstack((X1,X2))
    return X,Y
def generate_latent_variables(n,latent_dim):
    return np.random.randn(n*latent_dim).reshape(n,latent_dim)

def define_discriminator():
    model = Sequential()
    model.add(Dense(25,activation="relu",kernel_initializer="he_uniform",input_shape=(2,)))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer="Adam",metrics=["accuracy"])
    return model

def generate_fake_samples(model, latent_dim, n):
    x_input = generate_latent_variables(n,latent_dim)
    x = model.predict(x_input)
    y = np.zeros((n,1))
    return x,y

def define_gan(generator,discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss="binary_crossentropy",optimizer="Adam")
    return model

def define_generator(latent_dim):
    model = Sequential()
    model.add(Dense(15,activation="relu",kernel_initializer="he_uniform",input_shape=(latent_dim,)))
    model.add(Dense(15,activation="relu"))
    model.add(Dense(2,activation="tanh"))
    return model

def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    # prepare real samples
    x_real, y_real = generate_real_samples(n)
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print(epoch, acc_real, acc_fake)
    # scatter plot real and fake data points
    plt.scatter(x_real[:, 0], x_real[:, 1], color="red")
    plt.scatter(x_fake[:, 0], x_fake[:, 1], color="blue")
    # save plot to file
    filename = "D:\Youssief\myProjects\MachineLearning\GANS\\1DGan\generated_plot_e%03d.png" % (epoch+1)
    plt.savefig(filename)
    plt.close()


def train(generator,discriminator,gan,latent_dim,n_epochs=10000,n_batch=128,n_eval=2000):
    half_batch = int(n_batch/2)
    for i in range(n_epochs):
        x_real,y_real = generate_real_samples(half_batch)
        x_fake,y_fake = generate_fake_samples(generator,latent_dim,half_batch)
        discriminator.train_on_batch(x_real,y_real)
        discriminator.train_on_batch(x_fake,y_fake)
        x_gan = generate_latent_variables(n_batch,latent_dim)
        y_gan = np.ones((n_batch,1))
        gan.train_on_batch(x_gan,y_gan)
        if (i+1) % n_eval == 0:
            summarize_performance(i+1,generator,discriminator,latent_dim)



# size of the latent space
latent_dim = 5
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# train model
train(generator, discriminator, gan_model, latent_dim)