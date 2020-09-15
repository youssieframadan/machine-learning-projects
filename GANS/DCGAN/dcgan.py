from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, Flatten, Reshape
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot

def define_discriminator(input_shape):
    model = Sequential()
    model.add(Conv2D(64,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1,activation="sigmoid"))
    opt = Adam(lr=0.0002,beta_1=0.5)
    model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])
    return model

def define_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128*7*7,input_dim = latent_dim))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))
    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(1,(7,7),activation="sigmoid",padding="same"))
    return model

def define_gan(generator,discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(learning_rate=0.0002,beta_1=0.5)
    model.compile(loss="binary_crossentropy",optimizer=opt)
    return model

def load_real_samples():
    (train_x,_),(_,_) = load_data()
    x = np.expand_dims(train_x,axis=-1)
    x = x.astype("float32")
    x = x/255.0
    return x

def generate_real_samples(dataset,n):
    ix = np.random.randint(0,dataset.shape[0],n)
    X = dataset[ix]
    Y = np.ones((n,1))
    return X,Y

def generate_fake_samples(generator,latent_dim,n):
    latent = generate_latent_variables(latent_dim,n)
    X = generator.predict(latent)
    Y = np.zeros((n,1))
    return X,Y

def generate_latent_variables(latent_dim,n):
     return np.random.randn(n*latent_dim).reshape(n,latent_dim)

def save_plot(examples, epoch, n=10):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis("off")
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap="gray_r")
    # save plot to file
    filename = "D:\Youssief\myProjects\MachineLearning\GANS\DCGAN\generated_plot_e%03d.png" % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()


def summarize_performance(epoch, generator, discriminator, dataset, latent_dim, n_samples=100):
    X_real,Y_real = generate_real_samples(dataset,n_samples)
    X_fake,Y_fake = generate_fake_samples(generator,latent_dim,n_samples)

    _,acc_real = discriminator.evaluate(X_real,Y_real,verbose=0)
    _,acc_fake = discriminator.evaluate(X_fake,Y_fake,verbose=0)

    print(">Accuracy real: %.0f%%, fake: %.0f%%" % (acc_real*100, acc_fake*100))
    save_plot(X_fake, epoch)

    filename = "D:\Youssief\myProjects\MachineLearning\GANS\DCGAN\generator_model_%03d.h5" % (epoch + 1)
    generator.save(filename)

def train(discriminator,generator,gan,latent_dim,data,n_epochs=100,batch_size=256,n_eval=10):
    batch_per_epoch = int(data.shape[0]/batch_size)
    half_batch = int(batch_size/2)
    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            ##Generate Samples
            X_real,Y_real = generate_real_samples(data,half_batch)
            X_fake,Y_fake = generate_fake_samples(generator,latent_dim,half_batch)
            X, y = np.vstack((X_real, X_fake)), np.vstack((Y_real, Y_fake))
            ##Train discriminator
            d_loss, _ = discriminator.train_on_batch(X, y)
            ##Train GAN
            X_gan = generate_latent_variables(latent_dim,batch_size)
            Y_gan = np.ones((batch_size,1))
            g_loss = gan.train_on_batch(X_gan,Y_gan)
            print(">%d, %d/%d, d=%.3f, g=%.3f" % (i+1, j+1, batch_per_epoch, d_loss, g_loss))
        if (i+1) % 10 == 0:
            summarize_performance(i, generator, discriminator, data, latent_dim)



latent_dim = 100

discriminator = define_discriminator((28,28,1))

generator = define_generator(latent_dim)

gan = define_gan(generator,discriminator)

dataset = load_real_samples()


train(discriminator,generator,gan,latent_dim,dataset)