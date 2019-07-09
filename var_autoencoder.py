from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
from mnistHelper import mnistHelper

def sample_layer(args):
    mean, logstd = args
    batch = K.shape(mean)[0]
    dim = K.int_shape(mean)[1]
    eps = K.random_normal(shape=(batch,dim))
    return (mean + K.exp(0.5 * logstd) * eps)

def vae_loss(inputs, outputs):
    recon = keras.losses.binary_crossentropy(inputs, outputs)
    recon *= input_dim
    kl = 1 + logstd - K.square(mean) - K.exp(logstd)
    kl = -0.5 * K.sum(kl, axis=-1)
    total_loss = K.mean(recon + kl)
    return total_loss

x_train, y_train = mnistHelper.load_data(train=True)
x_test, y_test = mnistHelper.load_data(train=False)

x_train = x_train.reshape([len(x_train), 784])
x_test = x_test.reshape([len(x_test), 784])

input_dim = 784
hidden_dim = 128
latent_dim = 2


input_enc = keras.layers.Input(shape=(input_dim,))
hidden_enc = keras.layers.Dense(hidden_dim, activation='relu')(input_enc)
mean = keras.layers.Dense(latent_dim)(hidden_enc)
logstd = keras.layers.Dense(latent_dim)(hidden_enc)
code = keras.layers.Lambda(sample_layer, output_shape=(latent_dim,))([mean, logstd])
encoder = keras.models.Model(input_enc, [mean, logstd, code])

input_dec = keras.layers.Input(shape=(latent_dim,))
hidden_dec = keras.layers.Dense(hidden_dim, activation='relu')(input_dec)
output_dec = keras.layers.Dense(input_dim, activation='sigmoid')(hidden_dec)
decoder = keras.models.Model(input_dec, output_dec)

output_vae = decoder(encoder(input_enc)[2])
vae = keras.models.Model(input_enc, output_vae)
vae.compile(optimizer='adam', loss=vae_loss)
vae.fit(x_train, x_train, epochs=50, batch_size=512)


def visualize_classes():
    mean, _, _ = encoder.predict(x_test, batch_size=256)
    plt.figure(figsize=(12,10))
    plt.scatter(mean[:, 0], mean[:, 1], c=y_test)
    classbar = plt.colorbar()
    classbar.ax.get_yaxis().set_ticks([])
    for j, num in enumerate(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
	    classbar.ax.text(10, j, num, ha='left', va='center')
    classbar.ax.get_yaxis().labelpad = 15
    plt.xlabel('Latent X')
    plt.ylabel('Latent Y')
    plt.savefig('attempt2.png')
    plt.show()

def visualize_imgs():
    num_imgs = 30
    figure = np.zeros((28*num_imgs, 28*num_imgs))
    x_space = np.linspace(-4, 4, num_imgs)
    y_space = np.linspace(4, -4, num_imgs)

    for i, xi in enumerate(x_space):
        for j, yi in enumerate(y_space):
            pt = np.array([[-yi, -xi]])
            flat_img = decoder.predict(pt)
            img = flat_img.reshape(28, 28)
            figure[28*i: 28*(i+1), 28*j: 28*(j+1)] = img

    plt.figure(figsize=(10,10))
    img_range = np.arange(14, (num_imgs*28)+14, 28)
    grid_x = np.round(x_space, 1)
    grid_y = np.round(y_space, 1)
    plt.xticks(img_range, grid_x)
    plt.yticks(img_range, grid_y)
    plt.xlabel('Latent X')
    plt.ylabel('Latent Y')
    plt.imshow(figure, cmap='gray')
    plt.savefig('attempt2_gen.png')
    plt.show()
