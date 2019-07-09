from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
from mnistHelper import mnistHelper

x_train, _ = mnistHelper.load_data(train=True, blanks = False)
x_test, _ = mnistHelper.load_data(train=False, blanks = False)

x_train = x_train.reshape([len(x_train), 784])
x_test = x_test.reshape([len(x_test), 784])

x_train_noisy = mnistHelper.add_noise(x_train, noise_range=(-0.5, 0.5), clip=True)
##x_train_noisy = mnistHelper.add_noise(x_train, gaussian=True, gauss_params=(0, 0.17), clip=True)

for i in range(5):
        plt.imshow(x_train_noisy[i].reshape(28, 28))
        plt.show()

input_size = 784
hidden_size = 128
code_size = 32

input_img = keras.layers.Input(shape=(input_size,))
hidden_1 = keras.layers.Dense(hidden_size, activation='relu')(input_img)
code = keras.layers.Dense(code_size, activation='relu')(hidden_1)
hidden_2 = keras.layers.Dense(hidden_size, activation='relu')(code)
output_img = keras.layers.Dense(input_size, activation='sigmoid')(hidden_2)

autoencoder = keras.models.Model(input_img, output_img)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train, epochs=15, batch_size=512)

offset = np.random.rand(*x_test.shape) - 0.5
x_test_noisy = np.clip(x_test+offset, 0, 1)

decoded = autoencoder.predict(x_test_noisy[:10])
decoded = np.reshape(decoded, [10, 28, 28])

for i in range(10):
	before = np.reshape(x_test_noisy[i], [28, 28])
	plt.imshow(np.concatenate((before, decoded[i]), axis=1), cmap='gray')
	plt.show()
