from keras import layers, models
from keras import backend 
from keraspp.skeras import plot_history
import matplotlib.pyplot as plt
from cnn_mnist_classification_relu_rmsprop import DATA


def Conv2D(filters, kernel_size, padding='same', activation='relu'):
    return layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)


class AE(models.Model):
    def __init__(self, org_shape=(1, 28, 28)):
        original = layers.Input(shape=org_shape)
        x1 = Conv2D(4, (3, 3))(original)
        x2 = layers.MaxPooling2D((2, 2), padding='same')(x1)
        x3 = Conv2D(8, (3, 3))(x2)
        x4 = layers.MaxPooling2D((2, 2), padding='same')(x3)
        z = Conv2D(1, (7, 7))(x4)
        y1 = Conv2D(16, (3, 3))(z)
        y2 = layers.UpSampling2D((2, 2))(y1)
        y3 = Conv2D(8, (3, 3))(y2)
        y4 = layers.UpSampling2D((2, 2))(y3)
        y5 = Conv2D(4, (3, 3))(y4)
        decoded = Conv2D(1, (3, 3), activation='sigmoid')(y5)
        super().__init__(original, decoded)
        print("Compiling model...")
        self.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])


def show_ae(autoencoder, data):
    x_test = data.x_test
    decoded_imgs = autoencoder.predict(x_test)
    print(decoded_imgs.shape, data.x_test.shape)

    if backend.image_data_format() == 'channels_first':
        N, n_ch, n_i, n_j = x_test.shape
    else:
        N, n_i, n_j, n_ch = x_test.shape

    x_test = x_test.reshape(N, n_i, n_j)
    decoded_imgs = decoded_imgs.reshape(decoded_imgs.shape[0], n_i, n_j)
    
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()
    

if __name__ == '__main__':
    batch_size = 128
    epochs = 10
    data = DATA()
    autoencoder = AE(data.input_shape)
    history = autoencoder.fit(data.x_train, data.x_train,
                              epochs=epochs, batch_size=batch_size,
                              shuffle=True, validation_split=0.2)
    plt.figure(figsize=(10, 5), dpi=100)
    plot_history(history)
    plt.show()
    show_ae(autoencoder, data)
    plt.show()
