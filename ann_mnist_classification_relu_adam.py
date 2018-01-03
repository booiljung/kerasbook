import numpy as np
from keras import layers, models, datasets
from keras.utils import np_utils
import matplotlib.pyplot as plt


def configure_ANN_model(Nin, Nh, Nout):
    x = layers.Input(shape=(Nin,))
    h = layers.Activation("relu")(layers.Dense(Nh)(x))
    y = layers.Activation("softmax")(layers.Dense(Nout)(h))
    model = models.Model(x, y)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def configure_ANN_sequential_model(Nin, Nh, Nout):
    model = models.Sequential()
    model.add(layers.Dense(Nh, activation="relu", input_shape=(Nin,)))
    model.add(layers.Dense(Nout, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

class ANNModel(models.Model):
    def __init__(self, Nin, Nh, Nout):
        x = layers.Input(shape=(Nin,))
        h = layers.Activation("relu")(layers.Dense(Nh)(x))
        y = layers.Activation("softmax")(layers.Dense(Nout)(x))
        super().__init__(x, y)
        self.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

class ANNSequentialModel(models.Sequential):
    def __init__(self, Nin, Nh, Nout):
        super().__init__()
        self.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def load_data():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)

def plot_accuracy(history, title=None):
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history["acc"])
    plt.plot(history["val_acc"])
    if title is not None:
        plt.title(title)
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train data", "test data"], loc=0)

def plot_loss(history, title=None):
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train data', 'test data'], loc=0)

def train_and_test(model, X_train, Y_train, X_test, Y_test, subplot, title):
    history = model.fit(X_train, Y_train, epochs=15, batch_size=100, validation_split=0.2)
    performace_test = model.evaluate(X_test, Y_test, batch_size=100)
    print('Test Loss and Accuracy ->', performace_test)
    plt.subplot(subplot)
    plt.title(title)
    plot_loss(history)

if __name__ == "__main__":
    Nin = 784
    Nh = 100
    number_of_class = 10
    Nout = number_of_class

    (X_train, Y_train), (X_test, Y_test) = load_data()

    plt.figure(figsize=(16, 4), dpi=100)

    train_and_test(model=configure_ANN_model(Nin, Nh, Nout),
            X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,
            subplot=141, title="ANN_model")
    train_and_test(model=configure_ANN_sequential_model(Nin, Nh, Nout),
            X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,
            subplot=142, title="ANN_sequential_model")
    train_and_test(model=ANNModel(Nin, Nh, Nout),
            X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,
            subplot=143, title="ANNModel")
    train_and_test(model=ANNSequentialModel(Nin, Nh, Nout),
            X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,
            subplot=144, title="ANNSequentialModel")
    plt.show()
