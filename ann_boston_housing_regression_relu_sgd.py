from keras import layers, models, datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keraspp.skeras import plot_loss

class ANN(models.Model):
    def __init__(self, Nin, Nh, Nout):
        x = layers.Input(shape=(Nin,))
        h = layers.Activation("relu")(layers.Dense(Nh)(x))
        y = layers.Dense(Nout)(h)
        super().__init__(x, y)
        self.compile(loss="mse", optimizer="sgd")

def load_data():
    (X_train, y_train), (X_test, y_test) = datasets.boston_housing.load_data()
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    Nin = 13
    Nh = 5
    Nout = 1

    model = ANN(Nin, Nh, Nout)
    (X_train, y_train), (X_test, y_test) = load_data()

    history = model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=2)

    performace_test = model.evaluate(X_test, y_test, batch_size=100)
    print("\nTest Loss -> {:.2f}".format(performace_test))

    plt.figure(figsize=(5, 5), dpi=100)
    plot_loss(history)
    plt.show()
