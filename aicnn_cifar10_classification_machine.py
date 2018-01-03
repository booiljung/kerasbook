import numpy as np
import keras as kr
from keras import datasets, backend
from keraspp.skeras import plot_history
from keraspp import aicnn
import matplotlib.pyplot as plt


assert kr.backend.image_data_format() == 'channels_last'


class Machine(aicnn.Machine):
    def __init__(self):
        (X, y), (x_test, y_test) = datasets.cifar10.load_data()
        super().__init__(X, y, nb_classes=10)

if __name__ == "__main__":
    m = Machine()
    m.run()
