import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

#font_location = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
#font_name = fm.FontProperties(fname=font_location).get_name()
#matplotlib.rc('font', family=font_name)

def save_history_history(fname, history_history, fold=''):
    np.save(os.path.join(fold, fname), history_history)


def load_history_history(fname, fold=''):
    history_history = np.load(os.path.join(fold, fname)).item(0)
    return history_history


def plot_acc(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train data', 'test data'], loc=0)


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


def plot_history(history):
    plt.subplot(121)
    plt.title("Accuracy")
    plot_acc(history)
    plt.subplot(122)
    plt.title("Loss")
    plot_loss(history)

    
def plot_loss_acc(history):
    plot_loss(history, '(a) loss')
    plt.show()            
    plot_acc(history, '(b) accuracy')
    plt.show()
    
    
def plot_acc_loss(history):
    plot_acc(history, '(a) accuracy')
    plt.show()
    plot_loss(history, '(b) loss')
    plt.show()
