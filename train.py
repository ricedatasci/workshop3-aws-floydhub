import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from plotter_callback import Plotter

import smartphone6 as sm6
from models import naive as model_func

SEED = 2263
np.random.seed(SEED)
SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 300
HIDDEN = 85
MODEL_FILE = os.path.join(
    "../models/", model_func.__name__ + str(HIDDEN) + ".h5")


if __name__ == "__main__":
    # Load the data
    x, y = sm6.load_train()

    # Split the data
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=SPLIT, random_state=np.random.randint(10000))
    train_index, val_index = next(sss.split(x, y))
    # Keras needs categorical data
    y = to_categorical(y)
    # Split using the train and val indicies
    xtr, xval = x[train_index], x[val_index]
    ytr, yval = y[train_index], y[val_index]
    # Clear x and y for memory
    x, y = None, None

    # This will save the best scoring model weights to the parent directory
    best_model = ModelCheckpoint(MODEL_FILE, monitor='val_acc', mode='max', verbose=1,
                                 save_best_only=True, save_weights_only=True)
    # These will plot the loss and accuracy during training
    acc_plotter = Plotter(monitor='acc', scale='linear')
    loss_plotter = Plotter(monitor='loss', scale='log')
    # Set up the callback
    callbacks = [best_model, acc_plotter, loss_plotter]

    # initialize and train the model
    model = model_func(xtr.shape[1], hidden=HIDDEN)
    fit = model.fit(xtr, ytr, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks,
                    validation_data=(xval, yval), shuffle=True)
