import keras
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class Plotter(keras.callbacks.Callback):

    def __init__(self, monitor, scale='linear', plot_during_train=True, save_to_file=None):
        super().__init__()
        if plt is None:
            raise ValueError(
                "Must be able to import Matplotlib to use the Plotter.")
        self.scale = scale
        self.monitor = monitor
        self.plot_during_train = plot_during_train
        self.save_to_file = save_to_file
        plt.ion()
        self.fig = plt.figure()
        self.title = "{} per Epoch".format(self.monitor)
        self.xlabel = "Epoch"
        self.ylabel = self.monitor
        self.ax = self.fig.add_subplot(111, title=self.title,
                                       xlabel=self.xlabel, ylabel=self.ylabel)
        self.ax.set_yscale(self.scale)
        self.x = []
        self.y_train = []
        self.y_val = []
        # self.ax.plot(self.x, self.y_train, 'b-', self.x, self.y_val, 'g-')

    def on_train_end(self, logs={}):
        # plt.ioff()
        # plt.show()
        return

    def on_epoch_end(self, epoch, logs={}):
        # self.line1.set_ydata(logs.get('loss'))
        # self.line2.set_ydata(logs.get('val_loss'))
        self.x.append(len(self.x))
        self.y_train.append(logs.get(self.monitor))
        self.y_val.append(logs.get("val_" + self.monitor))
        self.ax.clear()
        # Set up the plot
        self.fig.suptitle(self.title)
        self.ax.set_yscale(self.scale)
        # Actually plot
        self.ax.plot(self.x, self.y_train, 'b-', self.x, self.y_val, 'g-')
        self.fig.canvas.draw()
        return
