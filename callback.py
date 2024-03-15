import logging

from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import Callback


class ProgressBar(Callback):
    def __init__(self, logger=None):
        self.logger = logger

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        if self.logger.handlers[1].level == logging.INFO:
            self.progress_bar = tqdm(total=self.params['epochs'], unit='回合')

    def on_epoch_end(self, epoch, logs=None):
        if self.logger.handlers[1].level == logging.INFO:
            self.progress_bar.update(1)

    def on_train_end(self, logs=None):
        if self.logger.handlers[1].level == logging.INFO:
            self.progress_bar.close()

# class PlotLosses(Callback):
#     def __init__(self):
#         self.fig, self.ax = plt.subplots()
#         self.x, self.losses, self.val_losses = [], [], []
#         self.line_losses, = self.ax.plot([], [], label='loss')
#         self.line_val_losses, = self.ax.plot([], [], label='val_loss')
#         self.ax.legend()
#
#     def on_train_begin(self, logs=None):
#         self.x, self.losses, self.val_losses = [], [], []
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.x.append(epoch)
#         self.losses.append(logs.get('loss'))
#         self.val_losses.append(logs.get('val_loss'))
#         self.line_losses.set_data(self.x, self.losses)
#         self.line_val_losses.set_data(self.x, self.val_losses)
#         self.ax.relim()
#         self.ax.autoscale_view()
#         plt.draw()
#         plt.pause(0.001)
#
#     def on_train_end(self, logs=None):
#         plt.close()
#
#
# from matplotlib.animation import FuncAnimation
#
#
# class PlotLossesAnimation(Callback):
#     def __init__(self):
#         self.fig, self.ax = plt.subplots()
#         self.x, self.losses, self.val_losses = [], [], []
#         self.line_losses, = self.ax.plot([], [], label='loss')
#         self.line_val_losses, = self.ax.plot([], [], label='val_loss')
#         self.ax.legend()
#         self.animation = None
#
#     def on_train_begin(self, logs=None):
#         self.x, self.losses, self.val_losses = [], [], []
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.x.append(epoch)
#         self.losses.append(logs.get('loss'))
#         self.val_losses.append(logs.get('val_loss'))
#
#         if self.animation is None:
#             self.animation = FuncAnimation(self.fig, self.update, interval=200, blit=True)
#
#     def update(self, frame):
#         self.line_losses.set_data(self.x, self.losses)
#         self.line_val_losses.set_data(self.x, self.val_losses)
#         self.ax.relim()
#         self.ax.autoscale_view()
#         return [self.line_losses, self.line_val_losses]
