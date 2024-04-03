import logging

from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import Callback
from tensorflow.keras.losses import sparse_categorical_crossentropy

from EWC import ewc
import tensorflow as tf

from loss import tilde_q_loss


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


def compile_model(model, learning_rate, extra_losses=None):
    def custom_loss(y_true, y_pred):
        loss = sparse_categorical_crossentropy(y_true, y_pred)
        if extra_losses is not None:
            for fn in extra_losses:
                loss += fn(model)

        return loss

    model.compile(
        loss=custom_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )


class ewcCallback(Callback):
    def __init__(self, train_X, train_Y, model):
        self.regularisers = []
        self.model = model
        self.train_X = train_X
        self.train_Y = train_Y

    def on_epoch_end(self, epoch, logs=None):
        loss_fn = ewc.ewc_loss(0.1, self.model, (self.train_X, self.train_Y),
                               100)
        self.regularisers.append(loss_fn)
        # compile_model(self.model, 0.001, extra_losses=self.regularisers)

    # def on_train_end(self, logs=None):
    #     self.regularisers = []

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
