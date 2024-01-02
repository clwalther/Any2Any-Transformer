import tensorflow as tf

class DefaultCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(
            filepath="__models__/__translator_v1__/model.keras"
        )
