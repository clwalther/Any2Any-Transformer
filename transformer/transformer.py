import tensorflow as tf

class Transformer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        y, x  = inputs

        y = self.encoder(y)
        x = self.decoder((y, x))

        return x

    def train(self, training_schedule, callbacks=None):
        for element in training_schedule:
            element.get_information()

            self.encoder = element.encoder
            self.decoder = element.decoder

            self.encoder.trainable = element.trainable_encoder
            self.decoder.trainable = element.trainable_decoder

            # fitting with element
            self.fit(
                x=element.training_data,
                epochs=element.epochs,
                steps_per_epoch=element.steps_per_epoch,
                validation_data=element.validation_data,
                callbacks=element.callbacks
            )

            # run end of element callback
            if callbacks is not None:
                for callback in callbacks: callback()
