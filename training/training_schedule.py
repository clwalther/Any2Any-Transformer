import os

class TrainingScheduleElement():
    def __init__(self, encoder, decoder, training_data, validation_data, epochs,
                    steps_per_epoch, trainable_encoder=True, trainable_decoder=True, callbacks=None):
        # public
        self.encoder = encoder
        self.decoder = decoder

        self.trainable_encoder = trainable_encoder
        self.trainable_decoder = trainable_decoder

        self.training_data = training_data
        self.validation_data = validation_data
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.callbacks = callbacks

    def get_information(self):
        if os.environ.get('MIN_LOG_LEVEL') == '0':
            print()
            print("Training Schedule")
            print("_"*65)
            print("{:<27} {:<53}".format("Type", "Trainable"))
            print("="*65)
            print("{:<27} {:<53}".format(f"Encoder: {self.encoder.encoder_name}", str(self.encoder.trainable)))
            print()
            print("{:<27} {:<53}".format(f"Encoder: {self.decoder.decoder_name}", str(self.decoder.trainable)))
            print()
            print("="*65)
            print(f"Epochs: {self.epochs}")
            print(f"Steps per epoch: {self.steps_per_epoch}")
            print("_"*65)
            print()


