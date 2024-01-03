import os

class TrainingScheduleElement():
    def __init__(self, encoder_id, decoder_id, training_data,
                    validation_data, epochs, steps_per_epoch, callbacks=None):
        # public
        self.active_encoder     = encoder_id
        self.active_decoder     = decoder_id

        self.training_data      = training_data
        self.validation_data    = validation_data
        self.epochs             = epochs
        self.steps_per_epoch    = steps_per_epoch
        self.callbacks          = callbacks

    def get_information(self):
        if os.environ.get('MIN_LOG_LEVEL') == '0':
            print(f"_"*65)
