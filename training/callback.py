import os
import time

import tensorflow as tf

class DefaultCallback(tf.keras.callbacks.Callback):
    def __init__(self, path, model):
        self.path = path
        self.timestamp = int(time.time())
        self.model = model

        self.log_archive = list()

    def on_epoch_end(self, epoch, logs):
        # performance eval
        # normal
        loss_decrease=True
        masked_increase=True
        # val
        val_loss_decrease=True
        val_masked_increase=True

        # performance eval
        if len(self.log_archive) > 0:
            # normal
            loss_decrease       = logs['loss']       <= min(list(map(lambda x: x['loss'], self.log_archive)))
            masked_increase     = logs['masked']     <= max(list(map(lambda x: x['val_masked'], self.log_archive)))

            # val
            val_loss_decrease   = logs['val_loss']   <= min(list(map(lambda x: x['val_loss'], self.log_archive)))
            val_masked_increase = logs['val_masked'] <= max(list(map(lambda x: x['val_masked'], self.log_archive)))


        # combined mode
        if loss_decrease and masked_increase and val_loss_decrease and val_masked_increase:
            if self.model.encoder.trainable: self.model.encoder.save(
                filepath=f"{ self.path }/{ self.timestamp }-best_model.keras"
            )

            if self.model.decoder.trainable: self.model.decoder.save(
                filepath=f"{ self.path }/{ self.timestamp }-best_model.keras"
            )

            if os.environ.get('MIN_LOG_LEVEL') == '0':
                print('saving model: "best" ...')

        # val mode
        if val_loss_decrease and val_masked_increase:
            if self.model.encoder.trainable: self.model.encoder.save(
                filepath=f"{ self.path }/{ self.timestamp }-best_val_model.keras"
            )

            if self.model.decoder.trainable: self.model.decoder.save(
                filepath=f"{ self.path }/{ self.timestamp }-best_val_model.keras"
            )

            if os.environ.get('MIN_LOG_LEVEL') == '0':
                print('saving model: "val" ...')

        # normal mode
        if loss_decrease and masked_increase:
            if self.model.encoder.trainable: self.model.encoder.save(
                filepath=f"{ self.path }/{ self.timestamp }-best_nor_model.keras"
            )

            if self.model.decoder.trainable: self.model.decoder.save(
                filepath=f"{ self.path }/{ self.timestamp }-best_nor_model.keras"
            )

            if os.environ.get('MIN_LOG_LEVEL') == '0':
                print('saving model: "normal" ...')

        # appends the log to the archive
        self.log_archive.append(logs)
