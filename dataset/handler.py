import tensorflow_datasets as tfds
import tensorflow as tf


class DatasetHandler():
    def __init__(self, dataset_name, model_name, batch_size, buffer_size):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def __call__(self):
        # loads the model
        # check whether model doesn't exists.
        if not tf.saved_model.contains_saved_model(export_dir=f"__models__/__datasets__/{self.model_name}"):
            # downloads the model
            self.download_model()

        # loads model as tokenizer
        self.tokenizer = tf.saved_model.load(f"__models__/__datasets__/{self.model_name}")

        # returns the loaded dataset
        return tfds.load(
            self.dataset_name,
            with_info=True,
            as_supervised=True
        )

    def make_batches(self, dataset, batch_config):
        return (
            dataset
            .shuffle(self.buffer_size)
            .batch(self.batch_size)
            .map(batch_config, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    def download_model(self):
        tf.keras.utils.get_file(
            f"{self.model_name}.zip",
            f"https://storage.googleapis.com/download.tensorflow.org/models/{self.model_name}.zip",
            cache_dir='./__models__/__datasets__', cache_subdir='', extract=True
        )
