import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text


class DatasetHandler():
    def __init__(self, dataset_name, model_name, max_tokens, batch_size, buffer_size):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def __call__(self, batch_config):
        # loads the model
        # check whether model doesn't exists.
        if not tf.saved_model.contains_saved_model(export_dir=f"__models__/__datasets__/{self.model_name}"):
            # downloads the model
            self.download_model()

        # loads model as tokenizer
        self.tokenizer = tf.saved_model.load(f"__models__/__datasets__/{self.model_name}")

        # inits the batch configuration
        self.batch_config = batch_config(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens
        )

        # returns the loaded dataset
        return tfds.load(
            self.dataset_name,
            with_info=True,
            as_supervised=True
        )

    def make_batches(self, dataset):
        return (
            dataset
            .shuffle(self.buffer_size)
            .batch(self.batch_size)
            .map(self.batch_config, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    def download_model(self):
        tf.keras.utils.get_file(
            f"{self.model_name}.zip",
            f"https://storage.googleapis.com/download.tensorflow.org/models/{self.model_name}.zip",
            cache_dir='./__models__/__datasets__', cache_subdir='', extract=True
        )



def main():
    # === TESTING class: DatasetHandler ===
    from batch_configurations import Default_PT_EN_BatchConfig

    sample_dataset = DatasetHandler(
        dataset_name="ted_hrlr_translate/pt_to_en",
        model_name="ted_hrlr_translate_pt_en_converter",
        max_tokens=128,
        batch_size=64,
        buffer_size=20000
    )
    sample_examples, sample_metadata = sample_dataset(batch_config=Default_PT_EN_BatchConfig)

    # create training and validation set batches.
    sample_train_batches = sample_dataset.make_batches(sample_examples['train'])
    sample_val_batches = sample_dataset.make_batches(sample_examples['validation'])

    # grab the first element from trainging batch
    for (pt, en), en_labels in sample_train_batches.take(1):
        break

    print(pt.shape)         # >>> (64, 62)
    print(en.shape)         # >>> (64, 58)
    print(en_labels.shape)  # >>> (64, 58)

    print(en[0][:10])           # >>> tf.Tensor([   2   72   82   76    9   55  154 1664   75  180], shape=(10,), dtype=int64)
    print(en_labels[0][:10])    # >>> tf.Tensor([  72   82   76    9   55  154 1664   75  180 6175], shape=(10,), dtype=int64)

if __name__ == "__main__":
    main()
