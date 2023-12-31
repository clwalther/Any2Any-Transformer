import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text


DOWNLAOD = False
MODEL_NAME = "ted_hrlr_translate_pt_en_converter"

MAX_TOKENS = 128

BUFFER_SIZE = 20000
BATCH_SIZE = 64

def download_dataset():
    tf.keras.utils.get_file(
        f"{MODEL_NAME}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{MODEL_NAME}.zip",
        cache_dir='.', cache_subdir='', extract=True
    )

def load_dataset():
    examples, metadata = tfds.load(
        'ted_hrlr_translate/pt_to_en',
        with_info=True,
        as_supervised=True
    )

    train_examples, val_examples = examples['train'], examples['validation']

    for pt_examples, en_examples in train_examples.batch(3).take(1):
        print('> Examples in Portuguese:')
        for pt in pt_examples.numpy():
                print(pt.decode('utf-8'))
        print()

        print('> Examples in English:')
        for en in en_examples.numpy():
            print(en.decode('utf-8'))

    tokenizers = tf.saved_model.load(f"__datasets__/{MODEL_NAME}")

    [item for item in dir(tokenizers.en) if not item.startswith('_')]

    print('> This is a batch of strings:')
    for en in en_examples.numpy():
        print(en.decode('utf-8'))

    encoded = tokenizers.en.tokenize(en_examples)

    print('> This is a padded-batch of token IDs:')
    for row in encoded.to_list():
        print(row)

    round_trip = tokenizers.en.detokenize(encoded)

    print('> This is human-readable text:')
    for line in round_trip.numpy():
        print(line.decode('utf-8'))

    print('> This is the text split into tokens:')
    tokens = tokenizers.en.lookup(encoded)
    tokens

    lengths = []

    for pt_examples, en_examples in train_examples.batch(1024):
        pt_tokens = tokenizers.pt.tokenize(pt_examples)
        lengths.append(pt_tokens.row_lengths())

        en_tokens = tokenizers.en.tokenize(en_examples)
        lengths.append(en_tokens.row_lengths())
        print('.', end='', flush=True)

    all_lengths = np.concatenate(lengths)

    plt.hist(all_lengths, np.linspace(0, 500, 101))
    plt.ylim(plt.ylim())
    max_length = max(all_lengths)
    plt.plot([max_length, max_length], plt.ylim())
    plt.title(f'Maximum tokens per example: {max_length}');

    return tokenizers, en, pt




# === FOR TRAINING ===
def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (pt, en_inputs), en_labels

def make_batches(ds):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))


def main():
    if DOWNLAOD: download_dataset()

if __name__ == "__main__":
    main()
