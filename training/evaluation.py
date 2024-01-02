import tensorflow as tf

class Loss():
    @staticmethod
    def masked(label, pred):
        mask = label != 0

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
        )

        loss = loss_object(label, pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
        return loss

class Accuarcy():
    @staticmethod
    def masked(label, pred):
        pred = tf.argmax(pred, axis=2)
        label = tf.cast(label, dtype=pred.dtype)
        match = label == pred

        mask = label != 0

        match = match & mask

        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(match)/tf.reduce_sum(mask)
