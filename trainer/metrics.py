"""Helper functions for calculating performance metrics."""

import tensorflow as tf
import metrics

def precision(labels, logits, class_id):
    return tf.metrics.precision_at_k(
        labels=labels,
        predictions=logits,
        k=1,
        class_id=class_id)


def recall(labels, logits, class_id):
    return tf.metrics.recall_at_k(
        labels=labels,
        predictions=logits,
        k=1,
        class_id=class_id)


def metric_fn(labels, logits):
    predictions = tf.argmax(logits, axis=1)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions)

    # calculate accuracy for each class, then takes the mean of that
    mean_per_class_accuracy = tf.metrics.mean_per_class_accuracy(
        labels=labels,
        predictions=predictions,
        num_classes=6
    )

    labels = tf.cast(labels, tf.int64)

    output = {
        "accuracy": accuracy,
        "mean_per_class_accuracy": mean_per_class_accuracy
    }

    for i in range(0, 6):
        key = "precision_class_"+str(i)
        output[key] = precision(labels, logits, i)

    for i in range(0, 6):
        key = "recall_class_"+str(i)
        output[key] = recall(labels, logits, i)

    return output