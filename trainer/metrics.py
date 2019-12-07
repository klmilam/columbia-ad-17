"""Helper functions for calculating performance metrics."""

import tensorflow as tf


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


def get_f1_score(precision, recall):
    """Returns metric and update op for a single class's F1-Score.

    Args:
        precision: tuple of the precision metric and its update op.
        recall: tuple of the recall metric and its update op.
    Returns:
        Tuple of F1-score (for a single class) and its update op.
    """
    precision_without_nans = tf.where(
        tf.is_nan(precision[0]),
        tf.zeros_like(precision[0]), precision[0])
    recall_without_nans = tf.where(
        tf.is_nan(recall[0]),
        tf.zeros_like(recall[0]), recall[0])
    
    return (2.0 * precision_without_nans * recall_without_nans / (
            precision_without_nans + recall_without_nans + 1e-8),
            tf.group(precision[1], recall[1]))


def get_macro_avg_f1s(f1s):
    """Returns metric and update op for macro-averaging all classes' F1-Scores.

    This function requires 6 classes, since it assumes input `f1s` contains 6
    tuples of f1-score values and f1-score update ops (one tuple for each
    class).

    Args:
        f1s: Array of 6 tuples, each containing a single class's F1-Score and
            its update op.
    Returns:
        Tuple of F1-score (macro-averaged for all classes) and its update op.
    """
    return ((f1s[0][0] + f1s[1][0] + f1s[2][0] + f1s[3][0] + f1s[4][0] + f1s[5][0])/6,
            tf.group(f1s[0][1], f1s[1][1], f1s[2][1], f1s[3][1], f1s[4][1], f1s[5][1]))


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


    output = {
        "accuracy": accuracy,
        "mean_per_class_accuracy": mean_per_class_accuracy
    }
    
    labels = tf.cast(labels, tf.int64)

    f1s = []
    for i in range(0, 6):
        key = "precision_class_"+str(i)
        p = precision(labels, logits, i)
        output[key] = p

        key = "recall_class_"+str(i)
        r = recall(labels, logits, i)
        output[key] = r
        f1s.append(get_f1_score(p, r))
    output["macro_f1"] = get_macro_avg_f1s(f1s)
    return output
    