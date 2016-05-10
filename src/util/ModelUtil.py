import numpy as np

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])

def error_count(predictions, labels):
    return np.sum(np.argmax(predictions, 1) != np.argmax(labels, 1))