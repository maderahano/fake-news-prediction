import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "label"
FEATURE_KEY = "text"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Convert a label (0 or 1) into a one-hot vector
    Args:
        int: label_tensor (0 or 1)
    Returns
        label tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    """

    outputs = {}

    dim = 0
    int_value = tft.compute_and_apply_vocabulary(
        inputs[LABEL_KEY], top_k=dim + 1
    )

    label_feature = convert_num_to_one_hot(
        int_value, num_labels=dim + 1
    )

    # Convert text feature to lowercase
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])

    # Convert label values to integers
    outputs[transformed_name(LABEL_KEY)] = tf.cast(label_feature, tf.int64)


    return outputs
