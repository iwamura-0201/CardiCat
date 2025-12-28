#########################
#### CardiCat ###########
#########################


import tensorflow as tf
from tensorflow.keras import layers

from src.utils import flatten_list


def encoder(all_inputs, all_features, latent_dim):
    """Returns the VAE encoder architecture

    Args:
        all_inputs (List[KerasTensor]): A list of KerasTensors of inputs
        all_features (KerasTensor): A concatination of all input features
        latent_dim (int): the dimension of the latent model

    Returns:
        tf.keras.Model: The VAE Encoder architecture model.
    """
    x = layers.Dense(128, activation="relu")(all_features)
    # x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    # x = layers.BatchNormalization()(x) # Bad performance
    # x = layers.Dense(128, activation="relu")(x)
    mean = layers.Dense(latent_dim, name="mean", activation="linear")(x)
    log_var = layers.Dense(latent_dim, name="log_var", activation="linear")(x)
    # relu/sigmoid in mean/log_var layers? --> bad performance
    enc_model = tf.keras.Model(all_inputs, (mean, log_var), name="Encoder")
    return enc_model


def decoder(all_inputs, all_features, layer_sizes):
    """Returns the VAE decoder architecture

    Args:
        all_inputs (List[KerasTensor]): A list of KerasTensors of inputs
        all_features (KerasTensor): A concatination of all input features
        layer_sizes (list): A list of the layer sizes for the output

    Returns:
        tf.keras.Model:  The VAE decoter architecture model.
    """
    x = layers.Dense(128, activation="relu")(all_features)
    # x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    # x = layers.BatchNormalization()(x) # Bad performance
    output = layers.Dense(sum(flatten_list(layer_sizes)), activation="linear")(x)
    # tanh,relu, linear --> what is best? --> linear
    dec_model = tf.keras.Model(all_inputs, output, name="Decoder")
    return dec_model
