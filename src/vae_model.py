#########################
#### CardiCat ###########
#########################


import time
from types import SimpleNamespace

import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from alive_progress import alive_bar
from keras import layers
from keras.losses import MeanSquaredError as MSE

from .loss_functions import kl, mse_loss_fun, get_type_specific_loss


def codex(all_inputs, all_features):
    """_summary_

    Args:
        all_inputs (_type_): _description_
        all_features (_type_): _description_

    Returns:
        _type_: _description_
    """
    # x=layers.concatenate ()(all_features)
    codex_model = tf.keras.Model(all_inputs, all_features)
    return codex_model


def sampling_model(distribution_params):
    """The VAE sampling model (function).

    Args:
        distribution_params (list): A list of mean and log-var arrays

    Returns:
        np.array: An array representing the z-model.
    """

    mean, log_var = distribution_params
    epsilon = tf.random.normal(shape=tf.shape(mean), mean=0.0, stddev=1.0)
    return mean + tf.exp(log_var / 2) * epsilon  # CHECK


def sampling(input_1, input_2):
    """_summary_

    Args:
        input_1 (_type_): _description_
        input_2 (_type_): _description_

    Returns:
        tf.keras.Model: _description_
    """

    mean = tf.keras.Input(shape=(input_1,), name="input_layer1")
    log_var = tf.keras.Input(shape=(input_2,), name="input_layer2")
    out = layers.Lambda(sampling_model, name="encoder_output", output_shape=(input_1,))(
        [mean, log_var]
    )
    enc_2 = tf.keras.Model([mean, log_var], out, name="Encoder_sampling")
    return enc_2


def set_cond_weights(enc, dec, embCols):
    enc_layers_dict_name_indx = {layer.name: en for en, layer in enumerate(enc.layers)}
    dec_layers_dict_name_indx = {layer.name: en for en, layer in enumerate(dec.layers)}
    for col in embCols:
        # setting cond embedding layers to inherit from embedding layers
        emb_layer_indx = enc_layers_dict_name_indx["emb_" + col]
        enc_cond_layer_indx = enc_layers_dict_name_indx["cond_dense_" + col]
        dec_cond_layer_indx = dec_layers_dict_name_indx["cond_dense_" + col]
        enc.layers[enc_cond_layer_indx].set_weights(
            enc.layers[emb_layer_indx].get_weights()
        )
        dec.layers[dec_cond_layer_indx].set_weights(
            enc.layers[emb_layer_indx].get_weights()
        )


# This annotation (`@tf.function`) causes the function to be "compiled".
@tf.function
def train_step(
    batch,
    enc,
    dec,
    cod,
    final,
    optimizer,
    layer_sizes,
    param_dict,
    weights,
    emb_init_var,
    tmpEmb,
):
    """_summary_

    Args:
        batch (_type_): _description_
        enc (_type_): _description_
        dec (_type_): _description_
        cod (_type_): _description_
        final (_type_): _description_
        optimizer (_type_): _description_
        layer_sizes (_type_): _description_
        param_dict (_type_): _description_
        weights (_type_): _description_

    Returns:
        _type_: _description_
    """
    d = SimpleNamespace(**param_dict)

    with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
        # Get forward pass of encoder:
        # setting cond emb weights according to entity embedding
        set_cond_weights(enc, dec, tmpEmb)

        mean, log_var = enc(batch, training=True)
        kl_loss = d.kl_factor * kl(mean, log_var)
        latent = final([mean, log_var])

        cond = {"cond_" + key: batch["cond_" + key] for key in tmpEmb}
        # get forward pass of decoder:
        if not tmpEmb:
            generated_x = dec(latent, training=True)
        else:
            generated_x = dec(cond, training=True)
        # Normalize original x for reconstruction term:
        x = cod(batch, training=False)
        batch_actual_size = x.shape[0]
        # Calculate loss
        # mse_loss, MSE_tf = recon_loss_fun(x, generated_x,layer_sizes, weights,
        #                     mixed_loss=d.mixed_loss,weighted_loss=d.weighted_loss)
        mse_loss, MSE_tf = mse_loss_fun(
            x, generated_x, layer_sizes, weights, weighted_loss=d.weighted_loss
        )
        loss_logs = get_type_specific_loss(MSE_tf, layer_sizes)
        mse_loss_factor = d.recon_factor * mse_loss
        loss_dict = {"mse": mse_loss, "mse_factor": mse_loss_factor, "kl": kl_loss}
        loss = mse_loss_factor + kl_loss

        ## Adding embedding regularization:
        emb_weights = {}
        for layer in enc.layers:
            if layer.name.startswith("emb_") and hasattr(layer, "trainable_weights") and layer.trainable_weights:
                emb_weights[layer.name] = layer.trainable_weights[0]
        tmp = [tf.math.reduce_variance(emb_weights[emb]) for emb in emb_weights.keys()]
        emb_mse = MSE()
        emb_reg_loss = emb_mse(tmp, emb_init_var)
        loss_dict["emb_reg_loss"] = emb_reg_loss
        if d.emb_regularization:
            loss = loss + emb_reg_loss * d.emb_reg_factor

    # Registering gradient tape:
    gradients_of_enc = encoder.gradient(loss, enc.trainable_variables)
    gradients_of_dec = decoder.gradient(loss, dec.trainable_variables)
    
    # Combine gradients and variables for a single optimizer call
    gradients = gradients_of_enc + gradients_of_dec
    variables = enc.trainable_variables + dec.trainable_variables
    optimizer.apply_gradients(zip(gradients, variables))

    return loss_dict, loss_logs, batch_actual_size


def train_high_vae(
    train_ds,
    enc,
    dec,
    cod,
    final,
    optimizer,
    layer_sizes,
    param_dict,
    weights,
    emb_init_var,
    tmpEmb,
):
    """_summary_

    Args:
        train_ds (_type_): _description_
        enc (_type_): _description_
        dec (_type_): _description_
        cod (_type_): _description_
        final (_type_): _description_
        optimizer (_type_): _description_
        layer_sizes (_type_): _description_
        param_dict (_type_): _description_
        weights (_type_): _description_

    Returns:
        _type_: _description_
    """
    print("\n### CardiCat: Training {} ###".format(param_dict["model"]))
    d = SimpleNamespace(**param_dict)
    # logging:
    losses = []
    loss_logs_all = []
    mse_losses = []
    emb_reg_losses = []
    mse_losses_factor = []

    kl_losses = []

    # Start mini-batch SGD:
    with alive_bar(
        d.epochs,
        length=10,
        bar="bubbles",
        spinner="twirls",
        force_tty=True,
        dual_line=False,
        max_cols=120,
    ) as bar:  #
        bar.title("Training")
        ## ##
        for epoch in range(d.epochs):
            # start = time.time()
            # i = 0
            loss_ = []
            loss_logs_ = []
            epoch_loss_dict = {"mse": [], "mse_factor": [], "kl": [], "emb_reg": []}

            ## ##
            for batch in train_ds:
                if d.is_target:
                    batch = batch[0]  # the batch is a (x,y) tuple in that case.
                ## train step--> for each batch:
                loss_dict, loss_logs, batch_actual_size = train_step(
                    batch,
                    enc,
                    dec,
                    cod,
                    final,
                    optimizer,
                    layer_sizes,
                    param_dict,
                    weights,
                    emb_init_var,
                    tmpEmb,
                )
                loss_logs_.append(loss_logs)

                loss = loss_dict["mse_factor"] + loss_dict["kl"]
                loss_.append(loss)

                epoch_loss_dict["mse"].append(loss_dict["mse"])
                epoch_loss_dict["mse_factor"].append(loss_dict["mse_factor"])

                epoch_loss_dict["kl"].append(loss_dict["kl"])
                epoch_loss_dict["emb_reg"].append(loss_dict["emb_reg_loss"])

            # if d.loss_type=='weighted_loss' or d.loss_type=='mixed_loss':
            #     loss_logs_all.append(tf.reduce_mean(loss_logs_,axis=0))
            # else:
            #     loss_logs_all.append(loss_logs_)
            loss_logs_all.append(tf.reduce_mean(loss_logs_, axis=0))
            batch_weights = [1] * len(loss_[:-1]) + [batch_actual_size / d.batch_size]
            epoch_loss = np.average(loss_, weights=batch_weights)

            epoch_mse_loss = np.average(epoch_loss_dict["mse"], weights=batch_weights)
            epoch_mse_loss_factor = np.average(
                epoch_loss_dict["mse_factor"], weights=batch_weights
            )
            mse_losses.append(epoch_mse_loss)
            mse_losses_factor.append(epoch_mse_loss_factor)

            emb_reg_loss = np.average(epoch_loss_dict["emb_reg"], weights=batch_weights)
            emb_reg_losses.append(emb_reg_loss)
            epoch_kl_loss = np.average(epoch_loss_dict["kl"], weights=batch_weights)
            kl_losses.append(epoch_kl_loss)
            bar.text(
                "Losses: Ttl: {:,.2f}, Rec: {:,.2f}({:,.2f}), KL: {:,.2f}, Emb: {:,.2f}".format(
                    loss,
                    loss - loss_dict["kl"],
                    (loss - loss_dict["kl"]) / d.recon_factor,
                    loss_dict["kl"],
                    loss_dict["emb_reg_loss"],
                )
            )
            bar()
            losses.append(epoch_loss)

            ## TEST EMB WEIGHTS
            emb_weights = {}
            for layer in enc.layers:
                if (
                    layer.name.startswith("emb_")
                    and hasattr(layer, "trainable_weights")
                    and layer.trainable_weights
                ):
                    emb_weights[layer.name] = layer.trainable_weights[0].numpy()

        bar.title("Training complete. Final loss: {:,.2f}".format(np.mean(loss_[:-1])))
        bar.title(
            "Final Losses: Ttl: {:,.2f}, Rec: {:,.2f}({:,.2f}), KL: {:,.2f}, Emb: {:,.2f}".format(
                loss,
                loss - loss_dict["kl"],
                (loss - loss_dict["kl"]) / d.recon_factor,
                loss_dict["kl"],
                loss_dict["emb_reg_loss"],
            )
        )

        loss_df = pd.DataFrame(
            data={
                "epoch": range(d.epochs),
                "total_loss": losses,
                "mse_loss": mse_losses,
                "mse_loss_factor": mse_losses_factor,
                "emb_reg_loss": emb_reg_losses,
                "kl_loss": kl_losses,
            }
        )

    return loss_df, loss_logs_all, emb_weights  # ,df_melt
