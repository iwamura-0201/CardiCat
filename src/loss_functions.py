#########################
#### CardiCat ###########
#########################

import keras.backend as K
import tensorflow as tf
from keras import layers

from src.utils import flatten_list

def kl(mean, log_var):
    """Returns the KL distance for a standard normal(mean,log_var)

    Args:
        mean (tensor): the mean of the standard normal
        log_var (tensor): the log_var of the standard normal

    Returns:
        tensor: the KL distance
    """
    # -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    kl_loss = -0.5 * tf.reduce_sum(
        1 + log_var - tf.square(mean) - K.exp(log_var), axis=1
    )
    return tf.reduce_mean(kl_loss) / mean.shape[1]


def get_type_specific_loss(wMSE_tf, layer_sizes):
    """Returns the type specific contribution of the total wMSE loss.

    Args:
        wMSE_tf (tensor): a tensor of the wMSE loss per component, where
                according to type, different types reserve different amount
                of components.
        layer_sizes (list of lists): the size of vector representation for each
                feature/type.

    Returns:
        list: a list of the mean reduced contribution of each type.
    """

    # WHATS HERE?
    # If weighted loss is chosen, we need to duplicate 1hots:
    if wMSE_tf.shape[0] != sum(flatten_list(layer_sizes)):
        hot, rest = tf.split(
            wMSE_tf, [len(layer_sizes[0]), wMSE_tf.shape[0] - len(layer_sizes[0])]
        )
        wMSE_tf = tf.concat(
            [tf.repeat(hot, int(sum(layer_sizes[0]) / len(layer_sizes[0]))), rest],
            axis=-1,
        )

    hot_tmp, emb_tmp, num_tmp = tf.split(wMSE_tf, [sum(layer) for layer in layer_sizes])

    if layer_sizes[0]:
        hot_tmp = [
            tf.reduce_mean(t)
            for t in tf.split(hot_tmp, [layer for layer in layer_sizes[0]])
        ]  # if hot_tmp else []
    if layer_sizes[1]:
        emb_tmp = [
            tf.reduce_mean(t)
            for t in tf.split(emb_tmp, [layer for layer in layer_sizes[1]])
        ]  # if emb_tmp else []
    # num_tmp = [tf.reduce_mean(t) for t in tf.split(num_tmp,[l for l in layer_sizes[2] ])] #if num_tmp else []


    return [
        t.numpy() for t in flatten_list([hot_tmp, emb_tmp, num_tmp])
    ]  # specific_loss


def emb_var(emb_splits):
    """Returns a list of the variances of each embedded feature.

    Args:
        emb_splits (list of embedded weights (array)): the list of arrays
        (embedded weights) for each categorical embedded feature.

    Returns:
        list: Returns a list of the variances of each embedded feature.
    """
    emb_var_list = [
        tf.reduce_mean(tf.square(split - tf.reduce_mean(split, axis=1)))
        for split in emb_splits
    ]
    return emb_var_list


def separate_types(x, layer_sizes):
    """Splits a concatinated tensor into data types splits and applies type
    specific activation. Returning a list of tensors for each data type.

    Args:
        x (tensor): a (concatinated) tesor to be split into types
        layer_sizes (list(lists)): A list of lists with split sizes

    Returns:
        list(lists): A list of lists of split tensors.
    """

    softmax_layer = layers.Softmax()
    # relu_layer = layers.ReLU() #Very bad performance
    tanh_layer = tf.math.tanh  #

    # if len(layer_sizes)==2: # assuming no emb, but that's not necessarily true

    x_split = tf.split(
        x, (sum(layer_sizes[0]), sum(layer_sizes[1]), sum(layer_sizes[2])), axis=1
    )
    if layer_sizes[0]:
        hots = [
            softmax_layer(layer)
            for layer in tf.split(x_split[0], layer_sizes[0], axis=1)
        ]
    else:
        hots = []
    if layer_sizes[1]:
        # embs = [tanh_layer(layer) for layer in tf.split(x_split[1], layer_sizes[1], axis=1)]
        embs = [layer for layer in tf.split(x_split[1], layer_sizes[1], axis=1)]
    else:
        embs = []
    if layer_sizes[2]:
        nums = [
            tanh_layer(layer)
            # layer
            for layer in tf.split(  # tanh_layer(layer)
                x_split[2], layer_sizes[2], axis=1
            )
        ]
    else:
        nums = []
    # for embs/nums, tanh vs linear: best- Linear very good, tanh less good.
    return hots + embs + nums


def mse_loss_fun(x_true, x_pred, layer_sizes, weights, weighted_loss=False):
    """Returns the mean squared error for (x_true),(x_pred).

    Args:
        x_true (tensor): _description_
        x_pred (tensor): _description_
        layer_sizes (list(lists)): A list of lists with split sizes

    Returns:
        scalar,tensor: general MSE (for all feature), and a list of feature specific MSE
    """
    # if len(x_pred)==2: # Don't remember why we need this
    #     x_pred = tf.concat( [x_pred[0],x_pred[1]], axis=1)
    #     print("GOT TO THIS POINT DADDY!!!")

    # Separate x_pred into types and applies type specific activation
    x_pred = separate_types(x_pred, layer_sizes)
    x_pred = tf.concat(x_pred, axis=1)

    MSE_tf = tf.reduce_mean(tf.square(x_true - x_pred), axis=0)  # reduce over batch
    # MSE = tf.reduce_mean(MSE_tf) #reduce over columns/features (scalar)

    if weighted_loss:
        MSE = tf.reduce_sum(MSE_tf * weights) / tf.reduce_sum(weights)  # reduce_mean??
    else:
        MSE = tf.reduce_mean(MSE_tf)  # reduce over columns/features (scalar)

    return MSE, MSE_tf


# def recon_loss_fun(
#     x_true, x_pred, layer_sizes, weights, mixed_loss=False, weighted_loss=False
# ):
#     """Returns the mean squared error for (x_true),(x_pred).

#     Args:
#         x_true (tensor): _description_
#         x_pred (tensor): _description_
#         layer_sizes (list(lists)): A list of lists with split sizes

#     Returns:
#         scalar,tensor: general MSE (for all feature), and a list of feature specific MSE
#     """
#     # if len(x_pred)==2: # Don't remember why we need this
#     #     x_pred = tf.concat( [x_pred[0],x_pred[1]], axis=1)
#     #     print("GOT TO THIS POINT DADDY!!!")

#     # Separate x_pred into types and applies type specific activation
#     x_pred = separate_types(x_pred, layer_sizes)
#     if not mixed_loss:  # Naive MSE loss
#         x_pred = tf.concat(x_pred, axis=1)
#         MSE_tf = tf.reduce_mean(tf.square(x_true - x_pred), axis=0)  # reduce over batch

#     else:  # Mixed loss:
#         if layer_sizes[0]:  # 1hot vars
#             loss_fun_hot = tf.keras.losses.CategoricalCrossentropy()
#             # Split true into type tensors:
#             x_true_hot, x_true_num = tf.split(
#                 x_true,
#                 [sum(l) for l in [layer_sizes[0], flatten_list(layer_sizes[1:])]],
#                 axis=1,
#             )
#             print("x_true_hot shape", x_true_hot.shape)
#             x_true_hot_split = tf.split(x_true_hot, layer_sizes[0], axis=1)

#             # Split pred into types:
#             x_pred_hot_split = x_pred[: len(layer_sizes[0])]
#             x_pred_num = tf.concat(x_pred[len(layer_sizes[0]) :], axis=1)

#             # Calculating 1hot loss:
#             hot_losses = tf.concat(
#                 [
#                     loss_fun_hot(i, j)
#                     for i, j in zip(x_true_hot_split, x_pred_hot_split)
#                 ],
#                 axis=0,
#             )

#             # hot_losses = loss_fun_hot(x_true_hot_split,x_pred_hot_split)
#             print("hot_losses shape", hot_losses)
#             # hot_losses = tf.reduce_mean(hot_losses,axis=0) # over batch
#             # # Adjusting weights
#             weights_num = weights[sum(layer_sizes[0]) :]
#             weights_hot = [1] * len(layer_sizes[0])
#             weights = weights_hot + weights_num

#         else:
#             hot_losses = []
#             x_pred_num = pred
#             x_true_num = true

#         MSE_tf = tf.reduce_mean(
#             tf.square(x_true_num - x_pred_num), axis=0
#         )  # over batch
#         print("MSE_tf shape", MSE_tf.shape)
#         print("MSE_tf ", MSE_tf)
#         print("hot_losses shape", hot_losses.shape)
#         MSE_tf = tf.concat([hot_losses, MSE_tf], 0)

#     if weighted_loss:
#         MSE = tf.reduce_sum(MSE_tf * weights) / tf.reduce_sum(weights)  # reduce_mean??
#     else:
#         MSE = tf.reduce_mean(MSE_tf)  # reduce over columns/features (scalar)
#     return MSE, MSE_tf


