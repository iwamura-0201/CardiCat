from alive_progress import alive_bar
import time,sys
import tensorflow as tf
from keras import layers
import numpy as np
import keras.backend as K

lib_path = "[USER FILL]/CardiCat_neurips_library/"
sys.path.insert(1, lib_path)
# Loading the CardiCat module:
from src import preprocessing as preprocessing,vae_model as vae


def mvae(uni_input,uni_feature,latent_dim,train_series,batch_size,epochs,optimizer,token=None):
    """
    """
    data = train_series.copy()

    ds = tf.data.Dataset.from_tensor_slices(dict(data))
    ds = ds.batch(batch_size)
    
    def encoder_uni(uni_input,uni_feature,latent_dim):    
        x = layers.Dense(50, activation="relu")(uni_feature)
        mean = layers.Dense(latent_dim, name='mean')(x)
        log_var = layers.Dense(latent_dim, name='log_var')(x)
        enc_model = tf.keras.Model(uni_input, (mean, log_var), name="Encoder")
        return enc_model

    def decoder_uni(input_decoder):    
        inputs = tf.keras.Input(shape=input_decoder, name='decoder_input_layer')
        x = layers.Dense(50, activation="relu")(inputs)
        if token:
            output = layers.Dense(token,activation="Softmax")(x)
        else:
            output = layers.Dense(1,activation="linear")(x)
        dec_model = tf.keras.Model(inputs, output, name="Decoder")
        return dec_model
    
    def codex_uni(uni_input,uni_feature):
        #x=layers.concatenate ()(all_features)
        codex_model = tf.keras.Model(uni_input,uni_feature)
        return codex_model

    def kl(mean, log_var):
        kl_loss =  -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis = 1)
        return kl_loss
    
    # This annotation (`@tf.function`) causes the function to be "compiled".
    @tf.function
    def train_step_mVAE(batch,enc,dec,cod,final,optimizer):
        with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
            mean, log_var = enc(batch, training=True) 
            kl_loss = kl(mean, log_var)
            latent = final([mean, log_var])
            generated_x = dec(latent, training=True)
            x = cod(batch, training=True) 
            # print(x)
            if token:    
                LF = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            else:
                LF =  tf.keras.losses.MeanSquaredError()
            # print(x.shape)
            # print(generated_x.shape)
            loss = kl_loss+1000*LF(x,generated_x)    

            gradients_of_enc = encoder.gradient(loss, enc.trainable_variables)
            gradients_of_dec = decoder.gradient(loss, dec.trainable_variables)
            optimizer.apply_gradients(zip(gradients_of_enc, enc.trainable_variables))
            optimizer.apply_gradients(zip(gradients_of_dec, dec.trainable_variables))

        return loss #cat_loss,emb_loss,num_loss,mse_loss,kl_loss

    def train_mvae(train_ds,enc,dec,cod,final,optimizer,epochs):
        with alive_bar(epochs,length=epochs, bar = 'bubbles', spinner = 'twirls',
                       force_tty=True,dual_line=False,) as bar:
            bar.title('Training mVAE')
            losses = []
            for epoch in range(epochs):
                start = time.time()
                i = 0
                loss_= []

                for batch in train_ds:
                    # print(batch)
                    i += 1
                    loss = train_step_mVAE(batch,enc,dec,cod,final,optimizer)
                    loss_.append(loss)


                epoch_loss=np.mean(loss_[:-1])
                losses.append(epoch_loss)
                bar.text('loss: {:,.2f}'.format(epoch_loss))
                bar()   

            bar.title('Training complete. Final loss: {:,.2f}'.format(np.mean(loss_[:-1])))
        return losses
    
    optimizer_uni = tf.keras.optimizers.Adam(learning_rate = 0.0005)
    final = vae.sampling(latent_dim,latent_dim)
    enc = encoder_uni(uni_input,uni_feature,latent_dim)
    input_decoder = (latent_dim,)
    dec = decoder_uni(input_decoder)
    cod = codex_uni(uni_input,uni_feature)
    # cod = vae.codex(all_inputs_1hot,all_features_1hot)
    output_loss = train_mvae(train_ds =ds,enc=enc,dec=dec,
                              cod=cod,final=final,optimizer=optimizer,epochs=epochs)
    return enc,final,dec
    


def multiVAE(inputs,features,latent_dim,data_input,batch_size,epochs,optimizer):
    """
    """
    data =data_input.copy()

    ds = tf.data.Dataset.from_tensor_slices(dict(data))
    ds = ds.batch(batch_size)
    
    def encoder_multi(inputs,features,latent_dim):    
        x = layers.Dense(200, activation="relu")(features)
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dense(50, activation="relu")(x)
        mean = layers.Dense(latent_dim, name='mean')(x)
        log_var = layers.Dense(latent_dim, name='log_var')(x)
        enc_model = tf.keras.Model(inputs, (mean, log_var), name="Encoder")
        return enc_model

    def decoder_multi(input_decoder):    
        inputs = tf.keras.Input(shape=input_decoder, name='decoder_input_layer')
        x = layers.Dense(200, activation="relu")(inputs)
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dense(100, activation="relu")(x)
        output = layers.Dense(len(data_input.columns),activation="linear")(x)
        dec_model = tf.keras.Model(inputs, output, name="Decoder")
        return dec_model
    
    def codex_multi(uni_input,uni_feature):
        codex_model = tf.keras.Model(inputs,features)
        return codex_model

    def kl(mean, log_var):
        kl_loss =  -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis = 1)
        return kl_loss
    
    # This annotation (`@tf.function`) causes the function to be "compiled".
    @tf.function
    def train_step_multiVAE(batch,enc,dec,cod,final,optimizer):
        with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
            mean, log_var = enc(batch, training=True) 
            kl_loss = kl(mean, log_var)
            latent = final([mean, log_var])
            generated_x = dec(latent, training=True)
            x = cod(batch, training=True) 
            LF =  tf.keras.losses.MeanSquaredError()
            loss = kl_loss+1000*LF(x,generated_x)    

            gradients_of_enc = encoder.gradient(loss, enc.trainable_variables)
            gradients_of_dec = decoder.gradient(loss, dec.trainable_variables)
            optimizer.apply_gradients(zip(gradients_of_enc, enc.trainable_variables))
            optimizer.apply_gradients(zip(gradients_of_dec, dec.trainable_variables))

        return loss #cat_loss,emb_loss,num_loss,mse_loss,kl_loss

    def train_multiVAE(train_ds,enc,dec,cod,final,optimizer,epochs):
        with alive_bar(epochs,length=epochs, bar = 'bubbles', spinner = 'twirls',
                       force_tty=True,dual_line=False,) as bar:
            bar.title('Training multiVAE')
            losses = []
            for epoch in range(epochs):
                start = time.time()
                i = 0
                loss_= []

                for batch in train_ds:
                    # print(batch)
                    i += 1
                    loss = train_step_multiVAE(batch,enc,dec,cod,final,optimizer)
                    loss_.append(loss)


                epoch_loss=np.mean(loss_[:-1])
                losses.append(epoch_loss)
                bar.text('loss: {:,.2f}'.format(epoch_loss))
                bar()   

            bar.title('Training complete. Final loss: {:,.2f}'.format(np.mean(loss_[:-1])))
        return losses
    
    optimizer_multi = tf.keras.optimizers.Adam(learning_rate = 0.0005)
    final = vae.sampling(latent_dim,latent_dim)
    enc = encoder_multi(inputs,features,latent_dim)
    input_decoder = (latent_dim,)
    dec = decoder_multi(input_decoder)
    cod = codex_multi(inputs,features)
    output_loss = train_multiVAE(train_ds =ds,enc=enc,dec=dec,
                              cod=cod,final=final,optimizer=optimizer,epochs=epochs)
    return enc,final,dec
    
def flatten(regular_list):
    return [item for sublist in regular_list for item in sublist]