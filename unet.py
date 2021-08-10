import tensorflow as tf 
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

#Encoder (Downsampling Block)
def conv_block(inputs , n_filters, dropout_prob=0 , max_pooling = True):
    """
    Convolutional downsampling block
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """
    conv1 = Conv2D(filters = n_filters , 
                  kernel_size = 3,
                  activation="relu",
                  padding="same",
                  kernel_initializer="he_normal")(inputs)
    conv2 = Conv2D(filters = n_filters , 
                  kernel_size = 3,
                  activation="relu",
                  padding="same",
                  kernel_initializer="he_normal")(conv1)
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv2)
        
    if max_pooling:
        next_layer = MaxPooling2D(pool_size = (2,2))(conv2)
    else :
        next_layer = conv2
        
    skip_connection = conv2
    return next_layer , skip_connection


#Decoder (Upsampling Block)
def upsampling_block(expansive_input , contractive_input , n_filters):
    """
    Convolutional upsampling block
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """
    up  = Conv2DTranspose(filters = n_filters ,
                         kernel_size =  3 ,
                         strides=2,
                         padding="same")(expansive_input)
    merge = concatenate([up, contractive_input], axis=3)
    conv1 = Conv2D(filters = n_filters,
                 kernel_size = 3,
                 activation="relu",
                 padding='same',
                 kernel_initializer="he_normal")(merge)
    conv = Conv2D(filters = n_filters,
                 kernel_size = 3,
                 activation= "relu",
                 padding="same",
                 kernel_initializer="he_normal")(conv1)
    return conv


def unet_model(input_size, n_classes, n_filters=32):
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """
    inputs = Input(input_size)
    cblock1 = conv_block(inputs,n_filters)
    cblock2 = conv_block(cblock1[0], 2*n_filters)
    cblock3 = conv_block(cblock2[0], 4*n_filters)
    cblock4 = conv_block(cblock3[0], 8*n_filters, dropout_prob=0.3)
    cblock5 = conv_block(cblock4[0], 16*n_filters, dropout_prob=0.3, max_pooling=False) 
    ublock6 = upsampling_block(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters )
    
    conv9 = Conv2D(filters = n_filters,
                 kernel_size = 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)
    conv10 = Conv2D(n_classes, 1, padding="same")(conv9)
    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    
    return model