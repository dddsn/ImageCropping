from keras.layers import Input
from keras.models import Model
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten, BatchNormalization
from keras.layers import Conv2D, Dropout
from keras.layers import Dense,Lambda
from keras.applications import vgg16
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU



def get_sxsy_initial_weights(output_size):
    b = np.ones((2, 1), dtype='float32')
    W = np.zeros((output_size, 2), dtype='float32')#2(output_size,6)
    weights = [W, b.flatten()]
    return weights
    
def get_txty_initial_weights(output_size):
    b = np.zeros((2, 1), dtype='float32')#1(2,3)
    W = np.zeros((output_size, 2), dtype='float32')#2(output_size,6)
    weights = [W, b.flatten()]
    return weights
    
def normalization(data):
    zero = tf.convert_to_tensor(0.0,dtype=tf.float32)
    data = tf.map_fn(fn=lambda x:tf.stack([x[0][0],zero,x[1][0],zero,x[0][1],x[1][1]],axis=0),elems=data,dtype=tf.float32)
    
    Max = [1.0,0.0,1.0,0.0,1.0,1.0]
    Min = [0.3,0.0,-1.0,0.0,0.3,-1.0]
    Max = np.array(Max)
    Min = np.array(Min)
    Max = tf.convert_to_tensor(Max,dtype=tf.float32)
    Min = tf.convert_to_tensor(Min,dtype=tf.float32)
    res = tf.map_fn(fn=lambda x:K.minimum(x,Max), elems=data, dtype=tf.float32)
    res = tf.map_fn(fn=lambda x:K.maximum(x,Min), elems=res, dtype=tf.float32)
    Sx = res[:,0]
    Sy = res[:,4]
    tx_max = (1.0-Sx)
    tx_min = (Sx-1.0)

    ty_max = (1.0-Sy)
    ty_min = (Sy-1.0)
    tmax = tf.concat([tf.expand_dims(tx_max,axis=1),tf.expand_dims(ty_max,axis=1)],axis=1)
    tmin = tf.concat([tf.expand_dims(tx_min,axis=1),tf.expand_dims(ty_min,axis=1)],axis=1)
    #print('tmax: '+str(tmax))
    new_Max = tf.map_fn(fn=lambda x:tf.stack([Max[0],Max[1],x[0],Max[3],Max[4],x[1]],axis=0),elems=tmax,dtype=tf.float32)
    new_Min = tf.map_fn(fn=lambda x:tf.stack([Min[0],Min[1],x[0],Min[3],Min[4],x[1]],axis=0),elems=tmin,dtype=tf.float32)
    #print('new_Max: '+str(new_Max))

    #res = tf.map_fn(fn=lambda x:K.minimum(x,new_Max), elems=res, dtype=tf.float32)
    #res = tf.map_fn(fn=lambda x:K.maximum(x,new_Min), elems=res, dtype=tf.float32)
    res = K.minimum(res,new_Max)
    res = K.maximum(res,new_Min)
    #print('res: '+str(res))
    return res

def transform_generator(input_shape=(224, 224, 3), sampling_size=(224, 224), num_classes=10):#(240,320,3)
    image = Input(shape=input_shape)
    # Locnet
    model = vgg16.VGG16(include_top=False, weights=None, input_shape=input_shape)
    #model.load_weights('/lfs1/users/hzhang/project/crop/data/model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model.trainalbel = False
    #model.summary()
    locnet = model(image)
    locnet = Flatten()(locnet)
    locnet = Dense(100, activation='relu', name='fc1')(locnet)
    locnet = Dense(50, activation='relu', name='fc2')(locnet)
    weights = get_sxsy_initial_weights(50)
    sxsy = Dense(2, weights=weights, activation='linear', name='sxsy_parameter')(locnet)
    weights = get_txty_initial_weights(50)
    txty = Dense(2, weights=weights, activation='linear', name='txty_parameter')(locnet)
    locnet_output = Lambda(normalization,name='stn_parameter_2')([sxsy,txty])#2
    
    #combine
    '''x = BilinearInterpolation(sampling_size, name='interpolation')([image, locnet_output])

    x = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv32_1')(x)

    x = MaxPooling2D(pool_size=(2, 2),name='maxpooling')(x)
    x = Flatten()(x)
    x = Dense(1, name='cls', activation='softmax')(x)'''

    return Model(inputs=image, outputs=locnet_output)
    
