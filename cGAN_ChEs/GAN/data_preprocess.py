import scipy
import scipy.misc
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import  random
from scipy.io import loadmat
import h5py
import time
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers


def load_image_train(path, batch_size = 1):
    """load, jitter, and normalize"""
    with h5py.File(path, 'r') as file:
        real_image = np.transpose(np.array(file['output_da']))

        
    with h5py.File(path, 'r') as file:
        input_image = np.transpose(np.array(file['input_da']))
        
    # real_image = real_image[1:10,:,:,:]
    # input_image = input_image[1:10,:,:,:]
        
    SIZE_IN= real_image.shape
    list_im=list(range(0, SIZE_IN[0]))

    batch_im = random.sample(list_im,SIZE_IN[0])
    real_image = real_image[batch_im,:,:,:]
    input_image = input_image[batch_im,:,:,:]
    
    n_batches = int(SIZE_IN[0] / batch_size)
    
    for i in range(n_batches-1):
        imgs_A = real_image[i*batch_size:(i+1)*batch_size]
        imgs_B = input_image[i*batch_size:(i+1)*batch_size]
        
    
        yield imgs_A, imgs_B
        



def load_image_test(path, batch_size = 1):
       
    with h5py.File(path, 'r') as file:
        real_image = np.transpose(np.array(file['output_da_test']))

        
    with h5py.File(path, 'r') as file:
        input_image = np.transpose(np.array(file['input_da_test']))
        
    SIZE_IN= real_image.shape

    
    n_batches = int(SIZE_IN[0] / batch_size)
    
    for i in range(n_batches-1):
        imgs_A = real_image[i*batch_size:(i+1)*batch_size]
        imgs_B = input_image[i*batch_size:(i+1)*batch_size]
        
    
        yield imgs_A, imgs_B
        
def load_image_test_y(path):
       
    with h5py.File(path, 'r') as file:
        real_image = np.transpose(np.array(file['output_da_test']))

        
    with h5py.File(path, 'r') as file:
        input_image = np.transpose(np.array(file['input_da_test']))
        
        
    
    return real_image, input_image
