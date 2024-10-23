# Deploy S2-UNET and Output Predicted PV ranges

# built-in
import pickle, copy, logging, os, sys

# packages
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image #2
import numpy as np

# ML
import tensorflow as tf
from tensorflow.python import keras
from keras.layers import Input, Dense, Activation, Cropping2D, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, UpSampling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model, to_categorical, Sequence
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from random import shuffle
from keras.callbacks import CSVLogger, Callback, ModelCheckpoint
import keras.backend as K

batch_size=8
outpath=os.path.join(os.getcwd(),'data','S2_unet')

class DataGenerator(Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, augment=False):
        """ Initialisation """
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augment=augment



    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate indexes of the batch"""
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples of shape:
        (n_samples, *dim, n_channels)
        """

        # Initialization
        X = np.zeros((self.batch_size, *self.dim))
        y = np.zeros((self.batch_size, self.dim[0], self.dim[1]))#, dtype=int)


        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample
            if self.augment:
                k_rot=np.random.choice([0,1,2,3])
                k_flip = np.random.choice([0,1])
                
                X[i,:,:,:] = np.flip(np.rot90(np.load(ID['data'])['data'], k_rot, axes=(0,1)),k_flip)
                y[i,:,:] = np.flip(np.rot90(np.load(ID['data'])['annotation'].astype('int')/255, k_rot, axes=(0,1)),k_flip)

            else:
            
                X[i,:,:,:] = np.load(ID['data'])['data']

                # Store class
                y[i,:,:] = np.load(ID['data'])['annotation'].astype('int')/255
                #print (y)


        return X, to_categorical(y, num_classes=self.n_classes)


class TestS2Unet:

    def __init__(self, data_dir, outp_fname, tst_records_pickle):
        # i/o
        self.data_dir = data_dir
        self.outp_fname = outp_fname

        self.BATCH_SIZE = 2
        self.N_CLASSES = 2
        self.EPOCHS = 40
        self.LEARNING_RATE = 1e-7
        self.INPUT_SHAPE = (200,200,14)

        self.tst_records = pickle.load(open(tst_records_pickle,'rb'))

        

    def test(self):
        model=tf.keras.models.load_model(self.outp_fname)
        tst_generator = DataGenerator(self.tst_records, 
                                batch_size=self.BATCH_SIZE, 
                                dim=self.INPUT_SHAPE, 
                                n_channels=1,
                                n_classes=self.N_CLASSES, 
                                shuffle=True,
                                augment=True)
        pv_classes=model.predict_generator(generator=tst_generator, steps=len(tst_generator)/batch_size)
        np.savez(os.path.join(outpath,'test_outp'+'.npz'), outp = pv_classes)
        print('Testing output saved:', pv_classes.shape())


if __name__ == "__main__":
    tst= TestS2Unet(
        data_dir=os.path.join(os.getcwd(),'data','S2_unet_test'),#not confirmed
        outp_fname='s2_unet.h5',
        tst_records_pickle=os.path.join(os.getcwd(),'data','S2_unet','records.pickle'))
    tst.test()


