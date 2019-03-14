#!/usr/bin/env python

import numpy as np
from keras.utils import Sequence

class DataGenerator(Sequence):

    def __init__(self, network, data_loader, batch_size, n_classes = 1, shuffle = True):

        self.__network = network
        self.__data_loader = data_loader
        self.__data_size = 1000
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__num_class = n_classes

        self.COUNT = 0
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.__data_size / self.__batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        try:
            return self.__data_generation()
        except Exception as e:
            raise ValueError(e)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.__indexes = np.arange(self.__data_size)
        if self.__shuffle == True:
            np.random.shuffle(self.__indexes)

    def __data_generation(self, sample_ids = None):
        'Generates data containing batch_size samples'
        input_datum, labels = self.__network.build(self.__data_loader, False) 
        return ({'input': input_datum}, labels)

