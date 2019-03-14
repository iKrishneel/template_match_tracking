#!/usr/bin/env python

import os
import sys
import numpy as np
import logging
import datetime
import multiprocessing

from keras import backend as K
from keras import layers as L
from keras.optimizers import Adam
from keras import callbacks

from network import Network
from dataloader import Dataloader
from data_generator import DataGenerator

def logger_setup(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(screen_handler)
    return logger

logger = logger_setup('logger')

class Trainer(object):
    def __init__(self, dataset_dir, log_dir, weight_file=None):

        self.network = Network()
        # create dataloader
        self.dataloader = Dataloader(dataset_dir, 'list.txt', self.network.get_input_shape)
        # load the auxillary networl model
        self.model = self.network.auxillary_network(input_shape=(7, 7, 512))

        self.initial_epoch = 0
        if weight_file is not None:
            assert os.path.isfile(weight_file), 'Weight file not found: {}'.format(weight_file)
            e = weight_file.split(os.sep)[-1].split('.')[0].split('_')[-1]
            try:
                self.initial_epoch = int(e)
            except ValueError:
                self.initial_epoch = 1
            
            logger.debug('Loading weights from {}'.format(weight_file))
            self.model.load_weights(weight_file)

            self.__log_dir = weight_file.replace(weight_file.split(os.sep)[-1], '')
            logger.debug('Working directory: {}'.format(self.__log_dir))
        else:
            if not os.path.isdir(log_dir):
                os.mkdir(log_dir)

            dir_name = datetime.datetime.now().strftime('%Y%m%d-%H%M')
            log_dir2 = os.path.join(log_dir, dir_name)
            if not os.path.isdir(log_dir2):
                os.mkdir(log_dir2)
            self.__log_dir = log_dir2


        self.__lrate = 0.001
        optimizer = Adam(lr = self.__lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.generator = DataGenerator(self.network, self.dataloader, 1)

        self.MODEL_NAME = "template_match"
        self.__checkpoint_path = os.path.join(self.__log_dir, self.MODEL_NAME + "_*epoch*.h5")
        logger.info(self.model.summary())

    def train(self):
        checkpoint_path = self.__checkpoint_path.replace("*epoch*", "{epoch:04d}")
        cb_handles = [
            callbacks.TensorBoard(log_dir=self.__log_dir, histogram_freq=0, write_graph=True, write_images=False),
            callbacks.ModelCheckpoint(checkpoint_path, verbose=0, save_weights_only=True),
        ]

        logger.debug('Starting at epoch {}. LR={}\n'.format(1, self.__lrate))
        logger.debug('Checkpoint Path: {}'.format(checkpoint_path))
        logger.debug('Number of Epochs: {}\n'.format(1000))

        num_workers = multiprocessing.cpu_count()

        self.model.fit_generator(
            self.generator,
            initial_epoch = self.initial_epoch,
            epochs = 100,
            steps_per_epoch = 1000,
            callbacks = cb_handles,
            max_queue_size = 100,
            workers = 1,
            use_multiprocessing = False,
            verbose = 1,
        )
        
        
        
        
t = Trainer('/home/krishneel/Documents/datasets/vot/vot2014/', 'logs')
t.train()
