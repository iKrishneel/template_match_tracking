#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os
import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras import layers as L


class Network(object):
            
    def build(self, input_shape):

        assert len(input_shape) is 3 and isinstance(input_shape, tuple),\
        'input_shape given is: {} but expects tuple '.format(type(input_shape))

        base_net = self.network(input_shape)

        
        input_a = L.Input(shape=input_shape, name='input_a')
        input_b = L.Input(shape=input_shape, name='input_b') 

        encoded_a = base_net(input_a)
        encoded_b = base_net(input_b)

        concat = L.concatenate([encoded_a, encoded_b], axis=3, name='concat')
        conv6 = L.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding = 'same', name='conv6')(concat)
        dp = L.Dropout(rate=0.5)(conv6)
        flatten = L.Flatten(name='flat')(dp)
        classifier = L.Dense(1, activation='sigmoid', name='classifier')(flatten)
        
        return Model(inputs=[input_a, input_b], outputs=classifier)
        
    def network(self, input_shape):

        act = 'relu'
        pad = 'same'
        
        input_data = L.Input(shape=input_shape)
        conv1_1 = L.Conv2D(32, (3, 3), activation=act, strides=(1, 1), padding=pad, name='conv1_1')(input_data)
        conv1_2 = L.Conv2D(32, (3, 3), activation=act, strides=(2, 2), padding=pad, name='conv1_2')(conv1_1)
        # pool1 = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')(conv1_2)
        bn1 = L.BatchNormalization(name='bn1')(conv1_2)
        dp1 = L.Dropout(rate=0.5)(bn1)
        pool1 = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')(dp1)

        conv2_1 = L.Conv2D(64, (3, 3), activation=act, strides=(2, 2), padding=pad, name='conv2_1')(pool1)
        bn2 = L.BatchNormalization(name='bn2')(conv2_1)
        dp2 = L.Dropout(rate=0.5)(bn2)

        conv3_1 = L.Conv2D(128, (3, 3), activation=act, strides=(2, 2), padding = pad, name='conv3_1')(dp2)
        bn3 = L.BatchNormalization(name='bn3')(conv3_1)
        dp3 = L.Dropout(rate=0.5)(bn3)

        conv4_1 = L.Conv2D(256, (3, 3), activation=act, strides=(1, 1), padding = pad, name='conv4_1')(dp3)
        
        conv5_1 = L.Conv2D(64, (1, 1), activation=act, strides=(1, 1), padding = pad, name='conv5_1')(conv4_1)

        return Model(inputs = input_data, outputs = conv5_1)

    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)
    

def main():
    net = Network()
    model = net.build((112,112,3))
    print(model.summary())
    
if __name__ == '__main__':
    main()

