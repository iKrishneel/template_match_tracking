#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2 as cv


import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras import layers as L

from dataloader import Dataloader
from iou import JaccardCoeff

class Network(object):

    def __init__(self):
        self.__iou_thresh = 0.60
        self.__session = K.get_session()

        # load the mask rcnn model
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models')
        model_path = os.path.join(model_dir, 'mask_rcnn_coco.h5')
        self.__mrcnn_model, self.__rpn_model, self.__input_shape = mask_rcnn_rpn(model_path)

        # dont train this model
        self.__mrcnn_model.trainable = False

    def build(self, dataloader, verbose=False):
        datum = dataloader.load()
        
        im_templ = datum['templ']
        bb_templ = datum['templ_bbox']
        image = datum['image']
        bbox = datum['bbox']

        # propagate through the network to obtain rois and features of the rois
        templ_rois, templ_feats = forward(self.__mrcnn_model, self.__rpn_model, im_templ)
        src_rois, src_feats = forward(self.__mrcnn_model, self.__rpn_model, image)

        # compute the feature of the roi that covers the template region
        max_iou, max_index = self.get_overlapping_rois(bb_templ, templ_rois, True)

        if max_iou < self.__iou_thresh:
            # todo: select another image
            return

        # tile the feature of the template roi
        max_index = max_index[0]
        templ_feat = templ_feats[:, max_index]
        templ_feat = tf.expand_dims(templ_feat, 0)
        templ_feats = tf.tile(templ_feat, (1,src_feats.shape[1], 1, 1, 1))

        # concate the features
        src_feats = tf.convert_to_tensor(src_feats)
        inputs_concat = L.concatenate([templ_feats[0], src_feats[0]], axis=-1, name='concate')

        # generate labels
        labels = self.label(bbox, src_rois)

        if verbose:
            y1,x1,y2,x2 = templ_rois[0][max_index] * self.__input_shape[0]
            im_templ = cv.rectangle(im_templ, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv.imshow('tmp', im_templ)
            cv.waitKey(0)

        print ([inputs_concat.shape, labels.shape])
        return inputs_concat, labels


    def auxillary_network(self, input_shape):
        """ create auxillary trainable network at the head """

        act = 'relu'
        pad = 'same'
        input_data = L.Input(shape=input_shape, name='input')
        x = L.Conv2D(512, (1, 1), activation=act, strides=(1, 1), padding=pad)(input_data)
        x = L.BatchNormalization()(x)
        x = L.Dropout(rate=0.5)(x)
        x = L.Conv2D(512, (3, 3), activation=act, strides=(3, 3), padding=pad)(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(rate=0.5)(x)
        x = L.Conv2D(1, (3, 3), activation='sigmoid', strides=(3, 3), padding=pad, name='classifier')(x)
        
        return Model(inputs = input_data, outputs = x)

    def label(self, gt_box, src_rois):
        """ function to generate the labels """
        overlaps = self.get_overlapping_rois(gt_box, src_rois, ret_max=False)
        gt_labels = tf.greater(overlaps, self.__iou_thresh)

        gt_labels = self.__session.run(gt_labels)
        gt_labels = gt_labels.astype(np.int0)
        gt_labels = tf.expand_dims(tf.expand_dims(gt_labels, 1), 1)
        return gt_labels
        
    
    def get_overlapping_rois(self, bbox, rpn_rois, ret_max=True):
        x1, y1, x2, y2 = bbox/self.__input_shape[0]
        gt_box = np.array([y1, x1, y2, x2], np.float32)
        gt_box = np.expand_dims(gt_box, 0)
        gt_box = tf.convert_to_tensor(gt_box)
        rois_tf = tf.convert_to_tensor(rpn_rois[0])

        if ret_max:
            max_iou, max_index = self.overlap(rois_tf, gt_box, ret_max)
            return self.__session.run([max_iou, max_index])
        return self.overlap(rois_tf, gt_box, ret_max)
    
    
    def overlap(self, boxes1, boxes2, ret_max):
        b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                                [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
        b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
        # 2. Compute import code; code.interact(local=locals())ersections
        b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
        b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
        y1 = tf.maximum(b1_y1, b2_y1)
        x1 = tf.maximum(b1_x1, b2_x1)
        y2 = tf.minimum(b1_y2, b2_y2)
        x2 = tf.minimum(b1_x2, b2_x2)
        intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
        # 3. Compute unions
        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        union = b1_area + b2_area - intersection
        # 4. Compute IoU and reshape to [boxes1, boxes2]
        iou = intersection / union
        overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
        
        if ret_max:
            return tf.reduce_max(overlaps), tf.argmax(overlaps)
            
        return overlaps

        
    
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

    @property
    def get_input_shape(self):
        return self.__input_shape


def forward(mrcnn_model, rpn_model, image):
    molded_image, image_meta, window = mrcnn_model.mold_inputs([image])
    anchors = mrcnn_model.get_anchors(molded_image[0].shape)
    anchors = np.broadcast_to(anchors, (1,) + anchors.shape)    
    rois_output, rpn_output = rpn_model.predict([molded_image, image_meta, anchors], verbose=0)
    return rois_output, rpn_output
    
def mask_rcnn_rpn(model_path, log_dir=None):
    
    assert os.path.isfile(model_path), 'Pretrained MASK_RCNN weights not found at: {}'.format(model_path)
    
    ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mask_rcnn')
    sys.path.append(ROOT_DIR)
    from mrcnn.config import Config
    from mrcnn import model as modellib, utils
    
    log_dir = os.path.join(os.environ['HOME'], '.ros/logs') if log_dir is None else log_dir
    
    print('Initializing the Mask RCNN model')
    
    class InferenceConfig(Config):
        # Give the configuration a recognizable name
        NAME = "handheld_objects"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0
        NUM_CLASSES = 81
        IMAGE_MIN_DIM = 224
        IMAGE_MAX_DIM = 448

    config = InferenceConfig()
    config.display()
    
    mrcnn_model = modellib.MaskRCNN(mode='inference', config=config, model_dir=log_dir)
    mrcnn_model.load_weights(model_path, by_name=True)

    layer_name=['ROI', 'roi_align_classifier']
    model = mrcnn_model.keras_model
    rpn_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name[0]).output, \
                                                   model.get_layer(layer_name[1]).output])
    print('Model is successfully initialized!')
    return mrcnn_model, rpn_model, (config.IMAGE_SHAPE)
        

def main(argv):
    net = Network()

    loader = Dataloader('/home/krishneel/Documents/datasets/vot/vot2014/', 'list.txt', net.get_input_shape)
    model = net.build(loader)
    # print(model.summary())

    # mask_rcnn_rpn(argv[1])
    
if __name__ == '__main__':
    main(sys.argv)

