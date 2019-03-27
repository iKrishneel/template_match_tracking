#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2 as cv
import random

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras import layers as L

from dataloader import Dataloader
from iou import JaccardCoeff

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mask_rcnn')
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

class Network(object):

    def __init__(self, model_path=None, log_dir=None):
        self.__iou_thresh = 0.04
        self.__session = K.get_session()

        # load the mask rcnn model
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models')
        model_path = os.path.join(model_dir, 'mask_rcnn_coco.h5')
        self.__mrcnn_model, self.__rpn_model, self.__input_shape = self.mask_rcnn_rpn(model_path, log_dir)

        self.__rpn_model._make_predict_function()

        # dont train this model
        self.__mrcnn_model.trainable = False

        self.__counter = 0

    def build(self, dataloader, is_test=False, verbose=False):
        
        while True:
            datum = dataloader.load(verbose=verbose, use_random=False)
        
            im_templ = datum['templ']
            bb_templ = datum['templ_bbox']
            image = datum['image']
            bbox = datum['bbox']

            # propagate through the network to obtain rois and features of the rois
            templ_rois, templ_feats = self.forward(im_templ)

            max_iou, max_index = self.get_overlapping_rois2(bb_templ, templ_rois, True)
            if max_iou >= self.__iou_thresh:
                self.__counter = 0
                break
            self.__counter += 1
            if self.__counter == 100:
                print ('SEEMS NOTHING IS FOUND...')
                sys.exit()

        # propagate the src image
        src_rois, src_feats = self.forward(image)

        top_k = None # random.randint(20, src_rois.shape[1])
        # generate labels
        labels, indices = self.label2(bbox, src_rois, top_k)

        # select the features
        if top_k is not None and indices is not None:
            src_feats = src_feats[0, :][indices]
            src_feats = np.expand_dims(src_feats, 0)
        
        # tile the feature of the template roi
        templ_feat = templ_feats[:, max_index]
        templ_feat = np.expand_dims(templ_feat, 0)
        templ_feats = np.tile(templ_feat, (1, src_feats.shape[1], 1, 1, 1))
        
        # concate the features
        inputs_concat = np.concatenate((templ_feats[0], src_feats[0]), axis=-1)
        
        if verbose:
            print('RPN SIZE: {}'.format(templ_rois.shape))
            y1,x1,y2,x2 = templ_rois[0][max_index] * self.__input_shape[0]
            im_templ = cv.rectangle(im_templ, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv.imshow('tmp', im_templ)
            cv.waitKey(3)

        # inputs_concat, labels = self.__session.run([inputs_concat, labels])
        if is_test:
            return inputs_concat, labels, src_rois, image
        return inputs_concat, labels


    def base_net_forward(self, im_templ, bb_templ, image):

        # propagate through the network to obtain rois and features of the rois
        templ_rois, templ_feats = self.forward(im_templ)
        max_iou, max_index = self.get_overlapping_rois2(bb_templ, templ_rois, True)

        # propagate the src image
        src_rois, src_feats = self.forward(image)

        top_k = None # random.randint(20, src_rois.shape[1])
        # generate labels
        # labels, indices = self.label2(bbox, src_rois, top_k)

        # select the features
        # if top_k is not None and indices is not None:
        #    src_feats = src_feats[0, :][indices]
        #    src_feats = np.expand_dims(src_feats, 0)
        
        # tile the feature of the template roi
        templ_feat = templ_feats[:, max_index]
        templ_feat = np.expand_dims(templ_feat, 0)
        templ_feats = np.tile(templ_feat, (1, src_feats.shape[1], 1, 1, 1))
        
        # concate the features
        inputs_concat = np.concatenate((templ_feats[0], src_feats[0]), axis=-1)
        return inputs_concat, src_rois

    
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

    def label2(self, gt_box, src_rois, k=None):
        """ function to generate the labels """
        overlaps = self.get_overlapping_rois2(gt_box, src_rois, ret_max=False)

        indices = None
        if k is None:
            gt_labels = np.greater(overlaps, self.__iou_thresh)
        else:
            indices = np.argsort(overlaps[:, 0])[-k:]
            gt_labels = overlaps[:, 0][indices]
            gt_labels = np.greater(gt_labels, self.__iou_thresh)
            gt_labels = np.expand_dims(gt_labels, 1)
            
        gt_labels = gt_labels.astype(np.int0)
        # gt_labels = np.tile(gt_labels, (1, 2))
        # gt_labels[:, 0] = 0
        gt_labels = np.expand_dims(np.expand_dims(gt_labels, 1), 1)
        return gt_labels, indices
        
    def get_overlapping_rois2(self, bbox, rpn_rois, ret_max=True):
        x1, y1, x2, y2 = bbox/self.__input_shape[0]
        gt_box = np.array([y1, x1, y2, x2], np.float32)
        gt_box = np.expand_dims(gt_box, 0)
        if ret_max:
            return self.overlap2(gt_box, rpn_rois, ret_max)
        return self.overlap2(gt_box, rpn_rois, ret_max)

    def overlap2(self, boxes1, boxes2, ret_max):
        b1 = np.tile(boxes1, (boxes2.shape[0], boxes2.shape[1], 1))
        b2 = boxes2
        # 2. Compute import code; code.interact(local=locals())ersections
        b1_y1, b1_x1, b1_y2, b1_x2 = np.split(b1, 4, axis=2)
        b2_y1, b2_x1, b2_y2, b2_x2 = np.split(b2, 4, axis=2)
        y1 = np.maximum(b1_y1, b2_y1)
        x1 = np.maximum(b1_x1, b2_x1)
        y2 = np.minimum(b1_y2, b2_y2)
        x2 = np.minimum(b1_x2, b2_x2)
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        # 3. Compute unions
        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        union = b1_area + b2_area - intersection
        # 4. Compute IoU and reshape to [boxes1, boxes2]
        iou = intersection / union
        overlaps = iou[0] # np.reshape(iou, [np.shape(boxes1)[0], np.shape(boxes2)[0]])
        if ret_max:
            return np.max(overlaps), np.argmax(overlaps)
        return overlaps

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

    @property
    def get_input_shape(self):
        return self.__input_shape

    def forward(self, image):
        molded_image, image_meta, window = self.__mrcnn_model.mold_inputs([image])
        anchors = self.__mrcnn_model.get_anchors(molded_image[0].shape)
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)    
        rois_output, rpn_output = self.__rpn_model.predict([molded_image, image_meta, anchors], verbose=0)
        return rois_output, rpn_output
    
    def mask_rcnn_rpn(self, model_path, log_dir=None):
    
        assert os.path.isfile(model_path), 'Pretrained MASK_RCNN weights not found at: {}'.format(model_path)
    
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
            # POST_NMS_ROIS_INFERENCE = 256

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

    for i in range(100):
        model = net.build(loader, verbose=True)
    # print(model.summary())

    # mask_rcnn_rpn(argv[1])
    
if __name__ == '__main__':
    main(sys.argv)

