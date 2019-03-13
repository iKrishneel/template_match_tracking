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
    
    def build(self, input_shape):

        assert len(input_shape) is 3 and isinstance(input_shape, tuple),\
        'input_shape given is: {} but expects tuple '.format(type(input_shape))

        # load the mask rcnn model
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models')
        model_path = os.path.join(model_dir, 'mask_rcnn_coco.h5')
        mrcnn_model, rpn_model, input_shape = mask_rcnn_rpn(model_path)

        # dont train this model
        mrcnn_model.trainable = False

        loader = Dataloader('/home/krishneel/Documents/datasets/vot/vot2014/', 'list.txt', input_shape)
        datum = loader.load()
        
        im_templ = datum['templ']
        bb_templ = datum['templ_bbox']
        image = datum['image']
        bbox = datum['bbox']

        templ_rois, templ_feats = forward(mrcnn_model, rpn_model, im_templ)

        
        print ('Shape: {}'.format([im_templ.shape]))
        jc = JaccardCoeff()
        x1,y1,x2,y2 = bb_templ
        r1 = np.array([x1, y1, x2-x1, y2-y1], dtype=np.int0)
        
        max_rect, max_iou, max_index = None, 0, -1
        for i, tr in enumerate(templ_rois[0, :]):
            y1,x1,y2,x2 = tr * input_shape[0]
            r2 = np.array([x1, y1, x2-x1, y2-y1], np.int0)

            iou_score = jc.iou(r1, r2)
            if iou_score > max_iou:
                max_iou = iou_score
                max_rect = np.array([x1, y1, x2, y2], np.int0)
                max_index = i

        x1,y1,x2,y2 = max_rect
        im_templ = cv.rectangle(im_templ, (x1, y1), (x2, y2), (255, 0, 0), 3)
        print ([templ_rois.shape, templ_feats.shape, max_iou, max_index])
        print ([input_shape, max_rect, bb_templ])
        
        cv.imshow('tmp', im_templ)

        # compute iou scores
        x1, y1, x2, y2 = bb_templ/input_shape[0]
        gt_box = np.array([y1, x1, y2, x2], np.float32)
        gt_box = np.expand_dims(gt_box, 0)
        gt_box = tf.convert_to_tensor(gt_box)
        templ_rois = tf.convert_to_tensor(templ_rois[0])

        max_iou, max_index = self.overlap(templ_rois, gt_box, ret_max=True)
        
        sess = tf.Session()
        print ([sess.run(max_iou), sess.run(max_index)])
        
        cv.waitKey(0)
        return
        

    
    def overlap(self, boxes1, boxes2, ret_max=False):
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
            return tf.argmax(overlaps), tf.reduce_max(overlaps)
            
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
    model = net.build((112,112,3))
    # print(model.summary())

    # mask_rcnn_rpn(argv[1])
    
if __name__ == '__main__':
    main(sys.argv)

