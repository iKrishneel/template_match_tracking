#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2 as cv

from network import Network
from dataloader import Dataloader

def to_rect(bbox):
    x1, x2 = np.min(bbox[:, 0]), np.max(bbox[:, 0])
    y1, y2 = np.min(bbox[:, 1]), np.max(bbox[:, 1])                                                                                          
    rect = np.array([x1, y1, x2, y2], dtype=np.int0)
    return rect

def enlarge_bbox(bbox, image_shape, factor):
     x1,y1,x2,y2 = bbox
     pad_x, pad_y = np.array([(x2-x1)/factor, (y2-y1)/factor], dtype=np.int32)
     
     x1 = 0 if x1-pad_x < 0 else x1-pad_x
     y1 = 0 if y1-pad_y < 0 else y1-pad_y
     x2 = image_shape[1] if x2+pad_x > image_shape[1] else x2+pad_x
     y2 = image_shape[0] if y2+pad_y > image_shape[1] else y2+pad_y
     return np.array([x1, y1, x2, y2], np.int0)

def mask_non_object_pixels(bbox, image, enlarge_factor=4):

    x1,y1,x2,y2 = bbox if enlarge_factor is 0 else \
                  enlarge_bbox(bbox, image.shape[:2], enlarge_factor)

    image[:, 0:x1] = 0
    image[0:y1, 0:x1] = 0
    image[0:y1, :] = 0
    image[y2:, x2:] = 0
    image[:, x2:] = 0
    image[y2:, :] = 0

    return image, np.array([x1, y1, x2, y2], dtype=np.int0)

def resize_image_and_bbox(image, bbox, input_shape):
    img = cv.resize(image, input_shape[:2], cv.INTER_CUBIC)
     
    ratio_x = np.float32(image.shape[1]) / np.float32(img.shape[1])
    ratio_y = np.float32(image.shape[0]) / np.float32(img.shape[0])

    bbox = bbox.astype(np.float32)
    if len(bbox.shape) is 2:
        bbox[:, 0] /= ratio_x
        bbox[:, 1] /= ratio_y
    elif len(bbox.shape) is 1:
        bbox[0::2] /= ratio_x
        bbox[1::2] /= ratio_y
    return img, bbox

def test(dataset_dir, model_weights):
    
    network = Network()
    dataloader = Dataloader(dataset_dir, 'test.txt', network.get_input_shape)
    
    model = network.auxillary_network(input_shape=(7, 7, 512))
    model.load_weights(model_weights, by_name=True)

    in_shape = tuple(network.get_input_shape[:2])
    
    index = 0
    dataset = dataloader.get_dataset[index]
    
    im_templ = cv.imread(dataset[index]['im_path'])
    bb_templ = dataset[0]['bbox']

    #! bbox center crop to make it big 
    # bb_templ = enlarge_bbox(to_rect(bb_templ), im_templ.shape, 1.5)
    # x1,y1,x2,y2 = bb_templ
    # im_templ = im_templ[y1:y2, x1:x2].copy()
    
    im_templ, bb_templ = resize_image_and_bbox(im_templ, bb_templ, in_shape)

    bb_templ = to_rect(bb_templ)
    x1,y1,x2,y2 = bb_templ
    
    width = x2 - x1
    height = y2 - y1

    
    #! mask out pixel
    im_templ, bb_templ = mask_non_object_pixels(bb_templ, im_templ, 0)
        
    for i in range(1, 100):
        # input_feats, labels, rois, image = network.build(dataloader, is_test=True, verbose=True)
        
        image = cv.imread(dataset[i]['im_path'], cv.IMREAD_COLOR)
        image = cv.resize(image, in_shape[:2], cv.INTER_CUBIC)
        input_feats, rois = network.base_net_forward(im_templ, bb_templ, image)

        result = model.predict(input_feats)
        
        index = np.argmax(result[:, 0, 0, :])
        
        rois *= network.get_input_shape[0]
        rois = rois.astype(np.int32)
        for i, r in enumerate(result):
            y1, x1, y2, x2 = rois[0, i]
            if r[0, 0, 0] > 0.5:
                image = cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                pass
            else:
                pass
                # image = cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
        print ('Max: {}'.format([index, result[index]]))
        y1, x1, y2, x2 = rois[0, index]

        #! update template
        print ([result[index, 0, 0, :], max(result[index])])

        if max(result[index]) > 0.5:
            # bb_templ = rois[0, index]
            # im_templ, bb_templ = mask_non_object_pixels(bb_templ, image.copy(), 0)

            a,b,c,d = bb_templ
            cv.rectangle(im_templ, (a, b), (c, d), (0, 0, 255), 3)
            cv.imshow('templ', im_templ)
            
            
        image = cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 127 ), 3)
        cv.imshow('result', image)

        cv.waitKey(3)

def main(argv):
    test(argv[1], argv[2])

if __name__ == '__main__':
    main(sys.argv)
