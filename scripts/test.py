#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2 as cv

from network import Network
from dataloader import Dataloader

def test(dataset_dir, model_weights):
    
    network = Network()
    dataloader = Dataloader(dataset_dir, 'test.txt', network.get_input_shape)

    model = network.auxillary_network(input_shape=(7, 7, 512))
    model.load_weights(model_weights, by_name=True)


    for _ in range(20):
        input_feats, labels, rois, image = network.build(dataloader, is_test=True, verbose=True)
        result = model.predict(input_feats)
        
        index = np.argmax(result[:, 0, 0, :])
        
        rois *= network.get_input_shape[0]
        rois = rois.astype(np.int32)
        for i, r in enumerate(result):
            y1, x1, y2, x2 = rois[0, i]
            if r[0, 0, 0] > 0.5:
                image = cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            else:
                pass
                # image = cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
        print ('Max: {}'.format([index, result[index]]))
        y1, x1, y2, x2 = rois[0, index]
        image = cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 127 ), 3)
        
        cv.imshow('result', image)
        cv.waitKey(0)

def main(argv):
    test(argv[1], argv[2])

if __name__ == '__main__':
    main(sys.argv)
