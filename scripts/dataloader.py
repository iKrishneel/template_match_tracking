#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
import cv2 as cv
import imgaug as ia
from imgaug import augmenters


class Dataloader(object):

    def __init__(self, dataset_dir, list_fname, input_shape):

        assert len(input_shape) is 3, 'Input shape should be of lenght 3'
        assert os.path.isdir(dataset_dir), 'Dataset directory not found at {}'.format(dataset_dir)

        filename = os.path.join(dataset_dir, list_fname)
        assert os.path.isfile(filename), 'Data list file not found at {}'.format(filename)
        
        self.read_dataset(dataset_dir, filename)
        
        assert len(self.dataset) is not 0, 'No images found'
        
        self.random_index = lambda: random.randint(0, len(self.dataset)-1)

        self.input_shape = input_shape[:2]
        
    def load(self, index=None, use_random=True, use_aug=True):        

        # object index
        index = self.random_index() if index is None else index
        # image index
        idx = random.randint(0, len(self.dataset[index])-1)
        # image
        im_path = self.dataset[index][idx]['im_path']
        image = cv.imread(im_path, cv.IMREAD_COLOR)
        # bounding box
        bbox = self.dataset[index][idx]['bbox']

        # reshape to network input_shape
        image, bbox = self.resize_image_and_labels(image, bbox)        
        image, bbox = self.color_space_argumentation(image, bbox)    
        image = self.normalize(image)

        self.plot(image, bbox)

        return dict(input=image, bbox=bbox)

    def plot(self, image, bbox):
        # x1, x2 = np.min(bbox[:, 0]), np.max(bbox[:, 0])
        # y1, y2 = np.min(bbox[:, 1]), np.max(bbox[:, 1])

        x1, y1, x2, y2 = bbox
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv.imshow('image', image)
        cv.waitKey(0)
    
    def normalize(self, image):
        return image.astype(np.float32) / image.max()

    def resize_image_and_labels(self, image, bbox):
        img = cv.resize(image, self.input_shape[:2], cv.INTER_CUBIC)
        
        ratio_x = np.float32(image.shape[1]) / np.float32(img.shape[1])
        ratio_y = np.float32(image.shape[0]) / np.float32(img.shape[0])

        bbox = bbox.astype(np.float32)
        bbox[:, 0] /= ratio_x
        bbox[:, 1] /= ratio_y
        return img, bbox
        
    def color_space_argumentation(self, image, bbox):

        x1, x2 = np.min(bbox[:, 0]), np.max(bbox[:, 0])
        y1, y2 = np.min(bbox[:, 1]), np.max(bbox[:, 1])
        
        bb = ia.BoundingBoxesOnImage(
            [ia.BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2)], shape=image.shape[:2])
        
        seq = augmenters.Sequential([
            augmenters.OneOf([
                augmenters.GaussianBlur((0, 3.0)),
                augmenters.AverageBlur(k=(2, 7)),
                augmenters.MedianBlur(k=(3, 7)),
            ]),
            augmenters.Fliplr(0.5),
        ], random_order=False)

        seq_det = seq.to_deterministic()

        image = seq_det.augment_image(image)
        bb_aug = seq_det.augment_bounding_boxes([bb])[0]
        bb = bb_aug.bounding_boxes[0]
        bbox = np.array([bb.x1, bb.y1, bb.x2, bb.y2], np.int32)
        return image, bbox
    
    def read_image(self, im_paths):
        images = [
            cv.imread(im_path, cv.IMREAD_COLOR)
            for im_path in im_paths
        ]
        return images
        
    def read_directory(self, dataset_dir):
        folders = [
            os.path.join(dataset_dir, os.path.join(folder, 'face'))
            for folder in os.listdir(dataset_dir)
        ]

        self.dataset = {}        
        for index, folder in enumerate(folders):
            files = [
                os.path.join(folder, ifile)
                for ifile in os.listdir(folder)
                if len(ifile.split('.')) == 2
                if ifile.split('.')[1] in ['jpg', 'png']
            ]
            self.dataset[index] = files

    def read_dataset(self, dataset_dir, filename):

        files = self.read_textfile(dataset_dir, filename)
        
        self.dataset = {}        
        for index, folder in enumerate(files):
            files = [
                os.path.join(folder, ifile)
                for ifile in os.listdir(folder)
                if len(ifile.split('.')) == 2
                if ifile.split('.')[1] in ['jpg',]
            ]

            files = sorted(files)
            
            # read the ground truth
            gt_path = os.path.join(folder, 'groundtruth.txt')

            bboxes = []
            for line in open(gt_path):
                bbox = line.rstrip('\n').split(',')
                bbox = np.array(map(float, bbox), dtype=np.int32).reshape((-1, 2))
                bboxes.append(bbox)

            data = []
            for f, b in zip(files, bboxes):
                data.append({'im_path': f, 'bbox': b})
                
            # assert len(bboxes) == len(files), 'Ground truth does not match {}'.format([len(bboxes), len(files)])            
            # self.dataset[index] = {'images': files, 'bboxes': bboxes}
            
            self.dataset[index] = data
            
    def read_textfile(self, dataset_dir, filename):
        files = [
            os.path.join(dataset_dir, line.rstrip('\n'))
            for line in open(filename)
        ]
        return files
            
                        
    @property
    def get_datasize(self):
        return len(self.dataset)
            
def main(argv):
    d = Dataloader(argv[1], 'list.txt', (224, 224, 3))


    for i in range(10):
        d.load(use_aug=False)
    
    
if __name__ == '__main__':
    main(sys.argv)