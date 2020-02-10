import os
import sys
import argparse
import _pickle as pickle
import json

import cv2
import numpy as np
from sklearn.cluster import KMeans

def build_arg_parser():
    parser = argparse.ArgumentPaeser(description='Creates features for given images')
    parser.add_argument("--samples", dest="cls", nargs="+", action="append", required=True,
                        help="Folders containing images."
                             "\The first element needs to be the class label.")
    parser.add_argument ("--codebook-file", dest='codebook file', required=True,
                         help="Base file name to store the codebook")
    parser.add_argument("--feature-map-file", dest='feature_map_file', required=True,
                        help="Base file name to store the feature map")
    return parser

#Loading the images from the input folder
def load_input_map(label, input_folder):
    combined_data = []

    if not os.path.isdir(input_folder):
        raise IOError("The folder " + input_folder + "doesn't exist")

    # Parse the input folder and assign the labels
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if x.endswith('.jpg')):
            combined_data.append({'label': label, 'image': os.path.join(root, filename)})
        return combined_data

class FeatureExtractor (object):
    def extract_image_features(self, img):
        #Dense feature detector
        kps = cv2.DenseDetector().detect(img)
        # SIFT feature extractor
        kps, fvs = cv2.SIFTExtractor().compute(img, kps)
        return fvs

    #Extract the centroids from the feature points
    def get_centroids(self, input_map, num_samples_to_fit=10):
        kps_all = []