#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import sys
import json

from detectron.utils.io import load_object
from detectron.utils.io import save_object

NUM_PUBLAYNET_CLS = 6

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a PubLayNet pre-trained model for fine-tuning on another target dataset')
    parser.add_argument(
        '--PubLayNet_model', dest='PubLayNet_model_file_name',
        help='Pretrained network weights file path',
        default=None, type=str)
    parser.add_argument(
        '--lookup_table', dest='lookup_table',
        help='Blob conversion lookup table',
        type=json.loads)
    parser.add_argument(
        '--output', dest='out_file_name',
        help='Output file path',
        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    args.NUM_TARGET_CLS = len(args.lookup_table)
    return args


def convert_PubLayNet_blobs_to_target_blobs(model_dict):
    for k, v in model_dict['blobs'].items():
        if hasattr(v, 'shape'):
            if v.shape:
                if v.shape[0] == NUM_PUBLAYNET_CLS or v.shape[0] == 4 * NUM_PUBLAYNET_CLS:
                    PubLayNet_blob = model_dict['blobs'][k]
                    print(
                        'Converting PUBLAYNET blob {} with shape {}'.
                        format(k, PubLayNet_blob.shape)
                    )
                    target_blob = convert_PubLayNet_blob_to_target_blob(
                        PubLayNet_blob, args.lookup_table
                    )
                    print(' -> converted shape {}'.format(target_blob.shape))
                    model_dict['blobs'][k] = target_blob


def convert_PubLayNet_blob_to_target_blob(PubLayNet_blob, lookup_table):
    # PubLayNet blob (6, ...) or (6*4, ...)
    PubLayNet_shape = PubLayNet_blob.shape
    leading_factor = int(PubLayNet_shape[0] / NUM_PUBLAYNET_CLS)
    tail_shape = list(PubLayNet_shape[1:])
    assert leading_factor == 1 or leading_factor == 4

    # Reshape in [num_classes, ...] form for easier manipulations
    PubLayNet_blob = PubLayNet_blob.reshape([NUM_PUBLAYNET_CLS, -1] + tail_shape)
    # Default initialization uses Gaussian with mean and std to match the
    # existing parameters
    std = PubLayNet_blob.std()
    mean = PubLayNet_blob.mean()
    target_shape = [args.NUM_TARGET_CLS] + list(PubLayNet_blob.shape[1:])
    target_blob = (np.random.randn(*target_shape) * std + mean).astype(np.float32)

    # Replace random parameters with PUBLAYNET parameters if class mapping exists
    for i in range(args.NUM_TARGET_CLS):
        PubLayNet_cls_id = lookup_table[i]
        if PubLayNet_cls_id >= 0:  # otherwise ignore (rand init)
            target_blob[i] = PubLayNet_blob[PubLayNet_cls_id]

    target_shape = [args.NUM_TARGET_CLS * leading_factor] + tail_shape
    return target_blob.reshape(target_shape)


def remove_momentum(model_dict):
    for k in list(model_dict['blobs'].keys()):
        if k.endswith('_momentum'):
            del model_dict['blobs'][k]


def load_and_convert_PubLayNet_model(args):
    model_dict = load_object(args.PubLayNet_model_file_name)
    remove_momentum(model_dict)
    convert_PubLayNet_blobs_to_target_blobs(model_dict)
    return model_dict


if __name__ == '__main__':
    args = parse_args()
    print(args)
    assert os.path.exists(args.PubLayNet_model_file_name), \
        'Weights file does not exist'
    weights = load_and_convert_PubLayNet_model(args)

    save_object(weights, args.out_file_name)
    print('Wrote blobs to {}:'.format(args.out_file_name))
    print(sorted(weights['blobs'].keys()))
