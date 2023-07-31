'''
@File    :   nwpu2coco.py
@Time    :   2023/07/31 01:39:32
@Author  :   Xie-Muxi
@Desc    :   None
'''

import argparse
import os.path as osp

import mmcv

from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress

def parse_args():
    parser = argparse.ArgumentParser(description='Convert balloon dataset to COCO format')
    parser.add_argument('-p', '--path', help='path to the balloon dataset directory')
    parser.add_argument('-o', '--out', default=None, help='path to the output directory')
    args = parser.parse_args()

    if args.out is None:
        args.out = args.path

    return args

#TODO: convert_xxx_to_coco, you can implement your own convert function
def convert_nwpu_to_coco(ann_file, out_file, image_prefix):
    pass


if __name__ == '__main__':
    args = parse_args()
    #ann_file, out_file, image_prefix
    for split in ['train', 'val']:
        convert_nwpu_to_coco(ann_file, out_file, image_prefix) #TODO: 

