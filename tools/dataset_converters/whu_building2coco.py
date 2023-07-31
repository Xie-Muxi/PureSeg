'''
@File    :   whu_building2coco.py
@Time    :   2023/07/30 23:39:03
@Author  :   Xie-Muxi
@Desc    :   None
'''

import argparse
import cv2
import numpy as np
import os
import os.path as osp
import mmcv
import pycocotools.mask as maskUtils
from mmengine.fileio import dump

def parse_args():
    parser = argparse.ArgumentParser(description='Convert WHU Building dataset to COCO format')
    parser.add_argument('-p', '--path', help='path to the WHU dataset directory')
    parser.add_argument('-o', '--out', default=None, help='path to the output directory')
    args = parser.parse_args()

    if args.out is None:
        args.out = args.path

    return args

def convert_whu_to_coco(img_dir, ann_dir, out_file):
    images = []
    annotations = []
    obj_count = 0
    img_files = sorted(os.listdir(img_dir))
    ann_files = sorted(os.listdir(ann_dir))

    for idx, (img_file, ann_file) in enumerate(zip(img_files, ann_files)):
        img_path = osp.join(img_dir, img_file)
        ann_path = osp.join(ann_dir, ann_file)
        img = mmcv.imread(img_path)
        height, width = img.shape[:2]
        images.append(dict(id=idx, file_name=img_file, height=height, width=width))

        ann_img = cv2.imread(ann_path, cv2.IMREAD_UNCHANGED)
        inst_ids = np.unique(ann_img)
        for i in inst_ids:
            mask = np.asarray(ann_img == i, dtype=np.uint8, order='F')
            mask_rle = maskUtils.encode(mask[:, :, None])[0]
            area = maskUtils.area(mask_rle)
            bbox = maskUtils.toBbox(mask_rle)
            mask_rle['counts'] = mask_rle['counts'].decode()

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=bbox.tolist(),
                area=area.tolist(),
                segmentation=mask_rle,
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'building'
        }])
    dump(coco_format_json, out_file)

def main():
    args = parse_args()
    for split in ['train', 'val']:
        img_dir = osp.join(args.path, split, 'image')
        ann_dir = osp.join(args.path, split, 'label')
        out_file = osp.join(args.out, f'{split}_annotation_coco.json')
        convert_whu_to_coco(img_dir, ann_dir, out_file)

if __name__ == '__main__':
    main()
