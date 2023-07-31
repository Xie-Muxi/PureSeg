'''
@File    :   tem.py
@Time    :   2023/07/31 01:37:51
@Author  :   Xie-Muxi
@Desc    :   converting xxx dataset to coco format
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
def convert_xxx_to_coco(ann_file, out_file, image_prefix):
    data_infos = load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(
            dict(id=idx, file_name=filename, height=height, width=width))

        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'balloon'
        }])
    dump(coco_format_json, out_file)


if __name__ == '__main__':
    args = parse_args()
    #ann_file, out_file, image_prefix
    for split in ['train', 'val']:
        ...
        # convert_xxx_to_coco(ann_file, out_file, image_prefix) #TODO: change the function name

