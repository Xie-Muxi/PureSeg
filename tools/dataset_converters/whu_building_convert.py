import argparse
import glob
import os.path as osp
import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmengine.fileio import dump
from mmengine.utils import track_iter_progress

# get the arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Convert WHU Building dataset to COCO format')
    parser.add_argument('-p', help='path to the image directory')
    parser.add_argument('-op', default=None, help='path to the output json file')
    parser.add_argument('--image_prefix', default='',help='prefix of image path')
    args = parser.parse_args()

    if args.op is None:
        args.op = args.p + '/annotation_coco.json'

    return args

def convert_whu_to_coco(img_dir, out_file, image_prefix):
    img_files = glob.glob(osp.join(img_dir, 'image/*.tif'))

    # print first 10 images
    print('The first 10 images are:')
    print(img_files[:10])

    print('find {} images in {}'.format(len(img_files), img_dir))
    # img_files = mmcv.utils.scandir(img_dir + '/image')
    annotations = []
    images = []
    obj_count = 0
    
    print('Converting WHU dataset to COCO format')
    # print out the arguments
    print('image directory: {}'.format(img_dir))
    print('output json file: {}'.format(out_file))

    for idx, img_file in enumerate(track_iter_progress(img_files)):
        filename = osp.basename(img_file)
        print(filename)
        img_path = filename
        height, width = mmcv.imread(img_path).shape[:2]
        images.append(
            dict(id=idx, file_name=filename, height=height, width=width))

        segm_file = img_dir + '/label/' + filename
        segm_img = mmcv.imread(segm_file, flag='unchanged', backend='cv2')

        num_labels, instances, stats, centroids = cv2.connectedComponentsWithStats(segm_img, connectivity=4)

        for inst_id in range(1, num_labels):
            mask = np.asarray(instances == inst_id, dtype=np.uint8, order='F')
            if mask.max() < 1:
                print(f'Ignore empty instance: {inst_id} in {segm_file}')
                continue
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

# def convert_whu_to_coco(img_dir, out_file, image_prefix):
#     img_files = glob.glob(osp.join(img_dir, 'image/*.tif'))
#     annotations = []
#     images = []
#     obj_count = 0
#     for idx, img_file in enumerate(track_iter_progress(img_files)):
#         filename = osp.basename(img_file)
#         img_path = osp.join(image_prefix, filename)  # use the full path of the image file
#         height, width = mmcv.imread(img_path).shape[:2]
#         images.append(
#             dict(id=idx, file_name=filename, height=height, width=width))

#         segm_file = img_dir + '/label/' + filename
#         segm_img = mmcv.imread(segm_file, flag='unchanged', backend='cv2')

#         num_labels, instances, stats, centroids = cv2.connectedComponentsWithStats(segm_img, connectivity=4)

#         for inst_id in range(1, num_labels):
#             mask = np.asarray(instances == inst_id, dtype=np.uint8, order='F')
#             if mask.max() < 1:
#                 print(f'Ignore empty instance: {inst_id} in {segm_file}')
#                 continue
#             mask_rle = maskUtils.encode(mask[:, :, None])[0]
#             area = maskUtils.area(mask_rle)
#             bbox = maskUtils.toBbox(mask_rle)
#             mask_rle['counts'] = mask_rle['counts'].decode()

#             data_anno = dict(
#                 image_id=idx,
#                 id=obj_count,
#                 category_id=0,
#                 bbox=bbox.tolist(),
#                 area=area.tolist(),
#                 segmentation=mask_rle,
#                 iscrowd=0)
#             annotations.append(data_anno)
#             obj_count += 1

#     coco_format_json = dict(
#         images=images,
#         annotations=annotations,
#         categories=[{
#             'id': 0,
#             'name': 'building'
#         }])
#     dump(coco_format_json, out_file)


def main():
    args = parse_args()
    data_root = args.p
    train_dir = osp.join(data_root, 'train')
    val_dir = osp.join(data_root, 'val')
    convert_whu_to_coco(train_dir, args.op, args.image_prefix)
    convert_whu_to_coco(val_dir, args.op, args.image_prefix)

if __name__ == '__main__':
    main()
