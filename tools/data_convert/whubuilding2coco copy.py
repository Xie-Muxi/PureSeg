import argparse
import glob
import os
import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmengine.fileio import dump
from mmengine.utils import (
    Timer,
    mkdir_or_exist,
    track_parallel_progress,
    track_progress,
)


def collect_files(img_dir, gt_dir):
    img_files = glob.glob(os.path.join(img_dir, "*.tif"))
    files = [
        (img_file, os.path.join(gt_dir, os.path.basename(img_file)))
        for img_file in img_files
    ]
    assert files, f"No images found in {img_dir}"
    print(f"Loaded {len(files)} images from {img_dir}")
    return files


def load_img_info(file_pair):
    img_file, segm_file = file_pair
    segm_img = mmcv.imread(segm_file, flag="unchanged", backend="cv2")
    num_labels, instances, stats, centroids = cv2.connectedComponentsWithStats(
        segm_img, connectivity=4
    )
    annotations = []
    for inst_id in range(1, num_labels):
        mask = (instances == inst_id).astype(np.uint8, order="F")
        if mask.max() < 1:
            print(f"Ignore empty instance: {inst_id} in {segm_file}")
            continue
        mask_rle = maskUtils.encode(np.asfortranarray(mask[:, :, None]))[0]
        mask_rle["counts"] = mask_rle["counts"].decode()
        annotations.append(
            {
                "iscrowd": 0,
                "category_id": 1,
                "bbox": maskUtils.toBbox(mask_rle).tolist(),
                "area": maskUtils.area(mask_rle).tolist(),
                "segmentation": mask_rle,
            }
        )
    return {
        "file_name": os.path.basename(img_file),
        "height": segm_img.shape[0],
        "width": segm_img.shape[1],
        "annotations": annotations,
        "segm_file": os.path.basename(segm_file),
    }


def collect_annotations(files, nproc=1):
    if nproc > 1:
        return track_parallel_progress(load_img_info, files, nproc=nproc)
    return track_progress(load_img_info, files)


def convert_annotations(image_infos, output_path):
    output_data = {
        "images": [],
        "categories": [{"id": 1, "name": "building"}],
        "annotations": [],
    }
    ann_id = 0
    for img_id, info in enumerate(image_infos):
        info["id"] = img_id
        output_data["images"].append(info)
        for anno in info.pop("annotations"):
            anno.update({"image_id": img_id, "id": ann_id})
            output_data["annotations"].append(anno)
            ann_id += 1
    dump(output_data, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert annotations to COCO format")
    parser.add_argument("--data-path", default="data", help="base data path")
    parser.add_argument("--img-dir", default="imgs", type=str, help="image directory")
    parser.add_argument(
        "--gt-dir", default="imgs", type=str, help="ground truth directory"
    )
    parser.add_argument(
        "-o", "--out-dir", default="annotations", help="output directory"
    )
    parser.add_argument("--nproc", default=0, type=int, help="number of processes")
    return parser.parse_args()


def main():
    args = parse_args()
    mkdir_or_exist(args.out_dir)
    for split in ["train", "val", "test"]:
        print(f"Processing {split} data...")
        files = collect_files(
            os.path.join(args.data_path, args.img_dir, split),
            os.path.join(args.data_path, args.gt_dir, split),
        )
        image_infos = collect_annotations(files, nproc=args.nproc)
        convert_annotations(
            image_infos, os.path.join(args.out_dir, f"WHU_building_{split}.json")
        )


if __name__ == "__main__":
    main()
