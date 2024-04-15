# -*- coding: utf-8 -*-
import json
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Split COCO dataset into training and validation sets.')
    parser.add_argument('--data-path', default='./data/nwpu/NWPU/NWPU VHR-10 dataset', help='The base path of the dataset.')
    parser.add_argument('--out-path', default='./data_dataset_coco/', help='The output path of the split dataset.')
    parser.add_argument('--split-ratio', type=float, default=0.7, help='The ratio of training set split.')
    return parser.parse_args()

def load_annotations(path):
    with open(path, encoding='utf-8') as file:
        return json.load(file)

def save_annotations(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)

def split_dataset(gt, split_ratio):
    train = {
        'info': gt['info'],
        'licenses': gt['licenses'],
        'categories': gt['categories'],
        'images': [],
        'annotations': []
    }
    
    val = {
        'info': gt['info'],
        'licenses': gt['licenses'],
        'categories': gt['categories'],
        'images': [],
        'annotations': []
    }
    
    train_image_size = int(len(gt['images']) * split_ratio)
    print(f'train_img_num: {train_image_size}')
    print(f'val_img_num: {len(gt['images']) - train_image_size}')
    
    random.shuffle(gt['images'])
    
    image_id_to_annotations = {anno['image_id']: [] for anno in gt['annotations']}
    for anno in gt['annotations']:
        image_id_to_annotations[anno['image_id']].append(anno)

    for img_info in gt['images']:
        if len(train['images']) < train_image_size:
            train['images'].append(img_info)
            train['annotations'].extend(image_id_to_annotations[img_info['id']])
        else:
            val['images'].append(img_info)
            val['annotations'].extend(image_id_to_annotations[img_info['id']])

    return train, val

def main():
    args = parse_args()
    annotations_path = f"{args.dataset_path}annotations.json"
    gt = load_annotations(annotations_path)
    
    train, val = split_dataset(gt, args.split_ratio)
    
    train_path = f"{args.out_path}nwpu-instances_train.json"
    val_path = f"{args.out_path}nwpu-instances_val.json"
    
    save_annotations(train, train_path)
    save_annotations(val, val_path)
    
    print('Dataset successfully split and saved.')

if __name__ == '__main__':
    main()
