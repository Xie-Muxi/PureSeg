def convert_whu_to_coco(img_dir, out_file, image_prefix):
    img_files = glob.glob(osp.join(img_dir, 'image/*.tif'))
    annotations = []
    images = []
    obj_count = 0
    for idx, img_file in enumerate(track_iter_progress(img_files)):
        filename = osp.basename(img_file)
        img_path = osp.join(image_prefix, filename)  # use the full path of the image file
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
