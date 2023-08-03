backend_args = None
callbacks = [
    dict(type='mmengine.hooks.ParamSchedulerHook'),
    dict(
        dirpath='results/dev/E20230629_1/checkpoints',
        filename='epoch_{epoch}-map_{valsegm_map_0:.4f}',
        mode='max',
        monitor='valsegm_map_0',
        save_last=True,
        save_top_k=3,
        type='ModelCheckpoint'),
    dict(logging_interval='step', type='LearningRateMonitor'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmseg.datasets',
        'mmseg.models',
    ])
data_parent = '/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset/'
data_preprocessor = dict(
    bgr_to_rgb=True,
    mask_pad_value=0,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_mask=True,
    pad_size_divisor=32,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='mmdet.models.data_preprocessors.DetDataPreprocessor')
datamodule_cfg = dict(
    predict_loader=dict(
        batch_size=2,
        dataset=dict(
            ann_file='nwpu-instances_val.json',
            backend_args=None,
            data_prefix=dict(img_path='positive image set'),
            data_root='/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset/',
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(backend_args=None, type='mmdet.LoadImageFromFile'),
                dict(scale=(
                    1024,
                    1024,
                ), type='mmdet.Resize'),
                dict(
                    type='mmdet.LoadAnnotations',
                    with_bbox=True,
                    with_mask=True),
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
            test_mode=True,
            type='NWPUInsSegDataset'),
        num_workers=2,
        persistent_workers=True,
        pin_memory=True),
    train_loader=dict(
        batch_size=2,
        dataset=dict(
            ann_file='nwpu-instances_train.json',
            backend_args=None,
            data_prefix=dict(img_path='positive image set'),
            data_root='/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset/',
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(type='mmdet.LoadImageFromFile'),
                dict(
                    type='mmdet.LoadAnnotations',
                    with_bbox=True,
                    with_mask=True),
                dict(scale=(
                    1024,
                    1024,
                ), type='mmdet.Resize'),
                dict(prob=0.5, type='mmdet.RandomFlip'),
                dict(type='mmdet.PackDetInputs'),
            ],
            type='NWPUInsSegDataset'),
        num_workers=2,
        persistent_workers=True,
        pin_memory=True),
    type='PLDataModule',
    val_loader=dict(
        batch_size=2,
        dataset=dict(
            ann_file='nwpu-instances_val.json',
            backend_args=None,
            data_prefix=dict(img_path='positive image set'),
            data_root='/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset/',
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(backend_args=None, type='mmdet.LoadImageFromFile'),
                dict(scale=(
                    1024,
                    1024,
                ), type='mmdet.Resize'),
                dict(
                    type='mmdet.LoadAnnotations',
                    with_bbox=True,
                    with_mask=True),
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
            test_mode=True,
            type='NWPUInsSegDataset'),
        num_workers=2,
        persistent_workers=True,
        pin_memory=True))
dataset_type = 'NWPUInsSegDataset'
evaluator = dict(
    val_evaluator=dict(
        metric=[
            'bbox',
            'segm',
        ],
        proposal_nums=[
            1,
            10,
            100,
        ],
        type='mmpl.evaluation.metrics.CocoPLMetric'))
evaluator_ = dict(
    metric=[
        'bbox',
        'segm',
    ],
    proposal_nums=[
        1,
        10,
        100,
    ],
    type='mmpl.evaluation.metrics.CocoPLMetric')
exp_name = 'E20230629_1'
image_size = (
    1024,
    1024,
)
logger = dict(
    group='sam-anchor', name='E20230629_1', project='dev', type='WandbLogger')
max_epochs = 100
model_cfg = dict(
    backbone=dict(
        checkpoint='pretrain/sam/sam_vit_h_4b8939.pth', type='vit_h'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mask_pad_value=0,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.models.data_preprocessors.DetDataPreprocessor'),
    hyperparameters=dict(
        evaluator=dict(
            val_evaluator=dict(
                metric=[
                    'bbox',
                    'segm',
                ],
                proposal_nums=[
                    1,
                    10,
                    100,
                ],
                type='mmpl.evaluation.metrics.CocoPLMetric')),
        optimizer=dict(
            lr=0.0005,
            sub_model=dict(panoptic_head=dict(lr_mult=1)),
            type='torch.optim.adamw.AdamW',
            weight_decay=0.001),
        param_scheduler=[
            dict(
                begin=0,
                by_epoch=True,
                convert_to_iter_based=True,
                end=1,
                start_factor=0.0001,
                type='mmengine.optim.scheduler.LinearLR'),
            dict(
                T_max=100,
                begin=1,
                by_epoch=True,
                end=100,
                type='mmengine.optim.scheduler.CosineAnnealingLR'),
        ]),
    need_train_names=[
        'panoptic_head',
        'data_preprocessor',
    ],
    panoptic_head=dict(
        neck=dict(
            in_channels=[
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
                1280,
            ],
            inner_channels=32,
            out_channels=256,
            selected_channels=range(8, 32, 2),
            type='mmpl.models.SAMAggregatorNeck',
            up_sample_scale=4),
        roi_head=dict(
            bbox_head=dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='mmdet.DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=1.0, type='mmdet.SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=10,
                reg_class_agnostic=False,
                roi_feat_size=7,
                type='mmdet.Shared2FCBBoxHead'),
            bbox_roi_extractor=dict(
                featmap_strides=[
                    8,
                    16,
                    32,
                ],
                out_channels=256,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='mmdet.SingleRoIExtractor'),
            mask_head=dict(
                class_agnostic=True,
                loss_mask=dict(
                    loss_weight=1.0,
                    type='mmdet.CrossEntropyLoss',
                    use_mask=True),
                per_query_point=4,
                type='SAMPromptMaskHead',
                with_sincos=True),
            mask_roi_extractor=dict(
                featmap_strides=[
                    8,
                    16,
                    32,
                ],
                out_channels=256,
                roi_layer=dict(
                    output_size=14, sampling_ratio=0, type='RoIAlign'),
                type='mmdet.SingleRoIExtractor'),
            type='SAMAnchorPromptRoIHead'),
        rpn_head=dict(
            anchor_generator=dict(
                ratios=[
                    0.5,
                    1.0,
                    2.0,
                ],
                scales=[
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                ],
                strides=[
                    8,
                    16,
                    32,
                ],
                type='mmdet.models.task_modules.AnchorGenerator'),
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                type='mmdet.DeltaXYWHBBoxCoder'),
            feat_channels=256,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='mmdet.SmoothL1Loss'),
            loss_cls=dict(
                loss_weight=1.0,
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True),
            type='mmdet.models.dense_heads.rpn_head.RPNHead'),
        test_cfg=dict(
            rcnn=dict(
                mask_thr_binary=0.5,
                max_per_img=100,
                nms=dict(iou_threshold=0.5, type='nms'),
                score_thr=0.05),
            rpn=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=1000)),
        train_cfg=dict(
            rcnn=dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='mmdet.MaxIoUAssigner'),
                debug=False,
                mask_size=1024,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.25,
                    type='mmdet.RandomSampler')),
            rpn=dict(
                allowed_border=-1,
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    min_pos_iou=0.3,
                    neg_iou_thr=0.3,
                    pos_iou_thr=0.7,
                    type='mmdet.MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=False,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.5,
                    type='mmdet.RandomSampler')),
            rpn_proposal=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=2000)),
        type='mmpl.models.SAMAnchorInstanceHead'),
    type='mmpl.models.SegSAMAnchorPLer')
num_classes = 10
num_stuff_classes = 0
num_things_classes = 10
optimizer = dict(
    lr=0.0005,
    sub_model=dict(panoptic_head=dict(lr_mult=1)),
    type='torch.optim.adamw.AdamW',
    weight_decay=0.001)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=1,
        start_factor=0.0001,
        type='mmengine.optim.scheduler.LinearLR'),
    dict(
        T_max=100,
        begin=1,
        by_epoch=True,
        end=100,
        type='mmengine.optim.scheduler.CosineAnnealingLR'),
]
param_scheduler_callback = dict(type='mmengine.hooks.ParamSchedulerHook')
persistent_workers = True
prompt_shape = (
    60,
    4,
)
sub_model_optim = dict(panoptic_head=dict(lr_mult=1))
sub_model_train = [
    'panoptic_head',
    'data_preprocessor',
]
task_name = 'dev'
test_batch_size_per_gpu = 2
test_num_workers = 2
test_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_batch_size_per_gpu = 2
train_data_prefix = ''
train_num_workers = 2
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(type='mmdet.PackDetInputs'),
]
trainer_cfg = dict(
    accelerator='auto',
    benchmark=True,
    callbacks=[
        dict(type='mmengine.hooks.ParamSchedulerHook'),
        dict(
            dirpath='results/dev/E20230629_1/checkpoints',
            filename='epoch_{epoch}-map_{valsegm_map_0:.4f}',
            mode='max',
            monitor='valsegm_map_0',
            save_last=True,
            save_top_k=3,
            type='ModelCheckpoint'),
        dict(logging_interval='step', type='LearningRateMonitor'),
    ],
    check_val_every_n_epoch=5,
    compiled_model=False,
    default_root_dir='results/dev/E20230629_1',
    devices=4,
    log_every_n_steps=5,
    logger=dict(
        group='sam-anchor',
        name='E20230629_1',
        project='dev',
        type='WandbLogger'),
    max_epochs=100,
    strategy='auto',
    use_distributed_sampler=True)
val_data_prefix = ''
val_loader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='nwpu-instances_val.json',
        backend_args=None,
        data_prefix=dict(img_path='positive image set'),
        data_root='/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='NWPUInsSegDataset'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True)
