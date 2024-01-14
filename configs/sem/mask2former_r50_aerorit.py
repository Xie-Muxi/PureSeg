# _base_ = [
#     # '_base_/default_runtime.py',
#     # '_base_/datasets/potsdam.py'
# ]

from mmpretrain.models.backbones import ViTEVA02
from mmseg.datasets import AeroRITDataSet

# from rsseg.datasets.transformers import LoadHyperspectralImageFromFile
from mmseg.datasets.transforms import LoadHyperspectralImageFromFile
from mmpretrain.models import VisionTransformer

max_epochs = 1000

batch_size = 2
num_workers = batch_size
start_lr = 0.1
val_interval = 5

custom_imports = dict(
    imports=["mmdet.models", "mmpretrain.models"], allow_failed_imports=False
)

crop_size = (512, 512)
num_bands = 51
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[110.0] * num_bands,
    std=[55.0] * num_bands,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32),
)
num_classes = 6
num_bands = 51

checkpoint = "https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-tiny-p14_pre_in21k_20230505-d703e7b1.pth"  # noqa
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    # data_preprocessor=None,
    backbone=dict(
        type="ResNet",
        depth=50,
        deep_stem=False,
        in_channels=num_bands,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="SyncBN", requires_grad=False),
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    decode_head=dict(
        type="Mask2FormerHead",
        in_channels=[256, 512, 1024, 2048],
        # in_channels=[192] * 4,
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type="mmdet.MSDeformAttnPixelDecoder",
            num_outs=3,
            # num_outs=num_bands,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None,
                    ),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                ),
                init_cfg=None,
            ),
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=128, normalize=True
            ),
            init_cfg=None,
        ),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True
        ),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type="ReLU", inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                ),
            ),
            init_cfg=None,
        ),
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            reduction="mean",
            class_weight=[1.0] * num_classes + [0.1],
        ),
        loss_mask=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=True,
            reduction="mean",
            loss_weight=5.0,
        ),
        loss_dice=dict(
            type="mmdet.DiceLoss",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
        ),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type="mmdet.HungarianAssigner",
                match_costs=[
                    dict(type="mmdet.ClassificationCost", weight=2.0),
                    dict(
                        type="mmdet.CrossEntropyLossCost", weight=5.0, use_sigmoid=True
                    ),
                    dict(type="mmdet.DiceCost", weight=5.0, pred_act=True, eps=1.0),
                ],
            ),
            sampler=dict(type="mmdet.MaskPseudoSampler"),
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# dataset settings
dataset_type = AeroRITDataSet

data_root = "data/AeroRIT/aerorit"

train_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type="LoadSingleRSImageFromFile"),
    # dict(type='LoadHyperspectralImageFromFile',
    #      imdecode_backend='tifffile'),  # !!
    # dict(type='LoadTiffImageFromFile', imdecode_backend='tifffile'),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    # dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(
    #     type='RandomResize',
    #     scale=(512, 512),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    # dict(type='PhotoMetricDistortion'), #! 停用测光畸变，否则会涉及opencv库，无法读取高光谱图像
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type="LoadSingleRSImageFromFile"),
    # dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="PackSegInputs"),
]


img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type="LoadSingleRSImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type="RandomFlip", prob=0.0, direction="horizontal"),
                dict(type="RandomFlip", prob=1.0, direction="horizontal"),
            ],
            [dict(type="LoadAnnotations")],
            [dict(type="PackSegInputs")],
        ],
    ),
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    # sampler=dict(type='InfiniteSampler', shuffle=True),
    # ! it's important to slove "Been training on the first Epoch"
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="img_dir/train", seg_map_path="ann_dir/train"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"])

test_evaluator = val_evaluator

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer = dict(
#     type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="Adam",
        lr=start_lr,
        weight_decay=1e-2,
    ),
)


# learning policy
param_scheduler = [
    dict(type="LinearLR", start_factor=start_lr, by_epoch=False, begin=0, end=1),
    dict(
        type="CosineAnnealingLR",
        by_epoch=True,
        begin=1,
        T_max=max_epochs,
        end=max_epochs,
    ),
]

# training schedule for 160k
train_cfg = dict(·
    type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=val_interval
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=500, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=5,
        save_best="mIoU",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


# Modify from default_runtime.py
default_scope = "mmseg"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
# vis_backends = [dict(type='LocalVisBackend')]
vis_backends = [
    dict(type="LocalVisBackend"),
    # dict(type='WandbVisBackend',
    #      init_kwargs=dict(
    #          project='pure-seg',
    #          name=f'mask2former_eva02-small-tiny_lr={start_lr}_{dataset_type}_{max_epochs}e',
    #          group='mask2former',
    #          tags=['mask2former', 'eva02', f'{dataset_type}'],
    #          #  resume=True
    #      )
    #      )
]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
log_processor = dict(by_epoch=True)
log_level = "INFO"
load_from = None
resume = False

tta_model = dict(type="SegTTAModel")
