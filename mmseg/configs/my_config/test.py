_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
                     in_channels=512,
                     channels=128, 
                     num_classes=6),
    auxiliary_head=dict(num_classes=6,in_channels=256, channels=64))

# training schedule for 160k
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop', 
    max_epochs=10, 
    val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    # timer=dict(type='EpochTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
    # param_scheduler=dict(type='ParamSchedulerHook',convert_to_iter_based=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
log_processor = dict(by_epoch=True)

