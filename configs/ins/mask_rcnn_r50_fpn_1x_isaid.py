_base_ = [
    './_base_/models/mask-rcnn_r50_fpn.py',
    './_base_/datasets/isaid_instance.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

log_processor = dict(type='LogProcessor',
                     window_size=200, by_epoch=True)

val_interval = 2
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # checkpoint=dict(type='CheckpointHook', interval=4,max_keep_ckpts=3),
    checkpoint=dict(type='CheckpointHook',
                    interval=val_interval,
                    max_keep_ckpts=3,
                    save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
