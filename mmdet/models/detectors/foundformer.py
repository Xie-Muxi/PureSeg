# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .mask2former import Mask2Former


@MODELS.register_module()
class FoundFormer(Mask2Former):
    r"""Implementation of `FoundFormer: Find Your Mask
    <https://arxiv.org/pdf/xxx>`_."""

    def __init__(
        self,
        backbone: ConfigType,
        neck: OptConfigType = None,
        panoptic_head: OptConfigType = None,
        panoptic_fusion_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
