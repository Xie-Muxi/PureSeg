import torch
# from mmengine.structures import InstanceData
from typing import List, Any
from mmengine.model.base_model import BaseModel

# from mmseg.utils import SampleList
# import torch.nn.functional as F
from .sam import sam_model_registry

from rssam.registry import MODELS

@MODELS.register_module()
class SegSAMAnchor(BaseModel):
    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 need_train_names=None,
                 train_cfg=None,
                 test_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.need_train_names = need_train_names

        backbone_type = backbone.pop('type')
        self.backbone = sam_model_registry[backbone_type](**backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        self.panoptic_head = MODELS.build(panoptic_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self):
        pass

    @torch.no_grad()
    def extract_feat(self, batch_inputs):
        feat, inter_features = self.backbone.image_encoder(batch_inputs)
        return feat, inter_features

    def forward(self, inputs, data_samples, mode='tensor'):
        if mode == 'loss':
            x = self.extract_feat(inputs)
            losses = self.panoptic_head.loss(x, data_samples, self.backbone)
            return losses
        elif mode == 'predict':
            x = self.extract_feat(inputs)
            results = self.panoptic_head.predict(x, data_samples, self.backbone)
            return results
        elif mode == 'tensor':
            x = self.extract_feat(inputs)
            return x
