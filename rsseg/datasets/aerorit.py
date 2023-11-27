# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class AeroRITDataSet(BaseSegDataset):
    """AeroRIT dataset.

    In segmentation map annotation for AeroRIT dataset, 5 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` was fixed to 
    '.tif' and seg_map_suffix`` was fixed to '.png'.
    """
    METAINFO = dict(
        classes=('Buildings', 'Vegetation', 'Roads', 'Water', 'Cars',
                 'Unspecified'),
        palette=[[255, 0, 0],  # Buildings,
                 [0, 255, 0],  # Vegetation
                 [0, 0, 255],  # Roads
                 [0, 255, 255],  # Water
                 [255, 127, 80],  # Cars
                 [153, 0, 0],  # Unspecified
                 ]
    )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
