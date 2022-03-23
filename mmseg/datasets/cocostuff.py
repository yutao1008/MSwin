# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class COCOStuffDataset(CustomDataset):
    """COCOStuff dataset.

    In segmentation map annotation for COCOStuff, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for COCOStuff.
    
    """
    CLASSES = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
               'train', 'truck', 'boat', 'trafficlight', 'firehydrant', 'streetsign', 
               'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
               'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 
               'shoe', 'eyeglasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
               'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove', 'skateboard', 
               'surfboard', 'tennisracket', 'bottle', 'plate', 'wineglass', 'cup', 'fork', 
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
               'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'pottedplant', 
               'bed', 'mirror', 'diningtable', 'window', 'desk', 'toilet', 'door', 'tv', 
               'laptop', 'mouse', 'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 
               'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 
               'scissors', 'teddybear', 'hairdrier', 'toothbrush', 'hairbrush', 'banner', 
               'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 
               'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 
               'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 
               'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 
               'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 
               'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 
               'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 
               'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 
               'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 
               'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 
               'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 
               'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 
               'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood']


    def __init__(self, split, **kwargs):
        super(COCOStuffDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

