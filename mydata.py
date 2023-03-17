import json
import os
from typing import Any, Tuple

import torch.utils.data
import torchvision.io
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights


class myDataset(torch.utils.data.Dataset):

    def __init__(self, root: str,
                 split: str,
                 t_rcnn: bool = False,
                 device = None) -> None:
        self.split = split
        self.t_rcnn = t_rcnn
        root = os.path.expanduser(root)
        self.root = root
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        with open(os.path.join(self.root, 'objects.json')) as f:
            self.objects = json.load(f)
        with open(os.path.join(self.root, 'predicates.json')) as f:
            self.predicates = json.load(f)
        with open(os.path.join(self.root, 'annotations_' + split + '.json')) as f:
            self.ann = json.load(f)
        self.ids = list(self.ann.keys())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = os.path.join(self.root, self.split, self.ids[index])
        image = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB).to(self.device)
        target = []
        if self.t_rcnn:
            for li in self.ann[self.ids[index]]:
                for k in ['object', 'subject']:
                    h = {'boxes': torch.tensor([li[k]['bbox'][2], li[k]['bbox'][0], li[k]['bbox'][3], li[k]['bbox'][1]],
                                               device=self.device,
                                               dtype=torch.float),
                         'labels': torch.tensor(li[k]['category'] + 1, device=self.device)}
                    t_h = True
                    # 检测重复物体
                    for t in target:
                        if h['boxes'].equal(t['boxes']) and h['labels'].equal(t['labels']):
                            t_h = False
                            break
                    if t_h:
                        target.append(h)
            # 处理无标签情况
            if len(target) == 0:
                h = {'boxes': torch.tensor([0, 0, image.size(1), image.size(2)],
                                           device=self.device,
                                           dtype=torch.float),
                     'labels': torch.tensor(0, device=self.device)}
                target.append(h)
        else:
            pass
        preprocess = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
        image = preprocess(image)
        return image, target

    def __len__(self):
        return len(self.ids)
