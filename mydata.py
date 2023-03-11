import json
import os
from typing import Any, Tuple

import torch.utils.data
from PIL import Image


class myDataset(torch.utils.data.Dataset):

    def __init__(self, root: str,
                 split: str,
                 t_rcnn: bool = False) -> None:
        self.split = split
        self.t_rcnn = t_rcnn
        root = os.path.expanduser(root)
        self.root = root
        with open(os.path.join(self.root, 'objects.json')) as f:
            self.objects = json.load(f)
        with open(os.path.join(self.root, 'predicates.json')) as f:
            self.predicates = json.load(f)
        with open(os.path.join(self.root, 'annotations_' + split + '.json')) as f:
            self.ann = json.load(f)
        self.ids = list(self.ann.keys())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = os.path.join(self.root, self.split, self.ids[index])
        image = Image.open(path).convert("RGB")
        target = []
        if self.t_rcnn:
            for li in self.ann[self.ids[index]]:
                for k in ['object', 'subject']:
                    h = li[k]['bbox'].append(li[k]['category'])
                    if h not in target:
                        target.append(h)
        else:
            pass
        return image, target
