import json
import os
from typing import Any, Tuple

import torch.utils.data
import torchvision.io
import torchvision.transforms.v2 as transforms

from torchvision import datapoints


class RcnnDataset(torch.utils.data.Dataset):

    def __init__(self, root: str,
                 split: str,
                 device=None) -> None:
        self.split = split
        root = os.path.expanduser(root)
        self.root = root
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        self.objects = ["person", "sky", "building", "truck", "bus", "table", "shirt", "chair", "car", "train",
                        "glasses", "tree", "boat", "hat", "trees", "grass", "pants", "road", "motorcycle", "jacket",
                        "monitor", "wheel", "umbrella", "plate", "bike", "clock", "bag", "shoe", "laptop", "desk",
                        "cabinet", "counter", "bench", "shoes", "tower", "bottle", "helmet", "stove", "lamp", "coat",
                        "bed", "dog", "mountain", "horse", "plane", "roof", "skateboard", "traffic light", "bush",
                        "phone", "airplane", "sofa", "cup", "sink", "shelf", "box", "van", "hand", "shorts", "post",
                        "jeans", "cat", "sunglasses", "bowl", "computer", "pillow", "pizza", "basket", "elephant",
                        "kite", "sand", "keyboard", "plant", "can", "vase", "refrigerator", "cart", "skis", "pot",
                        "surfboard", "paper", "mouse", "trash can", "cone", "camera", "ball", "bear", "giraffe",
                        "tie", "luggage", "faucet", "hydrant", "snowboard", "oven", "engine", "watch", "face",
                        "street", "ramp", "suitcase"]
        self.predicates = ["on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", "park next",
                           "walk next to", "above", "behind", "stand behind", "sit behind", "park behind",
                           "in the front of", "under", "stand under", "sit under", "near", "walk to", "walk",
                           "walk past", "in", "below", "beside", "walk beside", "over", "hold", "by", "beneath",
                           "with", "on the top of", "on the left of", "on the right of", "sit on", "ride", "carry",
                           "look", "stand on", "use", "at", "attach to", "cover", "touch", "watch", "against",
                           "inside", "adjacent to", "across", "contain", "drive", "drive on", "taller than", "eat",
                           "park on", "lying on", "pull", "talk", "lean on", "fly", "face", "play with", "sleep on",
                           "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"]
        with open(os.path.join(self.root, 'annotations_' + split + '.json')) as f:
            self.ann = json.load(f)
        self.ids = list(self.ann.keys())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = os.path.join(self.root, self.split, self.ids[index])
        image = datapoints.Image(torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB)).to(self.device)
        boxes = []
        labels = []
        for li in self.ann[self.ids[index]]:
            for k in ['object', 'subject']:
                h0 = [li[k]['bbox'][2], li[k]['bbox'][0], li[k]['bbox'][3], li[k]['bbox'][1]]
                # 检测重复物体
                if h0 not in boxes:
                    boxes.append(h0)
                    labels.append(li[k]['category'] + 1)
        target = {'boxes': torch.tensor(boxes, device=self.device, dtype=torch.float),
                  'labels': torch.tensor(labels, device=self.device, dtype=torch.int64)}
        # 处理无标签情况
        if len(self.ann[self.ids[index]]) == 0:
            h0 = torch.zeros((1, 4), device=self.device)
            h0[0, 2] = image.size(1)
            h0[0, 3] = image.size(2)
            target = {'boxes': h0,
                      'labels': torch.zeros((1), device=self.device, dtype=torch.int64)}
        target['boxes'] = datapoints.BoundingBox(target['boxes'], format=datapoints.BoundingBoxFormat.XYXY,
                                                 spatial_size=image.shape[-2:], device=self.device)
        # 数据预处理，变色翻转
        if self.split == 'train':
            transform = transforms.Compose(
                [
                    transforms.RandomPhotometricDistort(),
                    transforms.RandomIoUCrop(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToImageTensor(),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.SanitizeBoundingBox()
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToImageTensor(),
                    transforms.ConvertImageDtype(torch.float32)
                ]
            )
        image, target = transform(image, target)
        return image, target

    def __len__(self):
        return len(self.ids)


class MyModuleDataset(torch.utils.data.Dataset):
    def __init__(self, root: str,
                 split: str,
                 device=None) -> None:
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        self.split = split
        root = os.path.expanduser(root)
        self.root = root
        with open(os.path.join(self.root, 'annotations_' + split + '.json')) as f:
            self.ann = json.load(f)
        self.data = []
        for im in self.ann:
            if len(self.ann[im]) > 0:
                path = os.path.join(self.root, self.split, im)
                image = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB)
                h = image.size(-1)
                w = image.size(-2)
                for t in self.ann[im]:
                    l0 = [t['object']['category'] + 1, t['object']['bbox'][2] / w, t['object']['bbox'][0] / h,
                          t['object']['bbox'][3] / w, t['object']['bbox'][1] / h, t['subject']['category'] + 1,
                          t['subject']['bbox'][2] / w, t['subject']['bbox'][0] / h, t['subject']['bbox'][3] / w,
                          t['subject']['bbox'][1] / h, t['predicate']]
                    self.data.append(l0)
        del self.ann

    def __getitem__(self, index) -> Tuple[Any, Any]:
        # data = torch.tensor(self.data[index][0:10], device=self.device)
        # data[0] = data[0] / 101
        # data[5] = data[5] / 101
        data0 = torch.tensor(self.data[index][0:10], device=self.device)
        data1 = torch.zeros(4, device=self.device)
        data1[0] = data0[5] / 101
        data1[1] = (data0[1] + data0[3] - data0[6] - data0[8]) / 2
        data1[2] = (data0[2] + data0[4] - data0[7] - data0[9]) / 2
        data1[3] = data0[0] / 101
        target = torch.zeros(80, device=self.device)
        target[self.data[index][-1]] = 1
        # if torch.rand(1) < 0.5:
        #     s = data[0:5]
        #     data[0:5] = data[5:10]
        #     data[5:10] = s
        return data1, target

    def __len__(self):
        return len(self.data)


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, root: str,
                 device=None) -> None:
        root = os.path.expanduser(root)
        self.root = root
        if device is None:
            torch.device('cpu')
        else:
            self.device = device
        with open(os.path.join(self.root, 'annotations_test.json')) as f:
            self.ann = json.load(f)
        self.ids = list(self.ann.keys())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = os.path.join(self.root, 'test', self.ids[index])
        image = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB).to(self.device)
        target = []
        for t in self.ann[self.ids[index]]:
            l0 = [t['object']['category'] + 1, t['object']['bbox'][2], t['object']['bbox'][0],
                  t['object']['bbox'][3], t['object']['bbox'][1], t['subject']['category'] + 1,
                  t['subject']['bbox'][2], t['subject']['bbox'][0], t['subject']['bbox'][3],
                  t['subject']['bbox'][1], t['predicate']]
            target.append(l0)
        # 数据预处理
        transform = transforms.Compose(
            [
                transforms.ToImageTensor(),
                transforms.ConvertImageDtype(torch.float32)
            ]
        )
        image = transform(image)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return list(zip(*batch))
