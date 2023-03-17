import json
import logging
import time

import torch
import torchvision.models.detection
from torch.utils.data import DataLoader

import mydata

# 思源宋体
if __name__ == '__main__':
    # 相关设置
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    train_rcnn = True
    epoch = 100
    # 加载数据集
    dataset = mydata.myDataset('data/VRD', 'train', train_rcnn, device=device)
    dataloader = DataLoader(dataset,
                            shuffle= True)
    # 创建日志
    logger = logging.getLogger('logger')
    sh = logging.StreamHandler()
    fh = logging.FileHandler('log/{}_log.txt'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())),
                             encoding='utf-8')
    logger.addHandler(sh)
    logger.addHandler(fh)
    lf = logging.Formatter(fmt='%(asctime)s | %(levelname)s | %(message)s',
                           datefmt='%Y_%m_%d %H:%M:%S')
    sh.setFormatter(lf)
    fh.setFormatter(lf)
    logger.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)
    if train_rcnn:
        # 训练用于目标检测的网络并输出参数保存
        rcnn_module = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=101,
                                                                              weights_backbone="IMAGENET1K_V2")
        rcnn_module.train()
        rcnn_module.to(device=device)
        optimizer = torch.optim.Adam(rcnn_module.parameters(), 1e-3)
        t0 = time.time()
        for i in range(10):
            for iter_id, (images, targets) in enumerate(dataloader):
                losses = rcnn_module(images, targets)
                all_loss = 0
                optimizer.zero_grad()
                for loss in losses:
                    all_loss += losses[loss]
                all_loss.backward()
                optimizer.step()
                if iter_id % 100 == 0:
                    t1 = time.time() - t0
                    logger.info('[Epoch {:d}] [Iter {:d}/4000] '
                                '[Loss: loss_classifier {:.2f} || loss_box_reg {:.2f} '
                                '|| loss_objectness {:.2f} || loss_rpn_box_reg {:.2f} || total {:.2f} ]'
                                '[time: {:.2f}]'.format(i, iter_id, losses['loss_classifier'], losses['loss_box_reg'],
                                                        losses['loss_objectness'], losses['loss_rpn_box_reg']
                                                        , all_loss, t1))
                    t0 = time.time()
        # 保存模型到本地
        torch.save({'rcnn_module': rcnn_module,
                    'optimizer': optimizer}, 'module/frcnn.pth')
    pass
