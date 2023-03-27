import logging
import time

import torch
import torchvision.models.detection
from torch.utils.data import DataLoader

import mydata
import mymodel

if __name__ == '__main__':
    # 相关设置
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    train_rcnn = False
    train_mymodule = True
    epoch = 50
    r_b_s = 1
    m_b_s = 128
    # 加载数据集
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
        dataset = mydata.RcnnDataset('data/VRD', 'train', device=device)
        dataloader = DataLoader(dataset, batch_size=r_b_s,
                                shuffle=True, collate_fn=mydata.collate_fn)
        # 训练用于目标检测的网络并输出参数保存
        rcnn_module = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=101,
                                                                              weights_backbone="IMAGENET1K_V2")
        rcnn_module.train()
        rcnn_module.to(device=device)
        optimizer = torch.optim.SGD([p for p in rcnn_module.parameters() if p.requires_grad], lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=5,
                                                       gamma=0.33)
        logger.info('train faster rcnn module')
        t0 = time.time()
        for i in range(epoch):
            for iter_id, (images, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                losses = rcnn_module(images, targets)
                all_loss = 0
                for loss in losses:
                    all_loss += losses[loss]
                all_loss.backward()
                optimizer.step()
                if iter_id % 100 == 0:
                    t1 = time.time() - t0
                    logger.info('[Epoch {:d}] [Iter {:d}/{:d}] '
                                '[Loss: loss_classifier {:.2f} || loss_box_reg {:.2f} '
                                '|| loss_objectness {:.2f} || loss_rpn_box_reg {:.2f} || total {:.2f} ]'
                                '[time: {:.2f}]'.format(i, iter_id, 4000//r_b_s,
                                                        losses['loss_classifier'], losses['loss_box_reg'],
                                                        losses['loss_objectness'], losses['loss_rpn_box_reg']
                                                        , all_loss, t1))
                    t0 = time.time()
            lr_scheduler.step()
        # 保存模型到本地
        torch.save(rcnn_module.state_dict(), 'module/frcnn.pt')
    else:
        rcnn_module = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=101)
        rcnn_module.load_state_dict(torch.load('module/frcnn.pt', map_location=device))
        rcnn_module.eval()
    if train_mymodule:
        dataset = mydata.MyModuleDataset('data/VRD', 'train', device=device)
        dataloader = DataLoader(dataset, batch_size=m_b_s,
                                shuffle=True)
        module = mymodel.RelationModule()
        module.train()
        module.to(device)
        optimizer = torch.optim.Adam([p for p in module.parameters() if p.requires_grad], lr=0.0005)
        logger.info('train relation module')
        t0 = time.time()
        for i in range(50):
            for iter_id, (data, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                output = module(data)
                loss = -targets * torch.log(output) - (1 - targets) * torch.log(1 - output)
                loss = torch.nan_to_num(loss, nan=0)
                loss = loss.sum() / targets.size(0)
                loss.backward()
                optimizer.step()
                if iter_id % 100 == 0:
                    t1 = time.time() - t0
                    logger.info('[Epoch {:d}] [Iter {:d}/{:d}] '
                                '[Loss: {:.2f} ]'
                                '[time: {:.2f}]'.format(i, iter_id, 30355//m_b_s,
                                                        loss, t1))
                    t0 = time.time()
        torch.save(module.state_dict(), 'module/rmodule.pt')
    else:
        module = mymodel.RelationModule()
        module.load_state_dict(torch.load('module/rmodule.pt', map_location=device))
        module.eval()
    pass
