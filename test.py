import torch
import torchvision

import mydata
import mymodel

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]
    # inter[:, :, 0] is the width of intersection and inter[:, :, 1] is height


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [A,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [B,4]
    Return:
        jaccard overlap: (tensor) Shape: [A, B]
    """
    box_a = torch.tensor(box_a).unsqueeze(0)
    box_b = box_b.unsqueeze(0).to('cpu')
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return (inter / union).to('cuda:0')  # [A,B]


if __name__ == '__main__':
    # 相关设置
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # 加载测试集并验证
    dataset = mydata.TestDataset('data/VRD', device)
    r1 = 0
    r10 = 0
    r100 = 0
    num = 0
    # 加载模型
    rcnn_module = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=101).to(device)
    rcnn_module.load_state_dict(torch.load('module/frcnn.pt', map_location=device))
    rcnn_module.eval()
    mymodule = mymodel.RelationModule().to(device)
    mymodule.load_state_dict(torch.load('module/rmodule.pt', map_location=device))
    mymodule.eval()
    for i in range(1000):
        image, target = dataset[i]
        # 不使用dataloader需要人为扩维
        image = image.unsqueeze(0)
        h = image.size(-1)*2
        w = image.size(-2)*2
        # 跳过没有标签的数据
        if len(target) > 0:
            num += 1
            opt = rcnn_module(image)[0]
            ans = []
            for x in range(1,11):
                for y in range(0, x):
                    aa = {'subject': opt['labels'][y],
                          'bbox1': opt['boxes'][y, :],
                          'object': opt['labels'][x],
                          'bbox2': opt['boxes'][x, :],
                          'scores': opt['scores'][y] + opt['scores'][x]}
                    inputs = torch.tensor([aa['subject'] / 101,
                                          (aa['bbox2'][0]+aa['bbox2'][2]-aa['bbox1'][0]-aa['bbox1'][2]) / w,
                                          (aa['bbox2'][1]+aa['bbox2'][3]-aa['bbox1'][1]-aa['bbox1'][3]) / h,
                                          aa['object'] / 101], device=device)
                    outputs = mymodule(inputs.unsqueeze(0))
                    aa['predicate'] = outputs.argmax()
                    aa['scores'] = (aa['scores'] + outputs[:,aa['predicate']]) / 3
                    ans.append(aa)
            for x in range(1,11):
                for y in range(0, x):
                    aa = {'subject': opt['labels'][x],
                          'bbox1': opt['boxes'][x, :],
                          'object': opt['labels'][y],
                          'bbox2': opt['boxes'][y, :],
                          'scores': opt['scores'][y] + opt['scores'][x]}
                    inputs = torch.tensor([aa['subject'] / 101,
                                          (aa['bbox2'][0]+aa['bbox2'][2]-aa['bbox1'][0]-aa['bbox1'][2]) / w,
                                          (aa['bbox2'][1]+aa['bbox2'][3]-aa['bbox1'][1]-aa['bbox1'][3]) / h,
                                          aa['object'] / 101], device=device)
                    outputs = mymodule(inputs.unsqueeze(0))
                    aa['predicate'] = outputs.argmax()
                    aa['scores'] = (aa['scores'] + outputs[:,aa['predicate']]) / 3
                    ans.append(aa)
            ans.sort(key=lambda a:a['scores'], reverse=True)
            # 计算recall
            rx1 = 0
            rx10 = 0
            rx100 = 0
            for x in range(100):
                if x < 1:
                    for t in target:
                        if ans[x]['predicate'] == (t[10]):
                            if ans[x]['object'] == (t[0]):
                                if ans[x]['subject'] == (t[5]):
                                    if jaccard(t[1:5], ans[x]['bbox2']) > 0.5:
                                        if jaccard(t[6:10], ans[x]['bbox2']) > 0.5:
                                            rx1 += 1
                                            rx10 += 1
                                            rx100 += 1
                                            break
                elif x < 50:
                    for t in target:
                        if ans[x]['predicate']==(t[10]):
                            if ans[x]['object']==(t[0]):
                                if ans[x]['subject']==(t[5]):
                                    if jaccard(t[1:5], ans[x]['bbox2']) > 0.5:
                                        if jaccard(t[6:10], ans[x]['bbox2']) > 0.5:
                                            rx10 += 1
                                            rx100 += 1
                                            break
                else:
                    for t in target:
                        if ans[x]['predicate']==(t[10]):
                            if ans[x]['object']==(t[0]):
                                if ans[x]['subject']==(t[5]):
                                    if jaccard(t[1:5], ans[x]['bbox2']) > 0.5:
                                        if jaccard(t[6:10], ans[x]['bbox2']) > 0.5:
                                            rx100 += 1
                                            break
            r1 += rx1 / len(target)
            r10 += rx10 / len(target)
            r100 += rx100 / len(target)
    print('r1{:3f},r10{:3f},r100{:3f}'.format(r1/num,r10/num,r100/num))
    pass
