import torch

if __name__ == '__main__':
    # 相关设置
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    pass
    # 加载测试集并验证
