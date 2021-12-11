import os
import logging
import shutil
import torch
from os.path import join

def logger_config():
    path = 'log.txt'
    name = 'eecs545finalproject'
    formatter = logging.Formatter('%(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    filehandler = logging.FileHandler(path, encoding='UTF-8')
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(formatter)

    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(formatter)

    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    return logger

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, path, filename="checkpoint.pth"):
    torch.save(state, join(path, filename))
    if is_best:
        shutil.copyfile(join(path, filename), join(path, 'model_best.pth'))


@ torch.no_grad()
def accuracy(output, class_):
    bs = class_.size(0)
    _, prediction = output.topk(1, 1, True, True)
    prediction = prediction.t()
    correct = prediction.eq(class_.view(1, -1).expand_as(prediction)).reshape(-1).float().sum(0, keepdim=True)
    res = [correct.mul_(100.0 / bs)]
    return res
