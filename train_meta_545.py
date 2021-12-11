import argparse
import torch
import random
import os
from os.path import exists
import torch.nn.functional as F
from torch.optim import SGD
from torch.distributions import Beta
import torchvision

from dataloader import dataloader
from utils import AverageMeter, logger_config, accuracy, save_checkpoint
from model import SCNN, vgg11

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-ds', type=str,
                    default='mnist', choices=["mnist", "fmnist", "stl10"])
parser.add_argument('--model', '-m', type=str,
                    default='scnn', choices=['scnn', 'vgg'])
parser.add_argument('--iteration', '-i', type=int, default=100000)
args = parser.parse_args()

data_path = './data/' + args.dataset
result_path = './results/' + args.dataset

random.seed(8469)
torch.manual_seed(8469)

if not exists(result_path):
    os.makedirs(result_path)
logger = logger_config()
train_loader, test_loader = dataloader(
    dset=args.dataset,
    path=data_path,
    iteration=args.iteration,
)
beta_distribution = Beta(torch.tensor([1.0]), torch.tensor([1.0]))


num_classes = 10
kwargs = {'num_classes': num_classes}
if args.model == "scnn":
    model = SCNN(num_classes=num_classes,
                 input_channel=3 if args.dataset == 'stl10' else 1)
elif args.model == "vgg" and args.dataset == "stl10":
    model = torchvision.models.vgg11(pretrained=False, progress=True, **kwargs)
elif args.model == "vgg":
    model = vgg11(pretrained=False, progress=True, **kwargs)
model.cuda()
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9,
                weight_decay=1e-4)


def BatchSampler():
    li, lc, ui, uc = next(train_loader)
    lcm = F.one_hot(lc, num_classes=num_classes).float()
    li = li.cuda()
    lc = lc.cuda()
    ui = ui.cuda()
    uc = uc.cuda()
    lcm = lcm.cuda()
    return li, lc, lcm, ui, uc


def update_learning_rate_and_weight():
    lr = 0.1
    if iter < 4000:
        lr *= iter / 4000
    elif iter > 300000:
        lr *= 0.1
    elif iter > 350000:
        lr *= 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    w = 1 if iter > 4000 else iter / 4000
    return lr, w


def delta_theta_labeled():
    return torch.autograd.grad(labeled_loss_iter, model.parameters(), only_inputs=True)


@torch.no_grad()
def update_pseudo_label():
    upi = model(unlabeled_image)
    upcm = F.softmax(upi, dim=1)
    epsilon = 1e-2 / \
        torch.norm(torch.cat([item.view(-1)
                   for item in delta_theta_labeled_implementation]))

    for para, theta in zip(model.parameters(), delta_theta_labeled_implementation):
        para.data.add_(theta, alpha=epsilon)
    unlabeled_prediction_1 = model(unlabeled_image)
    for para, theta in zip(model.parameters(), delta_theta_labeled_implementation):
        para.data.add_(theta, alpha=2.*epsilon)
    unlabeled_prediction_0 = model(unlabeled_image)
    for para, theta in zip(model.parameters(), delta_theta_labeled_implementation):
        para.data.add_(theta, alpha=epsilon)

    unlabeled_gradient = F.softmax(unlabeled_prediction_1, dim=1) - \
        F.softmax(unlabeled_prediction_0, dim=1)
    unlabeled_gradient.div_(epsilon)

    upcm.sub_(unlabeled_gradient, alpha=learning_rate)
    torch.relu_(upcm)
    total_add = torch.sum(upcm, dim=1, keepdim=True)
    upcm /= torch.where(total_add ==
                        0., torch.ones_like(total_add), total_add)
    return upcm


@ torch.no_grad()
def improved_training_protocol():
    lambda_i_class = beta_distribution.sample((100,))
    lambda_i_class = lambda_i_class.cuda()
    lambda_i_image = lambda_i_class.view(-1, 1, 1, 1)
    ii = (labeled_image * lambda_i_image +
          unlabeled_image * (1. - lambda_i_image)).detach()
    ipcm = (labeled_class_matrix * lambda_i_class +
            unlabeled_pseudo_class_matrix * (1. - lambda_i_class)).detach()
    return ii, ipcm


def log_info():
    if iter % 100 == 0:
        logger.info("['Train': 'Iteration': {0:05d}, \n\t'Labeled': ['Loss': {labeled_loss.val:.3f}, 'Loss avg': {labeled_loss.avg:.3f}, 'Acc': {labeled_accuracy.val:.3f}, 'Acc avg': {labeled_accuracy.avg:.3f}], \n\t'Unlabeled': ['Loss': {unlabeled_loss.val:.3f}, 'Loss avg': {unlabeled_loss.avg:.3f}, 'Acc': {unlabeled_accuracy.val:.3f}, 'Acc avg': {unlabeled_accuracy.avg:.3f}]\n]".format(iter, labeled_loss=labeled_loss,
                                                                                                                                                                                                                                                                                                                                                                                                       labeled_accuracy=labeled_accuracy, unlabeled_loss=unlabeled_loss,  unlabeled_accuracy=unlabeled_accuracy))


@torch.no_grad()
def test_eval():
    test_loss = AverageMeter()
    test_accuracy = AverageMeter()
    model.eval()
    for i, (test_image, test_class) in enumerate(test_loader):
        test_image = test_image.cuda()
        test_class = test_class.cuda()

        test_prediction = model(test_image)
        test_loss_iter = F.cross_entropy(
            test_prediction, test_class, reduction='mean')

        test_accuracy_first, = accuracy(test_prediction, test_class)
        test_loss.update(test_loss_iter.item(), test_image.size(0))
        test_accuracy.update(test_accuracy_first.item(), test_image.size(0))

        if i % 400 == 0:
            logger.info("['Test': \n\t['Loss': {test_loss.val:.3f}, 'Loss avg': {test_loss.avg:.3f}, 'Acc': {test_accuracy.val:.3f}, 'Acc avg': {test_accuracy.avg:.3f}]\n]"
                        .format(test_loss=test_loss, test_accuracy=test_accuracy))
    return test_accuracy.avg


def save_check_point(best_test_accuracy):
    if (iter + 1) % 400 == 0 or iter == args.iteration - 1:
        test_accuracy = test_eval()
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
        logger.info("['Best accuracy': %.5f]" % best_test_accuracy)
        save_checkpoint({
            'Iteration': iter + 1,
            'Model': model.state_dict(),
            'Best_test_accuracy': best_test_accuracy,
            'Optimizer': optimizer.state_dict()
        }, test_accuracy > best_test_accuracy, path=result_path, filename="checkpoint.pth")
        labeled_loss.reset()
        labeled_accuracy.reset()
        unlabeled_loss.reset()
        unlabeled_accuracy.reset()
        interp_losses.reset()
    return best_test_accuracy


def show_info(idx):
    print(str(idx) + " : " + str(labeled_loss_iter))


if __name__ == "__main__":
    best_test_accuracy = 0.0
    labeled_loss = AverageMeter()
    labeled_accuracy = AverageMeter()
    unlabeled_loss = AverageMeter()
    unlabeled_accuracy = AverageMeter()
    interp_losses = AverageMeter()
    for iter in range(args.iteration):
        labeled_image, labeled_class, labeled_class_matrix, unlabeled_image, unlabeled_class = BatchSampler()

        learning_rate, weight = update_learning_rate_and_weight()
        print(learning_rate, weight)

        model.eval()

        labeled_prediction_iter = model(labeled_image)
        labeled_loss_iter = F.cross_entropy(
            labeled_prediction_iter, labeled_class, reduction='mean')
        print(labeled_prediction_iter, labeled_loss_iter)
        delta_theta_labeled_implementation = delta_theta_labeled()

        unlabeled_pseudo_class_matrix = update_pseudo_label()

        model.train()

        interpolate_image, interpolate_pseudo_class_matrix = improved_training_protocol()

        interpolare_prediction_iter = model(interpolate_image)
        interpolare_loss_iter = F.kl_div(F.log_softmax(
            interpolare_prediction_iter, dim=1), interpolate_pseudo_class_matrix, reduction='batchmean')

        unlabeled_prediction_iter = model(unlabeled_image)
        unlabeled_loss_iter = torch.norm(
            F.softmax(unlabeled_prediction_iter, dim=1)-unlabeled_pseudo_class_matrix, p=2, dim=1).pow(2).mean()
        overall_loss = interpolare_loss_iter + weight * unlabeled_loss_iter

        optimizer.zero_grad()
        overall_loss.backward()
        optimizer.step()

        labeled_accuracy_first, = accuracy(
            labeled_prediction_iter, labeled_class)
        unlabeled_accuracy_first, = accuracy(
            unlabeled_prediction_iter, unlabeled_class)

        labeled_loss.update(labeled_loss_iter.item(), labeled_image.size(0))
        labeled_accuracy.update(
            labeled_accuracy_first.item(), labeled_image.size(0))
        unlabeled_loss.update(unlabeled_loss_iter.item(),
                              unlabeled_image.size(0))
        unlabeled_accuracy.update(
            unlabeled_accuracy_first.item(), unlabeled_image.size(0))
        interp_losses.update(interpolare_loss_iter.item(),
                             labeled_image.size(0))

        log_info()
        best_test_accuracy = save_check_point(best_test_accuracy)
