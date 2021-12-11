import argparse
import torch
import random
import os
from os.path import exists
import torch.nn.functional as F
from torch.optim import SGD
from torch.distributions import Beta
import torchvision
import modified_vgg

from tensorboardX import SummaryWriter


from dataloader import dataloader
from utils import AverageMeter, logger_config, accuracy, save_checkpoint
from model import ConvLarge

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
writer = SummaryWriter(log_dir=result_path)

train_loader, test_loader = dataloader(
    dset=args.dataset,
    path=data_path,
    num_iters=args.iteration,
    return_unlabel=True,
    save_path=result_path
)

num_classes = 10
kwargs = {'num_classes': num_classes}



if args.model == "scnn":
    model = ConvLarge(num_classes=num_classes)
elif args.model == "vgg" and args.dataset == "lst10":
    model = torchvision.models.vgg11(pretrained=False, progress=True, **kwargs)
elif args.model == "vgg":
    model = modified_vgg.vgg11(pretrained=False, progress=True, **kwargs)
model.cuda()

optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9,
                weight_decay=1e-4)

beta_distribution = Beta(torch.tensor([1]), torch.tensor([1]))


def BatchSampler(data_loader):
    labeled_images, labeled_class, unlabeled_images, unlabeled_class = next(
        train_loader)
    labeled_class_matrix = F.one_hot(
        labeled_class, num_classes=num_classes).float()
    return labeled_images.cuda(), labeled_class.cuda(), labeled_class_matrix, unlabeled_images.cuda(), unlabeled_class.cuda()


def update_learning_rate_and_weight(iter):
    learning_rate = 0.1 if iter >= 4000 else 0.1 * iter / 4000

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    weight = 1 if iter > 4000 else iter / 4000
    return learning_rate, weight


def delta_theta_labeled(labeled_loss):
    return torch.autograd.grad(labeled_loss, model.parameters(), only_inputs=True)


@torch.no_grad()
def update_pseudo_label(unlabeled_image, dtli, learning_rate):
    unlabeled_prediction = model(unlabeled_image)
    unlabeled_pseudo_class_matrix = F.softmax(unlabeled_prediction, dim=1)
    epsilon = 1e-2 / torch.norm(torch.cat([x.view(-1) for x in dtli]))

    for i in range(len(dtli)):
        model.parameters()[i].data.add_(dtli[i], alpha=epsilon)
    unlabeled_prediction_1 = model(unlabeled_image)

    for i in range(len(dtli)):
        model.parameters()[i].data.add_(dtli[i], alpha=2.*epsilon)
    unlabeled_prediction_0 = model(unlabeled_image)

    for i in range(len(dtli)):
        model.parameters()[i].data.add_(dtli[i], alpha=epsilon)

    unlabeled_gradient = F.softmax(unlabeled_prediction_1, dim=1) - \
        F.softmax(unlabeled_prediction_0, dim=1)
    unlabeled_gradient.div_(epsilon)

    unlabeled_pseudo_class_matrix.sub_(unlabeled_gradient, alpha=learning_rate)
    torch.relu_(unlabeled_pseudo_class_matrix)
    sums = torch.sum(unlabeled_pseudo_class_matrix, dim=1, keepdim=True)
    unlabeled_pseudo_class_matrix /= torch.where(sums ==
                                                 0., torch.ones_like(sums), sums)
    return unlabeled_pseudo_class_matrix


@ torch.no_grad()
def improved_training_protocol(labeled_image, unlabeled_image, labeled_class_matrix, unlabeled_pseudo_class_matrix):
    lambda_i_class = beta_distribution.sample((100,)).cuda()
    lambda_i_image = lambda_i_class.view(-1, 1, 1, 1)
    interpolate_image = (labeled_image * lambda_i_image +
                         unlabeled_image * (1. - lambda_i_image)).detach()
    interpolate_pseudo_class = (labeled_class_matrix * lambda_i_class +
                                unlabeled_pseudo_class_matrix * (1. - lambda_i_class)).detach()
    return interpolate_image, interpolate_pseudo_class


def log_info(iter, learning_rate, labeled_loss, labeled_accuracy, unlabeled_loss, unlabeled_accuracy):
    if iter % 100 == 0:
        logger.info("{'Iteration': {0:05d}, 'Labeled': {'Loss': {labeled_loss.val:.3f}, 'Loss avg': {labeled_loss.avg:.3f}, 'Acc': {labeled_accuracy.val:.3f}, 'Acc avg': {labeled_accuracy.avg:.3f}}, 'Unlabeled': {'Loss': {unlabeled_loss.val:.3f}, 'Loss avg': {unlabeled_loss.avg:.3f}, 'Acc': {unlabeled_accuracy.val:.3f}, 'Acc avg': {unlabeled_accuracy.avg:.3f} \}\}"
                    "Learning Rate: {1:.4f}".format(iter, learning_rate, labeled_loss=labeled_loss,
                                                    labeled_accuracy=labeled_accuracy, unlabeled_loss=unlabeled_loss,  unlabeled_accuracy=unlabeled_accuracy))


@torch.no_grad()
def test_eval(test_loader, model):
    test_loss = AverageMeter()
    test_accuracy = AverageMeter()
    model.eval()
    for i, (test_image, test_class) in enumerate(test_loader):
        test_image = test_image.cuda()
        test_class = test_class.cuda()

        test_prediction = model(test_image)
        test_loss = F.cross_entropy(
            test_prediction, test_class, reduction='mean')

        test_accuracy_first, = accuracy(test_prediction, test_class, topk=(1,))
        test_loss.update(test_loss.item(), test_image.size(0))
        test_accuracy.update(test_accuracy_first.item(), test_image.size(0))

        if i % 400 == 0:
            logger.info("{'Test': {'Loss': {test_loss.val:.3f}, 'Loss avg': {test_loss.avg:.3f}, 'Acc': {test_accuracy.val:.3f}, 'Acc avg': {test_accuracy.avg:.3f}\}\}"
                        .format(test_loss=test_loss, test_accuracy=test_accuracy))
    return test_accuracy.avg


def save_check_point(iter, best_test_accuracy, labeled_loss, labeled_accuracy, unlabeled_loss, unlabeled_accuracy, interp_losses):
    if (iter + 1) % 400 == 0 or iter == args.iteration - 1:
        test_accuracy = test_eval(test_loader, model)
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
        logger.info("{'Best accuracy': %.5f}" % best_test_accuracy)
        save_checkpoint({
            'Iteration': iter + 1,
            'Model': model.state_dict(),
            'Best_test_accuracy': best_test_accuracy,
            'Optimizer': optimizer.state_dict()
        }, test_accuracy > best_test_accuracy, path=result_path, filename="checkpoint.pth")

        writer.add_scalar('train/label-acc', labeled_accuracy.avg, iter)
        writer.add_scalar('train/unlabel-acc', unlabeled_accuracy.avg, iter)
        writer.add_scalar('train/label-loss', labeled_loss.avg, iter)
        writer.add_scalar('train/unlabel-loss', unlabeled_loss.avg, iter)
        writer.add_scalar('train/lr', learning_rate, iter)
        writer.add_scalar('test/accuracy', test_accuracy, iter)
        writer.add_scalar('train/interp-loss', interp_losses.avg, iter)

        labeled_loss.reset()
        labeled_accuracy.reset()
        unlabeled_loss.reset()
        unlabeled_accuracy.reset()
        interp_losses.reset()
        return interp_losses

if __name__ == "__main__":
    best_test_accuracy = 0.
    labeled_loss = AverageMeter()
    labeled_accuracy = AverageMeter()
    unlabeled_loss = AverageMeter()
    unlabeled_accuracy = AverageMeter()
    interp_losses = AverageMeter()

    for iter in range(args.iteration):
        labeled_image, labeled_class, labeled_class_matrix, unlabeled_image, unlabeled_class = BatchSampler(
            train_loader)

        learning_rate, weight = update_learning_rate_and_weight(iter=iter)

        model.eval()

        labeled_prediction_iter = model(labeled_image)
        labeled_loss_iter = F.cross_entropy(
            labeled_prediction_iter, labeled_class, reduction='mean')
        delta_theta_labeled_implementation = delta_theta_labeled(
            labeled_loss_iter)

        unlabeled_pseudo_class_matrix = update_pseudo_label(
            unlabeled_image, delta_theta_labeled_implementation, learning_rate)

        model.train()

        interpolate_image, interpolate_pseudo_class = improved_training_protocol(
            labeled_image, unlabeled_image, labeled_class_matrix, unlabeled_pseudo_class_matrix)

        interpolare_prediction_iter = model(interpolate_image)
        interpolare_loss_iter = F.kl_div(F.log_softmax(
            interpolare_prediction_iter, dim=1), interpolate_pseudo_class, reduction='batchmean')

        unlabeled_prediction_iter = model(unlabeled_image)
        unlabeled_loss_iter = torch.norm(
            F.softmax(unlabeled_prediction_iter, dim=1)-unlabeled_pseudo_class_matrix, p=2, dim=1).pow(2).mean()
        overall_loss = interpolare_loss_iter + weight * unlabeled_loss_iter

        optimizer.zero_grad()
        overall_loss.backward()
        optimizer.step()

        labeled_accuracy_first, = accuracy(labeled_prediction_iter, labeled_class)
        unlabeled_accuracy_first, = accuracy(
            unlabeled_prediction_iter, unlabeled_class)

        labeled_loss.update(labeled_loss_iter.item(), labeled_image(0))
        labeled_accuracy.update(
            labeled_accuracy_first.item(), labeled_image.size(0))
        unlabeled_loss.update(unlabeled_loss_iter.item(),
                              unlabeled_image.size(0))
        unlabeled_accuracy.update(
            unlabeled_accuracy_first.item(), unlabeled_image.size(0))
        interp_losses.update(interpolare_loss_iter.item(),
                             labeled_image.size(0))

        log_info(iter, learning_rate, labeled_loss, labeled_accuracy,
                 unlabeled_loss, unlabeled_accuracy)

        best_test_accuracy = save_check_point(iter, best_test_accuracy, labeled_loss, labeled_accuracy, unlabeled_loss, unlabeled_accuracy, interp_losses)
        

    writer.close()
