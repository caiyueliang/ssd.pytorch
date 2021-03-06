# encoding:utf-8
from data import *
from utils.transform import TrainformTrain, TestformTest
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import numpy as np
import argparse
from data.datasets import SSDDataset

from tensorboardX import SummaryWriter

writer = SummaryWriter()


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='things', choices=['VOC', 'COCO', 'things'], type=str, help='VOC or COCO or things')
parser.add_argument('--dataset_root', default='/home/lijc08/deeplearning/Data/AI比赛/特定物品识别/images_train/', help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='Pretrained base model')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

parser.add_argument('--total_epochs', default=200, type=int, help='total_epochs')
parser.add_argument('--decay_epoch', default=60, type=int, help='decay_epoch')
parser.add_argument('--min_loss', default=5, type=float, help='min_loss')
parser.add_argument('--class_num', default=15, type=int, help='class_num')
parser.add_argument('--retrain', default=False, type=str2bool, help='retrain')
parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models')
parser.add_argument("--checkpoint_name", default="things_ssd.pth", type=str, help="")

args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root, transform=TrainformTrain(cfg['min_dim'], MEANS))
        ssd_net = build_ssd(phase='train', size=cfg['min_dim'], num_classes=cfg['num_classes'], cfg=cfg)

    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root, transform=TrainformTrain(cfg['min_dim'], MEANS))
        ssd_net = build_ssd(phase='train', size=cfg['min_dim'], num_classes=cfg['num_classes'], cfg=cfg)

    else:
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = things
        # cfg = voc
        print('[train] cfg', cfg)

        dataset = SSDDataset(root_path=os.path.join(args.dataset_root, 'train'),
                             image_file='image_path.txt',
                             img_size=cfg['min_dim'],
                             train=True,
                             transform=TrainformTrain(cfg['min_dim'], MEANS))
        test_dataset = SSDDataset(root_path=os.path.join(args.dataset_root, 'test'),
                                  image_file='image_path.txt',
                                  img_size=cfg['min_dim'],
                                  train=False,
                                  transform=TestformTest(cfg['min_dim'], MEANS))
        ssd_net = build_ssd(phase='train', size=cfg['min_dim'], num_classes=cfg['num_classes'], cfg=cfg)

    net = ssd_net

    file_name = os.path.join(args.save_folder, args.checkpoint_name)
    if os.path.exists(file_name) and not args.retrain:
        print('[load model] %s ...' % file_name)
        net.load_state_dict(torch.load(file_name))

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(num_classes=cfg['num_classes'],
                             overlap_thresh=0.5,
                             prior_for_matching=True,
                             bkg_label=0,
                             neg_mining=True,
                             neg_pos=3,
                             neg_overlap=0.5,
                             encode_target=False,
                             use_gpu=args.cuda)

    print('Loading the dataset...')
    print('Training SSD on:', dataset.name)
    print(args)

    # TODO
    # if args.dataset == 'things':
    #     data_loader = data.DataLoader(dataset, args.batch_size,
    #                                   num_workers=args.num_workers,
    #                                   shuffle=True)
    # else:
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    test_data_loader = data.DataLoader(test_dataset, args.batch_size,
                                       num_workers=args.num_workers,
                                       collate_fn=detection_collate,
                                       pin_memory=True)

    min_loss = args.min_loss

    for epoch in range(args.total_epochs):
        print("==================================================================")
        net.train()
        loc_loss = 0.0
        conf_loss = 0.0
        train_loss = 0.0
        loc_max = 0.0
        loc_min = 0.0
        loc_mean = 0.0
        conf_max = 0.0
        conf_min = 0.0
        conf_mean = 0.0

        if epoch >= args.decay_epoch and epoch % args.decay_epoch == 0:
            args.lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

        time_start = time.time()
        for batch_i, (images, targets) in enumerate(data_loader):
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]

            # forward
            out = net(images)

            # print(out[0].size(), out[1].size())
            for i in range(out[0].size()[0]):
                loc_mean += torch.mean(out[0][i]).item()
                loc_min += torch.min(out[0][i]).item()
                loc_max += torch.max(out[0][i]).item()
                conf_mean += torch.mean(out[1][i]).item()
                conf_min += torch.min(out[1][i]).item()
                conf_max += torch.max(out[1][i]).item()

            # back prop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            train_loss += loss_l.item() + loss_c.item()
            # print('[train_loss] ' + str(batch_i) + " " + str(train_loss))

        time_end = time.time()
        loc_loss /= len(data_loader)
        conf_loss /= len(data_loader)
        train_loss /= len(data_loader)
        print('[epoch] %d train Loss: %.4f, conf_loss: %.4f, loc_loss: %.4f, lr: %lf, time: %lf' %
              (epoch, train_loss, conf_loss, loc_loss, args.lr, time_end - time_start))

        loc_mean /= len(dataset)
        loc_min /= len(dataset)
        loc_max /= len(dataset)
        conf_mean /= len(dataset)
        conf_min /= len(dataset)
        conf_max /= len(dataset)
        print("loc_mean", loc_mean, "loc_min", loc_min, "loc_max", loc_max,
              "conf_mean", conf_mean, "conf_min", conf_min, "conf_max", conf_max)

        test_loss = test(net, criterion, test_data_loader, len(test_dataset), epoch)
        if test_loss < min_loss:
            min_loss = test_loss
            print('save best model')
            torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, args.checkpoint_name))

        writer.add_scalars('train/loc', {'mean': loc_mean, 'min': loc_min, 'max': loc_max}, epoch)
        writer.add_scalars('train/conf', {'mean': conf_mean, 'min': conf_min, 'max': conf_max}, epoch)
        writer.add_scalars('loss', {'train_loss': train_loss, 'test_loss': test_loss}, epoch)

    writer.close()


def test(model, criterion, test_data_loader, data_len, epoch):
        model.eval()
        loc_loss = 0.0
        conf_loss = 0.0
        test_loss = 0.0

        loc_max = 0.0
        loc_min = 0.0
        loc_mean = 0.0
        conf_max = 0.0
        conf_min = 0.0
        conf_mean = 0.0

        time_start = time.time()
        # 测试集
        for batch_i, (images, targets) in enumerate(test_data_loader):
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]
            # imgs = Variable(imgs.type(self.tensor))
            # targets = Variable(targets.type(self.tensor), requires_grad=False)

            out = model(images)
            loss_l, loss_c = criterion(out, targets)
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            test_loss += loss_l.item() + loss_c.item()
            # print('[test_loss] ' + str(batch_i) + " " + str(test_loss))

            for i in range(out[0].size()[0]):
                loc_mean += torch.mean(out[0][i]).item()
                loc_min += torch.min(out[0][i]).item()
                loc_max += torch.max(out[0][i]).item()
                conf_mean += torch.mean(out[1][i]).item()
                conf_min += torch.min(out[1][i]).item()
                conf_max += torch.max(out[1][i]).item()

        time_end = time.time()
        time_avg = float(time_end - time_start) / float(len(test_data_loader.dataset))

        loc_loss /= len(test_data_loader)
        conf_loss /= len(test_data_loader)
        test_loss /= len(test_data_loader)
        print('train Loss: %.4f, conf_loss: %.4f, loc_loss: %.4f, time_avg: %lf\n' %
              (test_loss, conf_loss, loc_loss, time_avg))

        loc_mean /= data_len
        loc_min /= data_len
        loc_max /= data_len
        conf_mean /= data_len
        conf_min /= data_len
        conf_max /= data_len
        print("loc_mean", loc_mean, "loc_min", loc_min, "loc_max", loc_max,
              "conf_mean", conf_mean, "conf_min", conf_min, "conf_max", conf_max)

        writer.add_scalars('test/loc', {'mean': loc_mean, 'min': loc_min, 'max': loc_max}, epoch)
        writer.add_scalars('test/conf', {'mean': conf_mean, 'min': conf_min, 'max': conf_max}, epoch)

        return test_loss


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    # nn.init.xavier_uniform(param)
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


# def create_vis_plot(_xlabel, _ylabel, _title, _legend):
#     return viz.line(
#         X=torch.zeros((1,)).cpu(),
#         Y=torch.zeros((1, 3)).cpu(),
#         opts=dict(
#             xlabel=_xlabel,
#             ylabel=_ylabel,
#             title=_title,
#             legend=_legend
#         )
#     )
#
#
# def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
#                     epoch_size=1):
#     viz.line(
#         X=torch.ones((1, 3)).cpu() * iteration,
#         Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
#         win=window1,
#         update=update_type
#     )
#     # initialize epoch plot on first iteration
#     if iteration == 0:
#         viz.line(
#             X=torch.zeros((1, 3)).cpu(),
#             Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
#             win=window2,
#             update=True
#         )


if __name__ == '__main__':
    train()
