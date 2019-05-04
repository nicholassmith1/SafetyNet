from data import *
from utils.augmentations import SSDAugmentation
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
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

import cv2 as cv

from pdb import set_trace as bp

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', choices=['VOC', 'COCO', 'okutama'], default='okutama', #default='VOC', 
                    type=str, help='VOC or COCO or okutama')
parser.add_argument('--dataset_root', default=OKUTAMA_ROOT, #default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=6, type=int,
                    help='Batch size for training')
parser.add_argument('--min_dim', default=300, type=int,
                    help='Defines the size of the SSD network.')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--start_epoch', default=1, type=int,
                    help='Resume training at this epoch.')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--name', type=str, default='test_okutama', help='Name of the experiment.')
parser.add_argument('--no_transform', action='store_true', help='Whether to transform images.')
parser.add_argument('--log_every', type=int, default=10, help='frequency to log')
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

# where to save loss logs
if not os.path.exists('./logs'):
    os.mkdir('./logs')
log_path = os.path.join('logs', args.name + '.txt')


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        transform = SSDAugmentation(cfg['min_dim'], MEANS) if not args.no_transform else None
        dataset = VOCDetection(root=args.dataset_root, transform=transform)

    elif args.dataset == 'okutama':
        if args.dataset_root != OKUTAMA_ROOT:
            parser.error('Please specify Okutama dataset root.')
        if args.min_dim == 300:
            cfg = okutama_300_cfg
        elif args.min_dim == 512:
            cfg = okutama_512_cfg
        dataset = OkutamaDetection(
            dataset_root=args.dataset_root,
            transform=SSDAugmentation(cfg['min_dim'], MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg)
    net = ssd_net

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
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, cfg, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = args.start_epoch
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    with open(log_path, 'a+') as l:
        l.write('~~~~~~~~~~~~~~~~~~~~ Epoch {} ~~~~~~~~~~~~~~~~~~~~ \n'.format(epoch))

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1

            with open(log_path) as l:
                past_losses = l.readlines()
                num_lines = epoch_size // args.log_every
                epoch_lines = past_losses[-num_lines:]
                epoch_losses = [float(line.split('||')[-1].split(':')[-1].strip()) \
                                for line in epoch_lines \
                                if not line.startswith('~')]
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)

            with open(log_path, 'a+') as l:
                l.write('~ Avg epoch loss: {:4f} \n'.format(avg_epoch_loss))
                l.write('~~~~~~~~~~~~~~~~~~~~ Epoch {} ~~~~~~~~~~~~~~~~~~~~ \n'.format(epoch))

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets, img_orig, targ_orig = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets, img_orig, targ_orig = next(batch_iterator)

        # print('new batch')
        # for q, (img, trg) in enumerate(zip(img_orig, targ_orig)):
        #     print(len(trg))
        #     for t in trg:
        #         pts = (t[0], t[1]), (t[2], t[3])
        #         tl, br = [(int(pt[0] / 1), int(pt[1] / 1)) \
        #                    for pt in pts] # 7.2 for 300, 4.2188,
        #         cv.rectangle(img, tl, br, (0, 0, 255), 2)
        #     cv.imwrite(os.path.join('test_imgs', str(q) + '.jpg'), img)
        # exit(0)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % args.log_every == 0:
            time_elapsed = t1 - t0
            print('iter: {} || '.format(iteration) +
                    'timer: {:4f} || '.format(time_elapsed) +
                    'loss_l: {:4f} || '.format(loss_l) + 
                    'loss_c: {:4f} || '.format(loss_c) + 
                    'total loss: {:4f} \n'.format(loss))

            with open(log_path, 'a+') as l:
                l.write(
                    'iter: {} || '.format(iteration) +
                    'timer: {:4f} || '.format(time_elapsed) +
                    'loss_l: {:4f} || '.format(loss_l) + 
                    'loss_c: {:4f} || '.format(loss_c) + 
                    'total loss: {:4f} \n'.format(loss))

        if args.visdom:
            update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/{}_'.format(args.name) +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


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
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
