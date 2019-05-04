from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform
from data import OKUTAMA_ROOT, OKUTAMA_CLASSES, \
                 OkutamaAnnotationTransform, OkutamaDetection, BaseTransform
import torch.utils.data as data
from ssd import build_ssd
import cv2 as cv
from data.config import voc as voc_cfg, coco as coco_cfg, okutama_300_cfg, okutama_512_cfg
import numpy as np

from pdb import set_trace as bp

def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = os.path.join(save_folder, 'test1.txt')
    combined_filename = os.path.join(save_folder, 'combined.txt')
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data

        i_mod = i % 6
        if i_mod == 0:
            combined_img = np.zeros((
                img.shape[0] * 2,
                img.shape[1] * 3,
                3), dtype=np.float32)
            combined_img_annos = []

        start_w = img.shape[1] * (i_mod % 3)
        start_h = 0 if i_mod < 3 else img.shape[0]

        if args.data_root == 'okutama':
            img_name = '_'.join(img_id.split('/')[-2:])
            combined_img_name = '_'.join(img_name.split('_')[:2])
        elif args.data_root == 'voc':
            img_name = img_id + '.jpg'

        with open(filename, mode='a+') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
                if box[0] > 0:
                    cv.rectangle(
                        img, 
                        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                        (0, 0, 255), 2)
        with open(combined_filename, mode='a+') as cf:
            if i_mod == 0:
                cf.write('\nGROUND TRUTH FOR: '+combined_img_name+'\n')
            for box in annotation:
                _box = [
                    int(box[0]) + start_w, int(box[1]) + start_h, 
                    int(box[2]) + start_w, int(box[3]) + start_h
                ]
                if box[0] > 0:
                    cf.write('label: '+' || '.join(str(b) for b in _box)+'\n')

        # scale each detection back up to the image
        h_scale = 1.15
        w_scale = 0.85
        scale = torch.Tensor([
            img.shape[1] * w_scale, img.shape[0] * h_scale,
            img.shape[1] * w_scale, img.shape[0] * h_scale])

        with open(combined_filename, 'a+') as cf:
            if i_mod == 0:
                cf.write('PREDICTIONS: '+'\n')
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                if pred_num == 0:
                    with open(filename, mode='a+') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                with open(combined_filename, 'a+') as cf:
                    _coords = [
                        int(pt[0]) + start_w, int(pt[1]) + start_h,
                        int(pt[2]) + start_w, int(pt[3]) + start_h
                    ]
                    cf.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in _coords) + '\n')

                cv.rectangle(
                    img, 
                    (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), 
                    (0, 255, 0), 2)

                j += 1
        
        combined_img[
            start_h : start_h + img.shape[0],
            start_w : start_w + img.shape[1],
            :] = img

        if i_mod == 5:
            combined_save_folder = os.path.join(save_path, 'combined')
            if not os.path.exists(combined_save_folder):
                os.mkdir(combined_save_folder)
            cv.imwrite(os.path.join(combined_save_folder, combined_img_name + '.png'), combined_img)

def test_voc(cfg):
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', cfg['min_dim'], cfg) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection(
        data_root, [('2012', 'person_val')], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(save_path, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

def test_okutama(cfg):
    num_classes = len(OKUTAMA_CLASSES) + 1 # +1 background
    net = build_ssd('test', cfg['min_dim'], cfg) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    print('Finished loading model!')

    testset = OkutamaDetection(
        data_root, None, OkutamaAnnotationTransform(), train=False)

    test_net(
        save_path, net, args.cuda, testset,
        BaseTransform(net.size, (104, 117, 123)),
        thresh=args.visual_threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
    parser.add_argument('--trained_model',  required=True,  #default='weights/ssd_300_VOC0712.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='eval/', type=str,
                        help='Dir to save results')
    parser.add_argument('--visual_threshold', default=0.6, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use cuda to train model')
    parser.add_argument('--data_root', default='okutama', #default='voc', 
                        help='Location of root directory')
    parser.add_argument('--min_dim', default=300, type=int,
                    help='Defines the size of the SSD network.')
    parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
    parser.add_argument('--name', default='test', help='Experiment name.')
    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    save_path = os.path.join(args.save_folder, args.name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if args.data_root == 'okutama':
        data_root = OKUTAMA_ROOT
        labelmap = OKUTAMA_CLASSES
        if args.min_dim == 300:
            test_okutama(okutama_300_cfg)
        elif args.min_dim == 512:
            test_okutama(okutama_512_cfg)
    elif args.data_root == 'voc':
        data_root = VOC_ROOT
        labelmap = VOC_CLASSES
        test_voc(voc_cfg)
