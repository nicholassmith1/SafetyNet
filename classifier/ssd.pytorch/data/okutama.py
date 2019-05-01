"""
Okutama-Action Dataset
Author: Michael Snower
"""

import os.path as osp
import os
import sys
import torch
import torch.utils.data as data
import cv2 as cv
import numpy as np
from glob import glob

from pdb import set_trace as bp

OKUTAMA_CLASSES = ('pedestrian') # always index 0 

OKUTAMA_ROOT = '/data/jhtlab/msnower/okutama_dataset/Train-Set/'


class OkutamaAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        # divide by extra scale lables are in 4k and images are lower res
        extra_scale = 3840. / width

        res = []
        scale = np.array([width, height, width, height])
        for t in target:
            bbox = np.asarray(t[:4], np.int32)
            bbox = bbox / extra_scale
            scaled = list(bbox / scale)
            scaled.append(t[4])
            res.append(scaled)

        return res # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class OkutamaDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(
        self, dataset_root, transform=None,
        target_transform=OkutamaAnnotationTransform(), name='okutama'):

        self.dataset_root = dataset_root
        self.name = name
        self.imgs_root = dataset_root
        self.labels_root = osp.join(dataset_root, 'Labels', 'MultiActionLabels', '3840x2160')

        print('creating dataset ids')

        self.ids = list()
        for d_dir in os.listdir(self.imgs_root):
            if d_dir == 'Labels':
                continue
            cur_d_dir = os.path.join(self.imgs_root, d_dir)
            for time_dir in os.listdir(cur_d_dir):
                cur_time_dir = os.path.join(cur_d_dir, time_dir, 'Extracted-Frames-1280x720')
                for vid_dir in os.listdir(cur_time_dir):
                    cur_vid_dir = os.path.join(cur_time_dir, vid_dir)
                    frame_paths = glob(os.path.join(cur_vid_dir, '*.jpg'))

                    def _get_frame_labels(vd):
                        p_ = [lp for lp in os.listdir(self.labels_root) \
                              if '.'.join(lp.split('.')[:3]) == vd][0]
                        p = os.path.join(self.labels_root, p_)

                        lines = [l.strip().split(' ')[:9] for l in open(p)]
                        lines = np.asarray(lines, dtype=np.int32)
                        num_frames = np.max(lines[:, 5])
                        fls = [[] for _ in range(num_frames + 1)]
                        for line in lines:
                            frame_num = line[5]
                            # ignore annotations where the human is lost
                            if line[7] == 1 or line[8] == 1:
                                continue
                            else:
                                # get bbox coords and add 0 for label
                                fls[frame_num].append(line.tolist()[1:5] + [0])
                        return fls

                    frame_labels = _get_frame_labels(vid_dir)
                    frame_paths.sort(key=lambda p: int(p.split('/')[-1].split('.')[0]))
                    
                    # remove extraneous frames at the end of the dataset with no people
                    if len(frame_labels) < len(frame_paths):
                        frame_paths = frame_paths[:len(frame_labels)]

                    for (f_p, f_l) in zip(frame_paths, frame_labels):
                        # skip frames with no people (and thus have no label)
                        f_l_ = np.asarray(f_l)
                        if len(f_l_.shape) < 2:
                            continue

                        self.ids.append((f_p, f_l))

        print('created dataset with {} ids'.format(len(self.ids)))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        im, gt, h, w, img_orig, target_orig = self.pull_item(index)

        return im, gt, img_orig, target_orig

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        f_p, f_l = self.ids[index]

        min_dim = 300
        aspect_ratio = 16 / 9.

        target = f_l
        img = cv.imread(f_p)
        img = cv.resize(img, (int(aspect_ratio * min_dim), min_dim))
        height, width, channels = img.shape

        img_orig = img.copy()
        target_orig = target[:]

        if self.target_transform is not None:
            target = self.target_transform(target, height, width)

        if self.transform is not None:
            target = np.asarray(target, dtype=np.float32)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

            img = img[:, :, (2, 1, 0)] # to rgb

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, img_orig, target_orig

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        f_p, _ = self.ids[index]
        return cv2.imread(f_p)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        _, f_l = self.ids[index]
        gt = self.target_transform(f_l, 1, 1) # CHNAGE THIS!!!!!!
        return str(index), gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
