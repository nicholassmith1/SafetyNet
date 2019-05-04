import cv2 as cv
import numpy as np
from math import floor

from glob import glob
import os
import os.path as osp

from pdb import set_trace as bp

def save_framesplit_and_els(fs, f_els, cur_vid_dir, split_index):
    fs_save_path_dir = _join_paths([SAVE_DATA_ROOT] + cur_vid_dir.split('/')[-4:])
    fs_save_path = osp.join(fs_save_path_dir, split_index + '.jpg')
    cv.imwrite(fs_save_path, fs)

    # get vid dir number
    vid_dir = cur_vid_dir.split('/')[-1]
    p_ = [lp for lp in os.listdir(LABELS_ROOT) \
          if '.'.join(lp.split('.')[:3]) == vid_dir][0]
    label_save_path = osp.join(SAVE_LABELS_ROOT, p_)
    with open(label_save_path, 'a+') as file:
        for f_el in f_els:
            x_min = str(f_el[0])
            y_min = str(f_el[1])
            x_max = str(f_el[2])
            y_max = str(f_el[3])
            to_write = x_min + ' ' + y_min + ' ' + x_max + ' ' + y_max + ' ' + split_index + '\n'
            file.write(to_write)
        if len(f_els) == 0:
            to_write = '-1' + ' ' + '-1' + ' ' + '-1' + ' ' + '-1' + ' ' + split_index + '\n'
            file.write(to_write)

def create_framesplits(frame_path, frame_label, cur_vid_dir, index):
    frame = cv.imread(frame_path)
    height, width, channels = frame.shape
    
    s_h = floor(height / HEIGHT_SPLITS)
    s_w = floor(width / WIDTH_SPLITS)

    for i in range(HEIGHT_SPLITS):
        for j in range(WIDTH_SPLITS):
            f_x_min = j * s_w
            f_x_max = (j+1) * s_w
            f_y_min = i * s_h
            f_y_max = (i+1) * s_h

            f_split = frame[f_y_min : f_y_max, f_x_min : f_x_max, :]

            f_els = []
            for el in frame_label:
                el_x_min = el[0]
                el_y_min = el[1]
                el_x_max = el[2]
                el_y_max = el[3]

                cond_x_min = f_x_min <= el_x_min <= f_x_max
                cond_x_max = f_x_min <= el_x_max <= f_x_max
                cond_y_min = f_y_min <= el_y_min <= f_y_max
                cond_y_max = f_y_min <= el_y_max <= f_y_max

                if cond_x_min and cond_x_max and cond_y_min and cond_y_max:
                    # convert coordinates and append
                    el_x_min = el_x_min - f_x_min
                    el_y_min = el_y_min - f_y_min
                    el_x_max = el_x_max - f_x_min
                    el_y_max = el_y_max - f_y_min
                    f_els.append([el_x_min, el_y_min, el_x_max, el_y_max])
                
            if len(f_els) > 0 or INCLUDE_ALL_SPLITS:
                split_index = str(index) + '_' + str(i) + '_' + str(j)
                save_framesplit_and_els(f_split, f_els, cur_vid_dir, split_index)

def main():
    for d_dir in os.listdir(DATA_ROOT):
        if d_dir == 'Labels':
            continue
        cur_d_dir = os.path.join(DATA_ROOT, d_dir)
        for time_dir in os.listdir(cur_d_dir):
            cur_time_dir = os.path.join(cur_d_dir, time_dir, 'Extracted-Frames-1280x720')

            for vid_dir in os.listdir(cur_time_dir):
                cur_vid_dir = os.path.join(cur_time_dir, vid_dir)
                frame_paths = glob(os.path.join(cur_vid_dir, '*.jpg'))
                frame_labels = _get_frame_labels(vid_dir)
                frame_paths.sort(key=lambda p: int(p.split('/')[-1].split('.')[0]))

                for frame_index, (f_p, f_l) in enumerate(zip(frame_paths, frame_labels)):
                    # skip frames with no people (and thus have no label)
                    f_l_ = np.asarray(f_l, dtype=np.int32)
                    if len(f_l_.shape) >= 2:
                        f_l_ = np.round_(f_l_ / 3) # divide by 3 because frames are 1080p and labels are 4k
                        create_framesplits(
                            f_p, f_l_.tolist(), cur_vid_dir, frame_index)

                    if frame_index == (min(len(frame_paths), len(frame_labels)) - 1):
                        print('completed {}'.format(cur_vid_dir))

def _get_frame_labels(vd):
    p_ = [lp for lp in os.listdir(LABELS_ROOT) \
          if '.'.join(lp.split('.')[:3]) == vd][0]
    p = os.path.join(LABELS_ROOT, p_)

    lines = [l.strip().split(' ')[:9] for l in open(p)]
    lines = np.asarray(lines, dtype=np.int32)
    num_frames = np.max(lines[:, 5])
    fls = [[] for _ in range(num_frames + 1)]
    for line in lines:
        frame_num = line[5]
        # ignore annotations where the human is lost
        if line[6] == 1:
            continue
        if line[7] == 1 and IGNORE_OCCLUDED:
            continue
        if line[8] == 1 and IGNORE_GENERATED:
            continue
        else:
            # get bbox coords and add 1 for label
            fls[frame_num].append(line.tolist()[1:5] + [1])
    return fls

def _join_paths(paths):
    cur_path = paths[0]
    for i in range(1, len(paths)):
        cur_path = osp.join(cur_path, paths[i])
        if not osp.exists(cur_path):
            os.mkdir(cur_path)
    return cur_path

if __name__ == '__main__':
    #### Runtime arguments ####

    train = True
    _base_path = '/data/jhtlab/msnower/okutama_dataset/'
    _base_save_path = '/data/jhtlab/msnower/okutama_all_splits/'
    HEIGHT_SPLITS = 2
    WIDTH_SPLITS = 3
    IGNORE_OCCLUDED = True
    IGNORE_GENERATED = True
    INCLUDE_ALL_SPLITS = True

    ###########################

    if not osp.exists(_base_save_path):
        os.mkdir(_base_save_path)

    if train:
        base_path = osp.join(_base_path, 'Train-Set')
        base_save_path = osp.join(_base_save_path, 'Train-Set')
    else:
        base_path = osp.join(_base_path, 'Test-Set')
        base_save_path = osp.join(_base_save_path, 'Test-Set')

    if not osp.exists(base_save_path):
        os.mkdir(base_save_path)

    DATA_ROOT = base_path
    LABELS_ROOT = osp.join(base_path, 'Labels', 'MultiActionLabels', '3840x2160')

    SAVE_DATA_ROOT = base_save_path
    SAVE_LABELS_ROOT = _join_paths([base_save_path, 'Labels', 'MultiActionLabels', '3840x2160'])

    main()
