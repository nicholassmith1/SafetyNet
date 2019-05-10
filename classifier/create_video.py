import cv2 as cv
import numpy as np

from glob import glob
import os
from tqdm import tqdm
import sys

from pdb import set_trace as bp

if __name__ == '__main__':
    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    FPS = 31
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('viz_final.mp4', fourcc, FPS, (3840, 2160))

    frame_paths = glob(os.path.join('viz', '*.png'))
    frame_paths.sort(key=lambda p: int(p.split('/')[-1].split('.')[0]))

    frame_paths2 = glob(os.path.join('viz2', '*.png'))
    frame_paths2.sort(key=lambda p: int(p.split('/')[-1].split('.')[0]))

    n = len(frame_paths)
    with TqdmUpTo(unit='frame', unit_scale=True, total=n, file=sys.stdout) as t:
        for i, (f_p, f_p2) in enumerate(zip(frame_paths, frame_paths2)):
            t.update_to(i)

            if i < 400:
                frame = cv.imread(f_p)
            else:
                frame = cv.imread(f_p2)
            out.write(frame)

        t.update_to(n)

    out.release()