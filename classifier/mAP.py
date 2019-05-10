import numpy as np
from tqdm import tqdm

def bb_intersection_over_union(boxA, boxB):
	'''
	from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	'''
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def _collect_data():
	predictions = {}
    with open(predictions_path, 'r') as p:
        lines = p.readlines()
        for line in lines:
            if line.startswith('GROUND TRUTH'):
                frame_name = line.split(':')[-1].strip()
                predictions[frame_name] = []
            else:
                if line.startswith('PREDICTIONS') \
                   or len(line) == 1:
                    continue
                elif line.startswith('label'):

                else:
                    line_els = line.split(' ')
                    score = float(line_els[4].split('(')[-1].split(')')[0])
                    x_min = int(line_els[5])
                    y_min = int(line_els[7])
                    x_max = int(line_els[9])
                    y_max = int(line_els[11])
                    predictions[frame_name].append(
                        [score, x_min, y_min, x_max, y_max])

def main():
	predictions = _data_to_predictions()

if __name__ == '__main__':
	########################################

	DATA_FILE = './ssd.pytorch/eval/okutama_512_115000/combined.txt'

	########################################
	main()