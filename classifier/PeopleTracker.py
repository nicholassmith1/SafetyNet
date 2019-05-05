import cv2 as cv
import numpy as np

from glob import glob
import os
from copy import copy
from tqdm import tqdm
import sys
import json

from pdb import set_trace as bp

class PeopleTracker(object):
    def __init__(self, frame_paths, predictions_path, export_as='list'):
        self.frame_paths = frame_paths
        self.predictions = self._get_predictions(predictions_path)
        self.cur_id = 0
        self.people_dict = {}
        self.export_as = export_as

    def _get_predictions(self, predictions_path):
        predictions = {}
        with open(predictions_path, 'r') as p:
            lines = p.readlines()
            for line in lines:
                if line.startswith('GROUND TRUTH'):
                    frame_name = line.split(':')[-1].strip()
                    predictions[frame_name] = []
                else:
                    if line.startswith('label') or line.startswith('PREDICTIONS') \
                       or len(line) == 1:
                        continue
                    else:
                        line_els = line.split(' ')
                        score = float(line_els[4].split('(')[-1].split(')')[0])
                        x_min = int(line_els[5])
                        y_min = int(line_els[7])
                        x_max = int(line_els[9])
                        y_max = int(line_els[11])
                        predictions[frame_name].append(
                            [score, x_min, y_min, x_max, y_max])

        print('collected predictions')

        return predictions

    def track(self):
        for f_p in self.frame_paths:
            f_name = f_p.split('/')[-1].split('.png')[0]
            bbox_predictions = self.predictions[f_name]
            self._analyze_frame(bbox_predictions, f_name)

        real_people = [
            p for p in self.people_dict.values()
            if p.confirmed_real
        ]

        if self.export_as == 'viz':
            return self._export_as_viz(real_people)
        elif self.export_as == 'json':
            return self._export_as_json(real_people)
        else:
            return real_people

    def _analyze_frame(self, bbox_predictions, f_name):
        ids_in_frame = []
        people_maybe_in_frame = [
            p for p in self.people_dict.values()
            if not p.confirmed_exited and not p.confirmed_fake
        ]

        # match bboxs to people
        for pred in bbox_predictions:
            person_proposal = Person(self.cur_id, pred + [f_name])

            best_match_id = None
            match_scores = [
                self._score_match(pmif, person_proposal)
                for pmif in people_maybe_in_frame
                if pmif.id not in ids_in_frame
            ]
            # either associate bbox with an existing person
            if any([ms[0] > MATCH_THRESH for ms in match_scores]):
                best_match_id = max(match_scores, key=lambda ms: ms[0])[1]
                best_match = self.people_dict[best_match_id]
                best_match.prev_locations.append(
                    person_proposal.prev_locations[0])
                best_match.prev_locations_bbox.append(
                    person_proposal.prev_locations_bbox[0])
                best_match.confidence_scores.append(
                    person_proposal.confidence_scores[0])

                if len(best_match.confidence_scores) >= MIN_PRESENT:
                    setattr(best_match, 'confirmed_real', True)

                ids_in_frame.append(best_match_id)

            # or add new person to dictionary
            else:
                self.people_dict[self.cur_id] = person_proposal
                ids_in_frame.append(self.cur_id)

            self.cur_id += 1

        # update people dict for all people not in frame
        for person in people_maybe_in_frame:
            if person.id in ids_in_frame:
                continue
            else:
                self._update_person_info(person, f_name)

    def _score_match(self, pm, prop):
        pm_x, pm_y = pm.prev_locations[-1]
        if pm_x == -1:
            for i in range(2, MAX_MISSING + 1):
                pm_x, pm_y = pm.prev_locations[-i]
                if pm_x != -1:
                    break

        prop_x, prop_y = prop.prev_locations[-1]
        distance = ( (pm_x - prop_x)**2 + (pm_y - prop_y)**2 )**0.5

        ratio = distance / (720**2 + 1278**2)**0.5
        score = 1 - ratio

        return (score, pm.id)

    def _update_person_info(self, person, f_name):
        person.prev_locations.append([-1, -1])
        person.prev_locations_bbox.append(
            [-1, -1, -1, -1, f_name])
        person.confidence_scores.append(-1)
        if len(person.prev_locations) < MAX_MISSING:
            return
        else:
            num_missing = -sum(
                [l[0] for l in person.prev_locations[-MAX_MISSING:]])
            # person is fake
            if num_missing >= MAX_MISSING and not person.confirmed_real:
                setattr(person, 'confirmed_fake', True)
            # person is real and exited
            elif num_missing >= MAX_MISSING and person.confirmed_real:
                setattr(person, 'confirmed_exited', True)
            # below num missing threshold, their fate will be decided later ;)
            else:
                return

    def _export_as_json(self, people):
        data = {}
        data['people'] = []
        for person in people:
            person_data = {}
            person_data['id'] = str(person.id)
            pos = []
            for p in person.prev_locations_bbox:
                p_keys = ('frame_id', 'box')
                p_vals = (p[-1], p[:-1])
                pos.append(dict(zip(p_keys, p_vals)))
            person_data['pos'] = pos
            data['people'].append(person_data)
        return json.dumps(data)

    def _export_as_viz(self, people):
        # [p_id, frame_id, xmin, ymin, xmax, ymax]
        data = []
        for p in people:
            for loc in p.prev_locations_bbox:
                if loc[0] > -1:
                    data.append([
                        p.id, int(loc[-1].split('_')[-1]), 
                        loc[0], loc[1], loc[2], loc[3]
                    ])
        data = np.asarray(data, dtype=np.float32)
        # to 4k res
        data *= 3
        return data

class Person(object):
    def __init__(self, _id, pred):
        self.id = _id
        self.confirmed_real = False

        midpoint = ( (pred[1] + pred[3]) / 2, (pred[2] + pred[4]) / 2 ) 
        self.prev_locations = [midpoint] # x, y
        self.confidence_scores = [pred[0]]

        self.confirmed_exited = False
        self.confirmed_fake = False

        self.prev_locations_bbox = [pred[1:]]

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

if __name__ == '__main__':
    ###############################################################

    frames_dir = './ssd.pytorch/eval/okutama_512_115000'
    MATCH_THRESH = 0.9
    MAX_MISSING = 5
    MIN_PRESENT = 4
    viz = False

    ###############################################################

    frame_paths = glob(os.path.join(frames_dir, 'combined', '*.png'))
    frame_paths_sorted = sorted(
        frame_paths, key=lambda p: int(p.split('/')[-1].split('_')[-1].split('.')[0]))
    predictions_path = os.path.join(frames_dir, 'combined.txt')
    PT = PeopleTracker(frame_paths_sorted, predictions_path, export_as='viz')
    people = PT.track()

    if viz:
        n = len(people)
        with TqdmUpTo(unit='person', unit_scale=True, total=n, file=sys.stdout) as t:
            for idx, person in enumerate(people):
                t.update_to(idx)
                locations = person.prev_locations_bbox
                for l in locations:
                    img_path = os.path.join('viz', l[-1] + '.png')
                    img = cv.imread(img_path)
                    cv.rectangle(
                        img, 
                        (l[0], l[1]), (l[2], l[3]), 
                        (255, 0, 0), 2)
                    cv.putText(
                        img, str(person.id), (l[0] - 5, l[1] - 5), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, color=(0, 0, 255), thickness=2)
                    cv.imwrite(img_path, img)
            t.update_to(n)
