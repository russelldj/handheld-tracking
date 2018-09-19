import sys
import pandas as pd
import numpy as np
import cv2
import ast
import logging

LINE_WIDTH = 2
VIDEO = "/home/drussel1/data/EPIC/videos/P01_01.MP4"

class Tracker(object):
    """there will only be a tracker initialized when a new instance of an object appears"""
    def __init__(self, bboxs, image):
        print(bboxs)
        exit()
        """make sure that the bbox is already in the opencv format"""
        #logging.info(bboxs)
        print("initializing")
        self.bbox = bbox
        trac = cv2.TrackerKCF_create()
        print(trac)
        print(image, bbox)
        trac.init(np.zeros([480, 640, 3]), bbox)
        #self.num_since_initialization 
        self.color = (int(np.random.random_integers(0, high=256)),\
                int(np.random.random_integers(0, high=256)),\
                int(np.random.random_integers(0, high=256)))

    def update(self, bbox, image): 
        print("updating")
        if len(self.bbox) == 0:
            self.tracker.release()
            self.tracker = cv2.TrackerKCF_create()
            self.tracker.init(image, bbox)
        self.bbox = bbox

    def increment(self, image):
        print('incrementing')
        self.tracked_bbox = tracker.update(image)

    def draw(self, image):
        logging.info(bbox)
        upper_left  = (self.tracked_bbox[0], self.tracked_bbox[1]) 
        lower_right = (self.tracked_bbox[0] + self.tracked_bbox[1], self.tracked_bbox[1] + self.tracked_bbox[3])
        cv2.rectangle(image, upper_left, lower_right, self.color, LINE_WIDTH)
        #cv2.imwrite('test.png', image)
        #exit()


ANNOTATION = '/home/drussel1/data/EPIC/Annotations/EPIC_train_object_labels.csv'

logging.info(sys.argv)
annots = pd.read_csv(ANNOTATION)
annots.sort_values(['video_id', 'frame'], axis=0, ascending=True, inplace=True)

#for index, row in annots.iterrows():
#    logging.info(row)
#    in_ = input('sadf')

i = 0

tracker_list = {}
video_ID = annots.loc[0, 'video_id'] 
last_frame = 0

cap = cv2.VideoCapture(VIDEO)

video_frame_num = 0 # it is possible that this should be 1

for index, row in annots.iterrows():
#for i in itertools.count():
    video_ID = row['video_id'] # looking ahead by one
    annotation_frame_num = int(row['frame'])
    noun  = row['noun']
    # we're going to do the parsing here
    bboxs  = ast.literal_eval(row['bounding_boxes'])
    if len(bboxs) > 0:
        bbox = bboxs[0] # take the first of the possibly many boxes
        bbox = (bbox[1], bbox[0], bbox[3], bbox[2]) # rearanging from ijhw to xywh
    else:
        bbox = ()
    #print("bboxs: {}, class: {}, video_frame_num: {}".format(bboxs, noun, video_frame_num))

    if video_ID != "P01_01":
        break
    while video_frame_num < annotation_frame_num + 30:
        #logging.info('{} {}'.format(video_frame_num, annotation_frame_num))
        ret, image = cap.read()
        output_image = image.copy()
        video_frame_num += 1
        for tracker in tracker_list.values():
            tracker.increment(image)
            tracker.draw(output_image) # you don't want it tracking on bounding box artifacts
        logging.info(video_frame_num)
        #if len(bboxs) > 0:
        #    cv2.rectangle(image, (bboxs[0][1], bboxs[0][0]), (bboxs[0][1] + bboxs[0][3], bboxs[0][0] + bboxs[0][2]), (0, 0, 0), LINE_WIDTH)
        #cv2.imwrite('outputs/annots/{:05d}.jpeg'.format(video_frame_num),image)

    if noun in tracker_list:
        tracker_list[noun].update(bbox, image)
    else:
        tracker_list[noun] = Tracker(bbox, image)
