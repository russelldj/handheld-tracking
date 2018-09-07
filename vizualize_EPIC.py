import sys
import pandas as pd
import numpy as np
import cv2

LINE_WIDTH = 2

class Annotation(object):
    def __init__(self, bbox):
        self.bboxs = bbox
        self.num_since_update = 0
        self.color = (np.random.random_integers(0, high=256),\
                np.random.random_integers(0, high=256),\
                np.random.random_integers(0, high=256))

    def update(self, bboxs): 
        self.bboxs = bboxs

    def increment(self):
        self.num_since_update += 1
        if self.num_since_update > 30:
            self.bboxs = []


    def draw(self, image):
        for bbox in self.bboxs:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.color, LINE_WIDTH)


ANNOTATION = '/home/drussel1/data/EPIC/Annotations/EPIC_train_object_labels.csv'

print(sys.argv)
annots = pd.read_csv(ANNOTATION)
annots.sort_values(['video_id', 'frame'], axis=0, ascending=True, inplace=True)

i = 0

annots_list = {}
video_ID = annots.loc[0, 'video_id'] 
last_frame = 0
for index, row in annots.iterrows():
    i += 1 
    video_ID = row['video_id'] # looking ahead by one
    frame = int(row['frame'])
    if last_frame > frame:
        print('error')
        #exit()
    last_frame = frame
    noun  = row['noun']
    bbox  = row['bounding_boxes']
    print("{}, {}".format(frame, video_ID))
    if noun not in annots_list:
        annots_list[noun] = frame
        in_ = input('press')
    if video_ID != "P01_01":
        break
