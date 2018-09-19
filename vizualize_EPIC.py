import sys
import pandas as pd
import numpy as np
import cv2
import ast
import logging

LINE_WIDTH = 2
VIDEO_FILE = "/home/drussel1/data/EPIC/videos/P01_01.MP4"

class Annotation(object):
    def __init__(self, bboxs):
        #logging.info(bboxs)
        self.bboxs = ast.literal_eval(bboxs)
        self.num_since_update = 0
        self.color = (int(np.random.random_integers(0, high=256)),\
                int(np.random.random_integers(0, high=256)),\
                int(np.random.random_integers(0, high=256)))

    def update(self, bboxs): 
        self.bboxs = bboxs
        self.num_since_update = 0

    def increment(self):
        self.num_since_update += 1
        if self.num_since_update > 0:
            self.bboxs = []

    def draw(self, image):
        for bbox in self.bboxs:
            logging.info(bbox)
            cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[1] + bbox[3], bbox[0] + bbox[2]), self.color, LINE_WIDTH)
        self.bboxs = []


ANNOTATION = '/home/drussel1/data/EPIC/Annotations/EPIC_train_object_labels.csv'

logging.info(sys.argv)
annots = pd.read_csv(ANNOTATION)
annots.sort_values(['video_id', 'frame'], axis=0, ascending=True, inplace=True)

#for index, row in annots.iterrows():
#    logging.info(row)
#    in_ = input('sadf')

i = 0

annots_list = {}
video_ID = annots.loc[0, 'video_id'] 
last_frame = 0


video_frame_num = 0 # it is possible that this should be 1
obj_list = []
id_list  = [] 
is_first = True
last_frame_id = annots['frame'].iloc[0]
current_frame_id = last_frame_id.copy()
print(annots)
in_ = raw_input('asdf')
cap = cv2.VideoCapture(VIDEO_FILE)
print(cap.isOpened())
input_ = raw_input('cap')
video_frame_id = 0

#NEW
for index, row in annots.iterrows():
    video_ID = row['video_id'] # looking ahead by one
    if video_ID != "P01_01":
        break
    noun  = row['noun']
    bboxs  = ast.literal_eval(row['bounding_boxes'])
    print('bbox is {}'.format(bboxs))
    #this is to make sure it isn't the trivial case
    if bboxs != []:
        current_frame_id = int(row['frame'])
        #in_ = raw_input('bboxs')
        #print("bboxs: {}, class: {}, video_frame_num: {}".format(bboxs, noun, current_frame_id))

        if last_frame_id != current_frame_id: # this means you've collected a full batch of annots from last frame, but have not yet added the new one to the list
             # increment the video reader to the frame id of the last full set you read
            cap.set(1,last_frame_id)
            # get the image and the success value
            ret, val = cap.read()
            is_first = True
            #print("id list is {}, last_frame_id is {}".format(id_list, last_frame_id))
            print('obj_list is {}'.format(obj_list))
            print('last_frame_id {}'.format(last_frame_id))
            in_ = raw_input('obj_list')

            for bbox_ in obj_list:
                val = cv2.rectangle(val, (bbox_[1], bbox_[0]), (bbox_[1] + bbox_[3], bbox_[0] + bbox_[2]), (255,255,0), 3)
                print("bbox_ is{}".format(bbox_))
                last_frame_id = current_frame_id
            cv2.imwrite('visualized_annots/{:06d}.jpeg'.format(last_frame_id), val)
        
        if is_first:
            is_first = False
            id_list  = [current_frame_id]
            obj_list = []
            obj_list = bboxs # they are actually a list
        else:
            id_list.append(current_frame_id)
            obj_list = obj_list + bboxs

#OLD

#for index, row in annots.iterrows():
##for i in itertools.count():
#    video_ID = row['video_id'] # looking ahead by one
#    annotation_frame_num = int(row['frame'])
#    noun  = row['noun']
#    bboxs  = ast.literal_eval(row['bounding_boxes'])
#    print("bboxs: {}, class: {}, video_frame_num: {}".format(bboxs, noun, video_frame_num))
#    #logging.info("{}, {}".format(annotation_frame_num, bboxs))
#    #if noun not in annots_list:
#    #    annots_list[noun] = Annotation(bboxs)
#    #    #in_ = input('press')
#    if video_ID != "P01_01":
#        break
#
#    while video_frame_num < annotation_frame_num:
#        #logging.info('{} {}'.format(video_frame_num, annotation_frame_num))
#        ret, image = cap.read()
#        video_frame_num += 1
#        for annot in annots_list.values():
#            annot.increment()
#
#        if len(bboxs) > 0:
#            cv2.rectangle(image, (bboxs[0][1], bboxs[0][0]), (bboxs[0][1] + bboxs[0][3], bboxs[0][0] + bboxs[0][2]), (255, 255, 0), LINE_WIDTH)
#        cv2.imwrite('outputs/annots/{:05d}.jpeg'.format(video_frame_num),image)
