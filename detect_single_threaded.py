from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import json
import time
import numpy as np
import os
from scipy.optimize import linear_sum_assignment

detection_graph, sess = detector_utils.load_inference_graph()
def bb_intersection_over_union(boxA, boxB, is_xywh=False):
    """it appears this takes in bbs in the form x1, y1, x2, y2"""
    # determine the (x, y)-coordinates of the intersection rectangle
    if is_xywh:
        boxA = xywh_to_xyxy(boxA)
        boxB = xywh_to_xyxy(boxB)
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


def corner_distances(boxA, boxB):
   "this might be a weird metric, but I think it is pretty informative"""
   del_x1  = np.abs(boxA[0] - boxB[0])
   del_y1  = np.abs(boxA[1] - boxB[1])
   del_x2  = np.abs(boxA[2] - boxB[2])
   del_y2  = np.abs(boxA[3] - boxB[3])

   return np.sqrt(np.square(del_x1) + np.square(del_y1)) + np.sqrt(np.square(del_x2) + np.square(del_y2))


def cost(bb_0, bb_1, is_xywh=False):
    return 1 - bb_intersection_over_union(bb_0, bb_1, is_xywh)


def match(b_f_initial, b_d_final, is_xywh=False):
    # Both of these should be lists of either dicts or lists, see above 
    affinity = np.zeros( [ len(b_f_initial), len( b_d_final ) ] )
    for i, b_i in enumerate( b_f_initial ):
        for j, b_f in enumerate( b_d_final ):
            affinity[i,j] = cost( b_i, b_f, is_xywh)
    row_ind, col_ind = linear_sum_assignment( affinity )
    return ( row_ind, col_ind ) # just to be more explicit


#def match_hands(good_hands, new_detected_hands):
#    """these should both be lists in the same form, likely xyxy"""
#    for 


def xywh_to_xyxy(xywh_bbox):
    x1, y1, w, h = xywh_bbox
    x2 = x1 + w
    y2 = y1 + h
    return (x1, y1, x2, y2)


def xyxy_to_xywh(xywh_bbox):
    x1, y1, x2, y2 = xywh_bbox
    w = x2 - x1
    h = y2 - y1
    return (x1, y1, w, h)


def combine_boxes(tracked_box, detected_boxes, old_box):
    """both should be in the same format, which is now going to be x1, y1, x2, y2"""
    # algorithm:
    # if the tracker is working, then take that
    # if is stops working, take the nearest (in IOU) detection
    # continue taking the nearest detection, for now
    # TODO use the appearance model from KCF to determine if the nearby objects are what we want
    #print("tracked_box: {}".format(tracked_box))
    #print("detected_box: {}".format(detected_boxes))
    if tracked_box[2] == 0 and tracked_box[3] == 0:
        #print("\n\ntracking failed\n\n")
        max_affinity = -np.inf
        best_detected_box = (0, 0, 0, 0)
        for db in detected_boxes:# detected box
            IOU = bb_intersection_over_union(db, old_box)
            distance = corner_distances(db, old_box)
            affinity = IOU - distance / 1000.0 # distance is really only used if both boxes are zero
            if affinity > max_affinity:
                max_affinity = affinity
                best_detected_box = db
            #print("db: {}, tracked_box: {}\nIOU: {}\ndist: {}\n\n".format(db, old_box, IOU, distance))
            
        return best_detected_box, True # whether tracking failed
    else:
        # currently, if the tracker works, we take it as correct
        return tracked_box, False # whether tracking failed
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.2, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source',
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=320, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=180, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=4, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    parser.add_argument('--tracker', type=int,
                                    help='the int representing the location of the tracker in the following list: [BOOSTING, MIL,KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT]')
    parser.add_argument('--video', type=str, help='path to the video file')
    parser.add_argument('--annotation-folder', type=str,
                                    default='/home/drussel1/data/custom_annotations/', help='the path to the data')
    parser.add_argument('--which-object', type=int,
                                    default=0, help='Usually either 0 or 1')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    #cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    #tracking section
    video_name = args.video.split('/')[-1].replace('.mp4', '')
    annotation_file = args.annotation_folder + video_name + '.json'
    print(annotation_file)
    json = json.load(open(annotation_file, 'r'))

    initial_bb_dict = list(json[0].values())[args.which_object]
    print(initial_bb_dict)
    init_bbox = (initial_bb_dict['x'], initial_bb_dict['y'], initial_bb_dict['w'], initial_bb_dict['h'])
    tracker = cv2.TrackerKCF_create()

    is_first = True
    reset_tracker = False
   
    print('rm -rf outputs/{}'.format(video_name))
    print(os.system('rm -rf outputs/{}'.format(video_name)))
    os.mkdir('outputs/{}'.format(video_name))

    last_boxes = []

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        #init tracker if it's the first frame
        if is_first:
            ok = tracker.init(frame, init_bbox)
            tracker_bbox = init_bbox
            good_bbox = init_bbox # this is used in case the tracker fails
            is_first = False
        elif reset_tracker:
            print("\n\nreset tracker: {}\n\n".format(num_frames))
            print("good box {}".format(xyxy_to_xywh(good_bbox)))
            #exit()
            tracker = cv2.TrackerKCF_create()
            ok = tracker.init(frame, xyxy_to_xywh(good_bbox))
            print("ok {}".format(ok))
            reset_tracker = False 
            tracker_bbox = xyxy_to_xywh(good_bbox)
        else:
            #TODO pick when to use each one and stick with it
            ok, tracker_bbox = tracker.update(frame)

        # actual detection
        boxes, scores = detector_utils.detect_objects(
            frame, detection_graph, sess)

        # draw bounding boxes
        good_detector_boxes = detector_utils.draw_box_on_image(
            num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, frame)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image(
                    "FPS : " + str(int(fps)), frame)

            #add the tracker bounding box
            p1 = (int(tracker_bbox[0]), int(tracker_bbox[1]))
            p2 = (int(tracker_bbox[0] + tracker_bbox[2]), int(tracker_bbox[1] + tracker_bbox[3]))
            # write the tracked box
            cv2.rectangle(frame, p1, p2, (255,0,0), 3, 1)

            #convert to the new type of annotation
            tracker_bbox = xywh_to_xyxy(tracker_bbox)
            good_bbox, reset_tracker = combine_boxes(tracker_bbox, good_detector_boxes, good_bbox)
            cv2.rectangle(frame, (int(good_bbox[0]), int(good_bbox[1])), (int(good_bbox[2]), int(good_bbox[3])), (0, 0, 255), 2, 1)

            #cv2.imshow('Single-Threaded Detection', cv2.cvtColor(
            #    frame, cv2.COLOR_RGB2BGR))
            #print('/home/drussel1/dev/handtracking/outputs/{}/output{:02d}.jpeg'.format(video_name, num_frames))
            cv2.imwrite('/home/drussel1/dev/handtracking/outputs/{}/output{:05d}.jpeg'.format(video_name, num_frames), cv2.cvtColor(
                frame, cv2.COLOR_RGB2BGR))

            #if cv2.waitKey(25) & 0xFF == ord('q'):
            #    cv2.destroyAllWindows()
            #    break
