from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import json
import time


detection_graph, sess = detector_utils.load_inference_graph()

def combine_boxes(tracked_box, detected_boxes):
    """both should be in the same format, I'm not sure what that should be"""

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
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    #tracking section
    annotation_file = args.annotation_folder + args.video.split('/')[-1].replace('mp4', 'json')
    print(annotation_file)
    json = json.load(open(annotation_file, 'r'))

    initial_bb_dict = list(json[0].values())[0]
    print(initial_bb_dict)
    init_bbox = (initial_bb_dict['x'], initial_bb_dict['y'], initial_bb_dict['w'], initial_bb_dict['h'])
    tracker = cv2.TrackerKCF_create()

    is_first = True

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        try:
            print(frame.shape)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        #init tracker if it's the first frame
        if is_first:
            ok = tracker.init(frame, init_bbox)
            tracker_bbox = init_bbox
            is_first = False
        else:
            ok, tracker_bbox = tracker.update(frame)

        # actual detection
        boxes, scores = detector_utils.detect_objects(
            frame, detection_graph, sess)

        # draw bounding boxes
        good_boxes = detector_utils.draw_box_on_image(
            num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, frame)
        print(good_boxes)

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
            print(tracker_bbox)
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

            cv2.imshow('Single-Threaded Detection', cv2.cvtColor(
                frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ",  num_frames,
                  "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))
