from utils import detector_utils as detector_utils 
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
import json
import time


frame_processed = 0
score_thresh = 0.2

# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue

def worker(input_q, output_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        #print("> ===== in worker loop, frame ", frame_processed)
        frame = input_q.get()
        if (frame is not None):
            # actual detection
            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # draw bounding boxes
            detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"], scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)
            # add frame annotated with bounding box to queue
            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=str,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-is-file',  dest='is_file', type=bool,
                        default=True, help='Device index of the camera.')
    parser.add_argument('-nhands', '--num_hands', dest='num_hands', type=int,
                        default=2, help='Max number of hands to detect.')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=300, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=200, help='Height of the frames in the video stream.')
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

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    if args.is_file:
        video_capture = cv2.VideoCapture(args.video)
        print(args.video)
        ok, frame = video_capture.read()
        #cv2.imshow('', frame)
        #time.sleep(2)
        
    else:
        video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()

    cap_params = {}
    frame_processed = 0
    if args.is_file:
        cap_params['im_width'] = video_capture.get(3)
        cap_params['im_height'] = video_capture.get(4)
    else:
        cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    # spin up workers to paralleize detection.
    pool = Pool(args.num_workers, worker,
                (input_q, output_q, cap_params, frame_processed))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)
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
        if args.is_file:
            ret, frame = video_capture.read()
        else:
            frame = video_capture.read()
        #frame = cv2.flip(frame, 1)
        
        #init tracker if it's the first frame
        if is_first:
            ok = tracker.init(frame, init_bbox)
            tracker_bbox = init_bbox
            is_first = False
        else:
            ok, tracker_bbox = tracker.update(frame)
        print(index)
        index += 1

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = output_q.get()

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time
        # print("frame ",  index, num_frames, elapsed_time, fps)

        if (output_frame is not None):
            if (args.display > 0):
                if (args.fps > 0):
                    detector_utils.draw_fps_on_image(
                        "FPS : " + str(int(fps)), output_frame)
                # draw the tracker output
                p1 = (int(tracker_bbox[0]), int(tracker_bbox[1]))
                p2 = (int(tracker_bbox[0] + tracker_bbox[2]), int(tracker_bbox[1] + tracker_bbox[3]))
                print(tracker_bbox)
                cv2.rectangle(output_frame, p1, p2, (255,0,0), 2, 1)

                cv2.imshow('Multi-Threaded Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if (num_frames == 400):
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("frames processed: ",  index,
                          "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))
        else:
            # print("video end")
            break
    elapsed_time = (datetime.datetime.now() -
                    start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
