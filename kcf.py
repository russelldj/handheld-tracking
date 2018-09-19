import cv2
import sys

video_name = sys.argv[1]
racker = cv2.TrackerKCF_create()
video = cv2.VideoCapture(video_name)
if not video.isOpened():
    print( 'video cannot be opened!')
    sys.exit(1)

_, frame = video.read()

# set box in 1st frame
box = (185, 342, 61, 46)
box = (168, 312, 100, 97)
#61: 182 81 51 69
#box = cv2.selectROI(frame, False)
print( box)

frame_no = 1

#
#while(frame_no <=61):
#    _, frame = video.read()
#    frame_no += 1
#box = (182, 81, 51, 69)


_ = tracker.init(frame, box)

for jj in range(200):
    _, frame = video.read()
    ok, box = tracker.update(frame)

    if ok:
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0]+box[2]), int(box[1]+box[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
    else:
        cv2.putText(frame, 'failure', (100, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        #box = cv2.selectROI(frame, False)
        #_ = tracker.init(frame, box)
        #ok = True
    cv2.putText(frame, str(frame_no), (30, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0),2)
    cv2.imwrite("satyaki/{}.png".format(frame_no), frame)
    print( frame_no, box)
    frame_no +=1
    #k = cv2.waitKey(11) & 0xff
    #if k==27:
    #    break
