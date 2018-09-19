from skvideo.io import VideoWriter
import numpy
w, h = 100
writer = VideoWriter(filename, frameSize=(w, h))
writer.open()
image = numpy.zeros((h, w, 3))
writer.write(image)
writer.release()

#import numpy as np
#import cv2
#
## Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
#
#frame = cv2.imread('images/accuracy.jpg')
#print(frame)
#for _ in range(100):
#    print(_)
#    # write the flipped frame
#    out.write(frame)
#
#    #cv2.imshow('frame',frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
## Release everything if job is finished
#out.release()
#cv2.destroyAllWindows()
#
