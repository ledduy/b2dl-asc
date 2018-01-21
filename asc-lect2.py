#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:34:25 2018

@author: ledinhduy
"""

from skimage import io
import cv2
import matplotlib.pyplot as plt


url = 'https://upload.wikimedia.org/wikipedia/en/2/24/Lenna.png'
img = io.imread(url)


return
#cv2.imwrite('lena-x.jpg', img)
#cv2.imwrite('lena.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

#plt.axis('off')
#plt.title('Result')
#plt.imshow(img)
#plt.show()



camera_url = 'http://4co2.vp9.tv/chn/DNG8/v.m3u8'
camera_url = 'http://113.161.67.249:8083//mjpg/video.mjpg'
cap = cv2.VideoCapture(camera_url)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()