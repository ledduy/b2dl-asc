#written by DuyLD

from skimage import io
import cv2
import matplotlib.pyplot as plt


def detect(tds_classifier_xml, frame, gray_img):

   tds_cascade = cv2.CascadeClassifier(tds_classifier_xml)

   sign = tds_cascade.detectMultiScale(gray_img, 1.25, 3)
   cnt = 0
   for (x,y,w,h) in sign:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
       print ('%d - %d - %d - %d', x, y, w, h
)
       cnt = cnt+1


 #   cv2.rectangle(frame,(10,10),(100,100),(255,0,0),2)

   return frame, cnt

video_name = 'drive'
video_name = 'MAH00019'
#video_name = 'test2'
camera_url='../../media/%s.mp4' % video_name
video = cv2.VideoCapture(camera_url)

frame_w =video.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
legend_loc_x = int(frame_w*0.1)
legend_loc_y = int(frame_h*0.1)


model_list = {'no_parking' : 
'../../media/enrich/no_parking_cascade.xml'
, 'speed_limit_50' : '../../media/enrich/speed_limit_50_cascade.xml', 'merge_line'
 : '../../media/enrich/traffic_merge_from_the_right_cascade.xml'


}

model_list = {'no_parking' : 
'../../media/no_parking_cascade.xml'
, 'speed_limit_50' : '../../media/speed_limit_50_cascade.xml', 'merge_line'
 : '../../media/traffic_merge_from_the_right_cascade.xml'


}


font = cv2.FONT_HERSHEY_SIMPLEX
cnt = 0
frame_id = 0
frame_rate = 10
while(True):
    # Capture frame-by-frame
    ret, frame = video.read()

    if ret != True:
        break
    cnt = cnt + 1
    frame_id = frame_id + 1

    if (cnt % frame_rate == 0):
        cnt = 0
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        output_text = 'frame %d' % (frame_id)
        all_cnt = 0
        for sign_name in model_list:
            frame, cnt_out = detect(model_list[sign_name], frame, gray_img)
            if (cnt_out > 0):
                all_cnt = all_cnt + cnt_out
                output_text = '%s - %s' % (output_text, sign_name)
                print(output_text)

        cv2.putText(frame, output_text, (legend_loc_x, legend_loc_y), font, 1, (0,255,0), 2, cv2.LINE_AA)

        if all_cnt > 0:
            output_file = '%s-%s.jpg' % (video_name, output_text)
            cv2.imwrite(output_file, frame)

        cv2.imshow('Demo TDS', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()

