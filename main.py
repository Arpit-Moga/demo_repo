import cv2
from datetime import datetime

video = cv2.VideoCapture(0)
first_frame = None
Status_List = []
Change_movement = True

while True :

    check , frame = video.read()
    movement = False

    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame,(9,9),0)

    if first_frame is None : 
        first_frame = gray_frame
        continue

    delta_mask = cv2.absdiff(first_frame,gray_frame)
    thresh_frame = cv2.threshold(delta_mask,80,255,cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame , None , iterations=5)

    (cnts , _) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    for i in cnts : 
        if movement != Change_movement : 
            Status_List.append(datetime.now())
            Change_movement = movement

        if cv2.contourArea(i) < 2000 : 
            continue

        movement = True
        (x,y,w,h) = cv2.boundingRect(i)
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,0) ,2)

    cv2.imshow("Camera_original",frame)
    cv2.imshow("Camera_delta",delta_mask)

    key = cv2.waitKey(1)
    if key == ord('q') : 
        break

video.release()
cv2.destroyAllWindows()
print(Status_List)