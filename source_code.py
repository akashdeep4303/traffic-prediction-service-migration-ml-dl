import cv2    # opencv library
import numpy as np

print('\33[1m'"..........VEHICLE PREDICTION.............."'\33[m')

# web camera
cap = cv2.VideoCapture('video1.mp4')  # give the camera information or video source

min_width_rect = 80  # min_width_of_rectangle
min_height_rect = 80  # min height of rectangle

count_line_position = 550  # width of the line

# initialize_substructor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()  # algorithm for the background removal except vehicle


def center_handle(x, y, w, h):  # rectangle box definition
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


detect = []
offset = 6  # allowable error between pixel
counter = 0

while True:
    ret, frame1 = cap.read()  # return as well as read video_cam
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # backend colour except vehicle
    blur = cv2.GaussianBlur(grey, (3, 3), 5)  # if any vehicle view as a blur
    # applying on each frame
    img_sub = algo.apply(blur)  # applying algorithm in blur
    d = cv2.dilate(img_sub, np.ones((5, 5)))  # It defines the kernel for the next step
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # getStructuringElement func elliptical/circular shaped kernel
    dilat = cv2.morphologyEx(d, cv2.MORPH_CLOSE, kernel)  # func used to give the structure and shape
    dilat = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  # func used to give the structure and shape
    countershape, h = cv2.findContours(dilat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # initialize shape of outer surface

    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    # it includes the line info

    for (i, c) in enumerate(countershape):  # loop for the rectangle box around the vehicle
        (x, y, w, h) = cv2.boundingRect(c)
        val_counter = (w >= min_width_rect) and (h >= min_height_rect)
        if not val_counter:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame1, "VEHICLE:" + str(counter), (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:  # loop for counting the vehicles when it passes through line
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                detect.remove((x, y))
                print("Vehicle Counter:" + str(counter))

    cv2.putText(frame1, "VEHICLE COUNTER:" + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    #cv2.imshow('Detecter', dilat)  # backend view output

    cv2.imshow('Video Original', frame1)  # original view output

    if cv2.waitKey(1) == 13:  # break the output when you click on enter
        break

cv2.destroyAllWindows()
cap.release()
