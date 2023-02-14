#
# COMPUTER VISION FOR ROBOTICS - FI-UNAM - 2023-2
# PRACTICE 01 - THE OPENCV LIBRARY
#

import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    global frame, frame_copy, circle_radius, x1,y1,x2,y2,show,crop
    if show and (event==cv2.EVENT_LBUTTONDOWN):
        x2,y2=x,y
        show=False
        crop=True
    elif event == cv2.EVENT_LBUTTONDOWN:
        x1,y1=x,y
        x2,y2=x+1,y+1
        show=True
    elif show and (flags != cv2.EVENT_FLAG_LBUTTON ):
        x2,y2=x,y
    elif event == cv2.EVENT_RBUTTONDOWN:
        frame_copy= frame.copy()
        show=False
        crop=False
        cv2.destroyWindow('Bin')

def trackbar_callback(val):
    global tolerance
    tolerance = val

def Change_background():
    global frame_copy, background, tolerance
    cropImg = frame_copy[y1:y2, x1:x2]
    frame_hsv=cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
    rgb_mean = cv2.mean(cropImg)

    upper_rgb = (rgb_mean[0]+tolerance, rgb_mean[1]+tolerance, rgb_mean[2]+tolerance)
    lower_rgb = (rgb_mean[0]-tolerance, rgb_mean[1]-tolerance, rgb_mean[2]-tolerance)

    u = np.uint8([[[rgb_mean[0]+tolerance, rgb_mean[1]+tolerance, rgb_mean[2]+tolerance]]])
    up=cv2.cvtColor(u, cv2.COLOR_BGR2HSV)
    l = np.uint8([[[rgb_mean[0]-tolerance, rgb_mean[1]-tolerance, rgb_mean[2]-tolerance]]])
    lo=cv2.cvtColor(l, cv2.COLOR_BGR2HSV)
    u=(float(up[0][0][0])+4, max(float(lo[0][0][1]),float(up[0][0][1])),max(float(lo[0][0][2]),float(up[0][0][2])))
    l=(float(lo[0][0][0])+4, min(float(lo[0][0][1]),float(up[0][0][1])),min(float(lo[0][0][2]),float(up[0][0][2])))

    print(l, u)
    Mask_background=cv2.inRange(frame_hsv, l, u)
    cv2.imshow('Mask', Mask_background)

    Mask_background=cv2.inRange(frame_copy, lower_rgb, upper_rgb)
    Mask_video=cv2.bitwise_not(Mask_background)
    b,g,r = cv2.split(frame_copy)
    video_r = cv2.bitwise_and(Mask_video, r)
    video_g = cv2.bitwise_and(Mask_video, g)
    video_b = cv2.bitwise_and(Mask_video, b)
    back_r = cv2.bitwise_and(Mask_background, background)
    back_g = cv2.bitwise_and(Mask_background, background)
    back_b = cv2.bitwise_and(Mask_background, background)
    img_back = cv2.merge((back_b,back_g,back_r))
    img_video = cv2.merge((video_b,video_g,video_r))
    img_processing = cv2.bitwise_or(img_back, img_video)
    cv2.imshow('New Background', img_processing)


def main():
    global frame,frame_copy,tolerance,x1,y1,x2,y2,show,crop,background
    print("Practice 01")
    show,crop=False,False
    x1,y1,x2,y2=0,0,0,0
    tolerance = 10
    background = 100*np.ones((480,640), np.uint8)
    cap  = cv2.VideoCapture(0) #Default resolution 1920x1080
    cap.set(3, 640) #Change the camera resolution to 640x480
    cap.set(4, 480)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_copy = frame.copy()
        cv2.namedWindow('Edit')
        cv2.setMouseCallback('Edit', mouse_callback) #captura el mouse
        cv2.createTrackbar('r','Edit',tolerance, 150, trackbar_callback)
        if show:
            cv2.rectangle(frame_copy, (x1,y1),(x2,y2), (0,255,0),3)
        if crop:
            Change_background()
        cv2.imshow('Edit', frame_copy)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()