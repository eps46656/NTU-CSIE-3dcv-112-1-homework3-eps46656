import cv2 as cv
import numpy as np
import os

DIR = os.path.dirname(__file__).replace("\\", "/")

video_path = f"{DIR}/calib_video.avi"

video_cap = cv.VideoCapture(video_path)

frams_dir = f"{DIR}/vedio_frames"

frame_i = 0

while True:
    ret, img = video_cap.read()

    if not ret:
        break

    print(f"frame_i = {frame_i}")
    cv.imwrite(f"{frams_dir}/frame-{frame_i}.png", img)
    frame_i += 1
