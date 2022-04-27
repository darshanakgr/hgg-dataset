import cv2
import glob
import os
import numpy as np

dataset_dir = "data/dataset"

video_files = glob.glob(os.path.join(dataset_dir, "*.mp4"))

for video_file in video_files[:1]:
    output_dir = os.path.join("data/frames_dataset", video_file.split("/")[-1].replace(".mp4", ""))
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)

    timestamps = [0.0]

    while(cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            timestamps.append(timestamps[-1] + 1000/fps)
            file_name = os.path.join(output_dir, f"{np.round(timestamps[-1])}.npz")
            # cv2.imwrite(file_name, curr_frame)
            np.savez_compressed(file_name, curr_frame)
        else:
            break

    cap.release()
