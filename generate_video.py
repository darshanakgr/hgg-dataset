import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import cv2
import os
import tqdm
import argparse

parser = argparse.ArgumentParser(description="Generate video from estimatede gestures")
parser.add_argument("--fps", type=int, default=10, help="Number of frames per second")
parser.add_argument("--id", type=str, required=True, help="ID of the video")
parser.add_argument("--prediction-dir", type=str, required=True, help="Directory with predictions")
parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the video")

args = parser.parse_args()

fps = args.fps
video_id = args.id

print("::   Check for output directory")

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
    print("::   Create a directory for the output")

print("::   Creating a temporary directory to store gesture frames")

temp_folder = os.path.join(args.output_dir, "hand_landmarks")

if not os.path.exists(temp_folder):
    os.mkdir(temp_folder)
else:
    os.system(f"rm -rf {temp_folder}/*")


caption_file = f"data/captions/{video_id}.csv"
video_file = f"data/video/{video_id}.mp4"

captions = pd.read_csv(caption_file)

print("::   Generating hand landmarks")

for c in tqdm.trange(len(captions)):
    t, d = captions.iloc[c, [0, 1]]
    
    start_ft = t * fps // 1000
    end_ft = (t + d) * fps // 1000

    frame_ids = np.arange(start_ft, end_ft)

    prediction_file = os.path.join(args.prediction_dir, "{v}/handpose_{t}_{d}.npy".format(v=video_id, t=t, d=d))
    
    if not os.path.exists(prediction_file):
        continue
    
    if os.path.getsize(prediction_file) == 278:
        continue
    
    landmarks = np.load(prediction_file)

    for f in frame_ids:
        i = f - start_ft
        
        fig, ax = plt.subplots(1, 2, figsize=(5, 3))

        lx, ly, rx, ry = landmarks[i, :21], landmarks[i, 21:42], landmarks[i, 42:63], landmarks[i, 63:]                
        ax[0].plot(rx, ry, "o")
        ax[0].set_xlim(-1, 1)
        ax[0].set_ylim(-1, 1)
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        
        ax[1].plot(lx, ly, "o")
        ax[1].set_xlim(-1, 1)
        ax[1].set_ylim(-1, 1)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        
        fig.tight_layout()
        fig.savefig(f"{temp_folder}/img_{f}.png")
        plt.close()


print("::   Generating video")

w, h = (1280, 720)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter(f"{args.output_dir}/{video_id}.mp4", fourcc, fps, (w, h))

cap = cv2.VideoCapture(video_file)
    
while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        current_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
        
        hands_landmarks_path = f"{temp_folder}/img_{current_frame_id}.png"
        
        if os.path.exists(hands_landmarks_path):
        
            hands_image = cv2.imread(hands_landmarks_path)
            
            x = np.zeros(curr_frame.shape, dtype=np.uint8)
            x[:hands_image.shape[0], :hands_image.shape[1], :] = hands_image

            curr_frame = cv2.addWeighted(curr_frame, 1, x, 0.6, 0)
        
        writer.write(curr_frame)
    else:
        break

writer.release()

print("::   Removing temporary directory")

os.system(f"rm -rf {temp_folder}")

print("::   Done")