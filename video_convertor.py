import subprocess
import os
import glob


dataset_dir = "data/dataset"

video_files = glob.glob(os.path.join(dataset_dir, "*.mp4"))

for video_file in video_files:
    output_file = os.path.join("data/video", video_file.split("/")[-1])
    
    command = f"ffmpeg -i {video_file} -filter:v fps=fps=10 {output_file}"

    subprocess.call(command, shell=True)