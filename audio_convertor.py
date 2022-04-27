import subprocess
import os
import glob


dataset_dir = "data/dataset"

video_files = glob.glob(os.path.join(dataset_dir, "*.mp4"))

for video_file in video_files:
    output_file = os.path.join("data/audio", video_file.split("/")[-1].replace(".mp4", ".wav"))
    
    command = f"ffmpeg -i {video_file} -ab 160k -ac 1 -ar 44100 -vn {output_file}"

    subprocess.call(command, shell=True)