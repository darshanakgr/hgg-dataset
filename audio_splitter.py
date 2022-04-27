import glob
import os
import tqdm

from xml.dom import minidom
from scipy.io import wavfile

dataset_dir = "data/dataset"

video_files = glob.glob(os.path.join(dataset_dir, "*.mp4"))

for video_file in tqdm.tqdm(video_files):
    video_id = video_file.split("/")[-1].replace(".mp4", "")
    output_dir = os.path.join("data/audio_chunks", video_id)
    
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    
    file = minidom.parse(f"data/dataset/{video_id}.xml")
    rate, data = wavfile.read(f"data/audio/{video_id}.wav")

    captions = file.getElementsByTagName("p")

    for i, caption in enumerate(captions):
        t = int(caption.attributes['t'].value)
        d = int(caption.attributes['d'].value)
        # print(f"{t} - {t + d}: {caption.firstChild.data}")
        
        star_idx = rate * (t // 1000)
        end_idx = rate * ((t + d) // 1000)
        
        wavfile.write(os.path.join(output_dir, f'chunk_{i}_{t}_{d}.wav'), rate, data[star_idx: end_idx+1])