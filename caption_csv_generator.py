import glob
import pandas as pd
from xml.dom import minidom
import tqdm
import os

dataset_dir = "data/dataset"

caption_files = glob.glob(os.path.join(dataset_dir, "*.xml"))

for caption_file in tqdm.tqdm(caption_files):
    output_file = os.path.join("data/captions", caption_file.split("/")[-1].replace(".xml", ".csv"))
    caption_file = minidom.parse(caption_file)
    
    captions = pd.DataFrame(columns=["start", "duration", "text"])

    for caption in caption_file.getElementsByTagName("p"):
        t = int(caption.attributes['t'].value)
        d = int(caption.attributes['d'].value)
        text = caption.firstChild.data
        
        df = pd.DataFrame([[t, d, text]], columns=["start", "duration", "text"])
        
        captions = pd.concat([captions, df], ignore_index=True)
    
    captions.to_csv(output_file, index=None)
    