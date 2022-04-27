'''
    To fix the issue with pytube >> https://stackoverflow.com/a/71903013
'''
import pytube as pyt
import pandas as pd
import tqdm


metadata = pd.read_csv("data/ted-metadata.csv")


def download_video(video_id):
    yt =  pyt.YouTube(f"http://youtube.com/watch?v={video_id}")
    try:
        yt.check_availability()
        stream_query = yt.streams.filter(progressive=True, file_extension='mp4', resolution="720p")

        if len(stream_query) > 0:
            stream_query.first().download(output_path="data/test/rawdata/", filename=f"{video_id}.mp4", skip_existing=True)
            caption = yt.captions['en']
            with open(f"data/dataset/{video_id}.xml", "w") as f:
                f.write(caption.xml_captions)
    except pyt.exceptions.PytubeError:
        print(f":: Video {video_id} is unavailable.")

video_ids = metadata.loc[70: 100, "video_id"].values

for video_id in tqdm.tqdm(video_ids):
    download_video(video_id.split("#")[0])