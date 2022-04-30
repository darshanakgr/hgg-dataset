import librosa
import numpy as np
import pandas as pd
import tqdm
import os
import glob


fps = 10                                # Number of frames per second
n_windows = 30                          # Number of windows for each frame

dataset_dir = "data/audio"

audio_files = glob.glob(os.path.join(dataset_dir, "*.wav"))

for audio_file in audio_files:
    caption_file = f"data/captions/{audio_file.split('/')[-1].replace('.wav', '.csv')}"

    output_folder = os.path.join("data/features/audio", audio_file.split("/")[-1].replace(".wav", ""))

    if not os.path.exists(output_folder): os.mkdir(output_folder)

    captions = pd.read_csv(caption_file)
    signal, sampling_rate = librosa.load(audio_file, sr=None)

    window_size = sampling_rate // fps      # Number of samples per frame

    duration = len(signal) / sampling_rate  # Duration in seconds
    n_frames = np.round(duration * fps)     # Number of frames

    mfccs = librosa.feature.mfcc(y=signal, sr=sampling_rate, n_mfcc=13, hop_length=window_size, n_fft=window_size)
    
    mfccs_d1 = librosa.feature.delta(mfccs, order=1)
    mfccs_d2 = librosa.feature.delta(mfccs, order=2)

    for i in tqdm.trange(len(captions)):
        t, d = captions.iloc[i, [0, 1]]
        
        start_ft = t * fps // 1000
        end_ft = (t + d) * fps // 1000
        
        if end_ft + n_windows > n_frames:
            continue
        
        indices = np.arange(start_ft - n_windows, end_ft + n_windows)
        windows = np.lib.stride_tricks.sliding_window_view(indices, window_shape=61)
        features = mfccs[:, windows]
        features_d1 = mfccs_d1[:, windows]
        features_d2 = mfccs_d2[:, windows]
        
        features = features.transpose(1, 2, 0)
        features_d1 = features_d1.transpose(1, 2, 0)
        features_d2 = features_d2.transpose(1, 2, 0)
        
        np.save(os.path.join(output_folder, f"audio_feat_{t}_{d}.npy"), features)
        np.save(os.path.join(output_folder, f"audio_feat_d1_{t}_{d}.npy"), features_d1)
        np.save(os.path.join(output_folder, f"audio_feat_d2_{t}_{d}.npy"), features_d2)
    

