import os
import numpy as np
import pandas as pd
import tqdm
import json
import glob


fps = 10

# Helper functions

def pre_process(X, scale=1):
    X[:, 0] = X[:, 0] - X[0, 0] 
    X[:, 1] = X[:, 1] - X[0, 1]
    # Flip over x axis
    X[:, 1] = -X[:, 1]
    return X * scale

def fill_missing_frames(x):
    return pd.DataFrame(np.array(x)).fillna(method="ffill").fillna(method="bfill").to_numpy()

def generate_keypoints(frame_ids, video_id, threshold=0.6):
    left_hand_landmarks_x = []
    left_hand_landmarks_y = []
    right_hand_landmarks_x = []
    right_hand_landmarks_y = []
    missing_frames = 0


    for frame_id in frame_ids:
        keypoints_frame_file = f"data/openpose/{video_id}/{video_id}_{frame_id:012d}_keypoints.json"

        json_file = open(keypoints_frame_file)
        keypts = json.loads(json_file.read())


        if len(keypts["people"]) == 1:
            pose_keypoints_2d = np.array(keypts["people"][0]["pose_keypoints_2d"]).reshape(25, 3).astype(np.int64)
            
            b1, b8 = pose_keypoints_2d[[1, 8], :2]
            person_height = np.linalg.norm(b1 - b8)
            
            if person_height > 0:
                scale = 100 / person_height 
                
                left_hand_exists = sum(keypts["people"][0]["hand_left_keypoints_2d"]) > 0
                right_hand_exists = sum(keypts["people"][0]["hand_right_keypoints_2d"]) > 0
                
                if left_hand_exists:
                    hand_left_keypoints_2d = np.array(keypts["people"][0]["hand_left_keypoints_2d"]).reshape(21, 3).astype(np.int64)
                    hand_left_keypoints_2d = pre_process(hand_left_keypoints_2d, scale)
                    left_hand_landmarks_x.append(hand_left_keypoints_2d[:, 0])
                    left_hand_landmarks_y.append(hand_left_keypoints_2d[:, 1])
                else:
                    missing_frames += 0.5
                    left_hand_landmarks_x.append([np.nan] * 21)
                    left_hand_landmarks_y.append([np.nan] * 21)
                
                if right_hand_exists:
                    hand_right_keypoints_2d = np.array(keypts["people"][0]["hand_right_keypoints_2d"]).reshape(21, 3).astype(np.int64)
                    hand_right_keypoints_2d = pre_process(hand_right_keypoints_2d, scale)
                    right_hand_landmarks_x.append(hand_right_keypoints_2d[:, 0])
                    right_hand_landmarks_y.append(hand_right_keypoints_2d[:, 1])
                else:
                    missing_frames += 0.5
                    right_hand_landmarks_x.append([np.nan] * 21)
                    right_hand_landmarks_y.append([np.nan] * 21)
            else:
                missing_frames += 1
                left_hand_landmarks_x.append([np.nan] * 21)
                left_hand_landmarks_y.append([np.nan] * 21)
                right_hand_landmarks_x.append([np.nan] * 21)
                right_hand_landmarks_y.append([np.nan] * 21)
                
        else:
            missing_frames += 1
            left_hand_landmarks_x.append([np.nan] * 21)
            left_hand_landmarks_y.append([np.nan] * 21)
            right_hand_landmarks_x.append([np.nan] * 21)
            right_hand_landmarks_y.append([np.nan] * 21)
            
    ratio = missing_frames / len(frame_ids)
    
    if ratio < threshold:
        landmarks = np.concatenate((left_hand_landmarks_x, left_hand_landmarks_y, right_hand_landmarks_x, right_hand_landmarks_y), axis=1)
        landmarks = fill_missing_frames(landmarks)
        return landmarks, True
    else:
        return None, False
    



dataset_dir = "data/captions"

caption_files = glob.glob(os.path.join(dataset_dir, "*.csv"))

for caption_file in caption_files:
        
    video_id = caption_file.split("/")[-1].split(".")[0]

    captions = pd.read_csv(caption_file)

    output_folder = os.path.join("data/features/handpose_openpose", video_id)

    if not os.path.exists(output_folder): os.mkdir(output_folder)

    valid_samples = 0

    for i in tqdm.trange(len(captions)):
        
        t, d = captions.iloc[i, [0, 1]]

        start_ft = t * fps // 1000
        end_ft = (t + d) * fps // 1000
        frame_ids = np.arange(start_ft, end_ft + 1)

        landmarks, is_valid = generate_keypoints(frame_ids, video_id, threshold=0.4)
        
        valid_samples = valid_samples + 1 if is_valid else valid_samples
        
        np.save(os.path.join(output_folder, f"handpose_{t}_{d}.npy"), landmarks)
        
    print(f"{caption_file} -> {valid_samples} samples.")
    
    


    