import os
import numpy as np
import pandas as pd
import tqdm
import cv2
import glob
import mediapipe as mp

from google.protobuf.json_format import MessageToDict


fps = 10                                # Number of frames per second

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7, model_complexity=1)

        
def decode_hand_landmarks(i, results):
    classification = MessageToDict(results.multi_handedness[i])["classification"][0]
    label = classification["label"]
    
    hand_landmarks = results.multi_hand_landmarks[i]

    X = []

    for j in range(len(hand_landmarks.landmark)):
        X.append([hand_landmarks.landmark[j].x, hand_landmarks.landmark[j].y, 1.0])
        
    X = np.array(X)
    T = np.identity(3)
    T[:2, 2] = -X[0, :2]
    
    x, y, _ = np.dot(T, X.T)
        
    return label, x, y


def fill_missing_frames(x):
    return pd.DataFrame(np.array(x)).fillna(method="ffill").fillna(method="bfill").to_numpy()


def get_images_for_caption(t, d, fps, video_file):
    start_ft = t * fps // 1000
    end_ft = (t + d) * fps // 1000

    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_ft)

    images = []

    while(cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            current_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
            
            if current_frame_id >= end_ft:
                break
            
            images.append(curr_frame)
        else:
            break
    
    return images


def generate_keypoints(images, model, threshold=0.6):

    missed_frames = 0
    left_hand_landmarks_x = []
    left_hand_landmarks_y = []
    right_hand_landmarks_x = []
    right_hand_landmarks_y = []

    for i in range(len(images)):

        image = cv2.flip(images[i], 1)

        results = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_hand_landmarks:
            missed_frames += 1
            left_hand_landmarks_x.append([np.nan] * 21)
            left_hand_landmarks_y.append([np.nan] * 21)
            right_hand_landmarks_x.append([np.nan] * 21)
            right_hand_landmarks_y.append([np.nan] * 21)
        elif len(results.multi_hand_landmarks) == 1:
            label, x, y = decode_hand_landmarks(0, results)
            missed_frames += 0.5
            if label == "Left":
                left_hand_landmarks_x.append(x)
                left_hand_landmarks_y.append(y)
                right_hand_landmarks_x.append([np.nan] * 21)
                right_hand_landmarks_y.append([np.nan] * 21)
            else:
                left_hand_landmarks_x.append([np.nan] * 21)
                left_hand_landmarks_y.append([np.nan] * 21)
                right_hand_landmarks_x.append(x)
                right_hand_landmarks_y.append(y)
        else:
            pLabel = None
            for i in range(2):
                label, x, y = decode_hand_landmarks(i, results)
                if label == pLabel:
                    missed_frames += 0.5
                    if label == "Left":
                        right_hand_landmarks_x.append([np.nan] * 21)
                        right_hand_landmarks_y.append([np.nan] * 21)
                    else:
                        left_hand_landmarks_x.append([np.nan] * 21)
                        left_hand_landmarks_y.append([np.nan] * 21)
                else:
                    if label == "Left":
                        left_hand_landmarks_x.append(x)
                        left_hand_landmarks_y.append(y)
                    else:
                        right_hand_landmarks_x.append(x)
                        right_hand_landmarks_y.append(y)
                pLabel = label
                    
    ratio = missed_frames / len(images)
    
    if ratio < threshold:
        left_hand_landmarks_x = fill_missing_frames(left_hand_landmarks_x)
        left_hand_landmarks_y = fill_missing_frames(left_hand_landmarks_y)
        right_hand_landmarks_x = fill_missing_frames(right_hand_landmarks_x)
        right_hand_landmarks_y = fill_missing_frames(right_hand_landmarks_y)
        
        drm = np.sqrt(np.square(right_hand_landmarks_x[:, 9]) + np.square(right_hand_landmarks_y[:, 9]))
        drs = 0.3 / drm

        dlm = np.sqrt(np.square(left_hand_landmarks_x[:, 9]) + np.square(left_hand_landmarks_y[:, 9]))
        dls = 0.3 / dlm

        left_hand_landmarks_x = np.multiply(left_hand_landmarks_x, dls[:, np.newaxis])
        left_hand_landmarks_y = np.multiply(left_hand_landmarks_y, dls[:, np.newaxis])
        right_hand_landmarks_x = np.multiply(right_hand_landmarks_x, drs[:, np.newaxis])
        right_hand_landmarks_y = np.multiply(right_hand_landmarks_y, drs[:, np.newaxis])

        landmarks = np.concatenate((left_hand_landmarks_x, left_hand_landmarks_y, right_hand_landmarks_x, right_hand_landmarks_y), axis=1)
        
        return landmarks, True
    
    else:
        return None, False
    

dataset_dir = "data/video"

video_files = glob.glob(os.path.join(dataset_dir, "*.mp4"))

for video_file in video_files:
        
    caption_file = f"data/captions/{video_file.split('/')[-1].replace('.mp4', '.csv')}"

    captions = pd.read_csv(caption_file)

    output_folder = os.path.join("data/features/handpose", video_file.split("/")[-1].replace(".mp4", ""))

    if not os.path.exists(output_folder): os.mkdir(output_folder)

    valid_samples = 0

    for i in tqdm.trange(len(captions)):
        
        t, d = captions.iloc[i, [0, 1]]

        images = get_images_for_caption(t, d, fps, video_file)

        landmarks, is_valid = generate_keypoints(images, model, threshold=0.5)
        
        valid_samples = valid_samples + 1 if is_valid else valid_samples
        
        np.save(os.path.join(output_folder, f"handpose_{t}_{d}.npy"), landmarks)
        
    print(f"{video_file} -> {valid_samples} samples.")
    
    


    