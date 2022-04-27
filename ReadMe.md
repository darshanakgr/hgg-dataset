## Hand Pose Estimation

MediaPipe estimates the hand pose estimation in two stages.
First, a palm detection model detects the hand(s) similar to an object detection model.
Second, a hand landmark model detects 21 key points in the given image of hand.

![Hand Landmarks](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

### Input Parameters

- STATIC_IMAGE_MODE: If it is set to false, the input is regarded as a video. The detection model will detect the hands 
tracks them in the subsequent frames.
  
- MAX_NUM_HANDS: Maximum number of hands to detect. Default to 2.

- MODEL_COMPLEXITY: Landmark accuracy as well as inference latency generally go up with the model complexity. 
  Default to 1.
  
- MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE: [0, 1], default=0.5



### Model output

- MULTI_HAND_LANDMARKS: Collection of detected/tracked hands, where each hand is represented as a list of 21 hand landmarks and each landmark is composed of x, y and z. x and y are normalized to [0.0, 1.0] by the image width and height respectively. z represents the landmark depth with the depth at the wrist being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.

- MULTI_HAND_WORLD_LANDMARKS: Collection of detected/tracked hands, where each hand is represented as a list of 21 hand landmarks in world coordinates. Each landmark is composed of x, y and z: real-world 3D coordinates in meters with the origin at the hand’s approximate geometric center.

- MULTI_HANDEDNESS: Collection of handedness of the detected/tracked hands (i.e. is it a left or right hand). Each hand is composed of label and score. label is a string of value either "Left" or "Right". score is the estimated probability of the predicted handedness and is always greater than or equal to 0.5 (and the opposite handedness has an estimated probability of 1 - score).

# Creating the Dataset

## Downloading YouTube Videos

PyTube is used to download the videos from the youtube with 720p resolution. Also the closed captions are downloaded in XML format. In closed captions, t denotes the current timestamp in milliseconds and d denotes the duration.


## Splitting the audio 

The audio is extracted from ffmpeg and is splitted into chunks according to closed captions. 