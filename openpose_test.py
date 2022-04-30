import sys
import cv2
import os


sys.path.append('/openpose/build/python/');

from openpose import pyopenpose as op

params = dict()
params["model_folder"] = "/openpose/models/"
params["hand"] = True

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()
imageToProcess = cv2.imread("data/hands_sample/hand_sample_1.jpeg")
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop([datum]) 

print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))


