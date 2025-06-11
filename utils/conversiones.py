import numpy as np
import cv2

def from_opencv_to_pygfx(rvec, tvec):
    pose = np.eye(4)
    pose[0:3,3] = tvec.T
    pose[0:3,0:3] = cv2.Rodrigues(rvec)[0]
    pose[1:3] *= -1
    return np.linalg.inv(pose)
