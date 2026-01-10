import cv2

class FrameWarper:
    def __init__(self, M, size):
        self.M = M
        self.size = size

    def warp(self, frame):
        return cv2.warpPerspective(frame, self.M, self.size)
