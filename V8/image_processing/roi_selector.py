import cv2
import numpy as np

class ROISelector:
    def __init__(self):
        self.points = []

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append([x, y])
            cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select ROI", self.frame)

    def select(self, frame):
        self.frame = frame.copy()
        cv2.imshow("Select ROI", self.frame)
        cv2.setMouseCallback("Select ROI", self._mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(self.points) == 4:
                break

        cv2.destroyAllWindows()
        return np.array(self.points, dtype="float32")
