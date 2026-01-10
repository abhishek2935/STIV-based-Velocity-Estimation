import cv2
import numpy as np

def get_perspective_transform(pts):
    width_px = int(max(
        np.linalg.norm(pts[1] - pts[0]),
        np.linalg.norm(pts[2] - pts[3])
    ))

    height_px = int(max(
        np.linalg.norm(pts[2] - pts[1]),
        np.linalg.norm(pts[3] - pts[0])
    ))

    dst = np.array([
        [0, 0],
        [width_px - 1, 0],
        [width_px - 1, height_px - 1],
        [0, height_px - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    return M, (width_px, height_px)
