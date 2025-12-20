from pathlib import Path

from video import load_video_frame
from camconfig import create_camera_config

from bbox import set_bbox_from_world_coords
# or: from bbox import set_bbox_from_image_corners

from pose import solve_camera_pose, project_world_points

from pathlib import Path


# ---------------- VIDEO PATH ----------------
VIDEO_PATH = Path(__file__).resolve().parent.parent / "Videos" / "bridge_main_1.MP4"


# ---------------- USER INPUT (EDIT HERE) ----------------

# Image coordinates (pixels)
src_points = [
    [1750, 600],   # bottom right
    [1400 , 360],   #  top right 
    [420, 370],    # top left 
    [250 , 470]     # bottom left
]

# World coordinates (UTM / meters)
dst_points = [
    [366400.720, 2053619.745],  # bottom right (matches src[0])
    [366418.640, 2053631.682],  # top right (matches src[1]) 
    [366362.501, 2053663.064],  # top left (matches src[2])
    [366353.367, 2053654.939]   # bottom left (matches src[3])
]

# Camera position in world coordinates (X, Y, Z)
lens_position = [366343.108, 2053606.870, 30]

# Coordinate Reference System (EPSG)
crs = 32643


# ---------------- PROCESSING ----------------
frame = load_video_frame(str(VIDEO_PATH))

cam_config , gcps= create_camera_config(
    frame=frame,
    src_points=src_points,
    dst_points=dst_points,
    crs=crs,
    lens_position=lens_position
)

print("Camera configuration created successfully")



# ---------------- USER INPUT ----------------

corners = [
    [250, 600],     # bottom left
    [500, 400],     # top left
    [1400, 400],    # top right
    [1500, 600]     # bottom right
]

bbox_coords = [
    (366351.969, 2053651.297),
    (366365.040, 2053663.932),
    (366409.232, 2053629.534),
    (366391.942, 2053617.261),
    (366351.969, 2053651.297)
]


# ---------------- APPLY BBOX ----------------

# Option A: image-based bbox
# cam_config = set_bbox_from_image_corners(cam_config, corners)

# Option B: world-coordinate bbox
cam_config = set_bbox_from_world_coords(cam_config, bbox_coords)



######################


# ---------------- POSE ESTIMATION ----------------
rvec, tvec = solve_camera_pose(gcps, cam_config)

print("rvec =", rvec)
print("tvec =", tvec)


# ---------------- BBOX PROJECTION ----------------
bbox_3d = [
    [366351.969, 2053651.297, 0],
    [366365.040, 2053663.932, 0],
    [366409.232, 2053629.534, 0],
    [366391.942, 2053617.261, 0],
    [366351.969, 2053651.297, 0],
]

img_bbox = project_world_points(
    world_points=bbox_3d,
    rvec=rvec,
    tvec=tvec,
    cam_config=cam_config
)


########################

from visualization import plot_projection
import numpy as np

# Image GCPs
img_pts = np.array(gcps["src"], dtype=np.float32)

# Camera intrinsics
camera_matrix = cam_config.camera_matrix
dist_coeffs   = cam_config.dist_coeffs

plot_projection(
    frame=frame,
    img_pts=img_pts,
    img_bbox=img_bbox,
    cam_config=cam_config,
    rvec=rvec,
    tvec=tvec,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)



################

BASE_DIR = Path(__file__).resolve().parent   # src/
OUTPUT_DIR = BASE_DIR / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cam_config.to_file(OUTPUT_DIR / "BR.json")