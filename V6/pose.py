import numpy as np
import cv2


def solve_camera_pose(gcps, cam_config):
    """
    Solves camera pose using GCPs and CameraConfig.

    Returns:
        rvec, tvec
    """
    img_pts = np.array(gcps["src"], dtype=np.float32)

    world_pts = np.array(
        [[x, y, 0] for x, y in gcps["dst"]],
        dtype=np.float32
    )

    camera_matrix = np.array(cam_config.camera_matrix, dtype=np.float32)
    dist_coeffs   = np.array(cam_config.dist_coeffs, dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        world_pts,
        img_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        raise RuntimeError("solvePnP failed")

    return rvec, tvec


def project_world_points(
    world_points,
    rvec,
    tvec,
    cam_config
):
    """
    Projects 3D world points to image coordinates.

    world_points: list of [X, Y, Z]
    """
    world_pts = np.array(world_points, dtype=np.float32)

    camera_matrix = np.array(cam_config.camera_matrix, dtype=np.float32)
    dist_coeffs   = np.array(cam_config.dist_coeffs, dtype=np.float32)

    img_pts, _ = cv2.projectPoints(
        world_pts,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    return img_pts.reshape(-1, 2)
