import pyorc

def build_gcps(
    src_points,
    dst_points,
    z0=0,
    h_ref=None
):
    if len(src_points) != len(dst_points):
        raise ValueError("src_points and dst_points must have the same length")

    gcps = {
        "src": src_points,
        "dst": dst_points,
        "z_0": z0
    }

    if h_ref is not None:
        gcps["h_ref"] = h_ref

    return gcps


def create_camera_config(
    frame,
    src_points,
    dst_points,
    crs,
    lens_position,
    z0=0,
    h_ref=None
):
    height, width = frame.shape[:2]

    gcps = build_gcps(
        src_points=src_points,
        dst_points=dst_points,
        z0=z0,
        h_ref=h_ref
    )

    camera_config = pyorc.CameraConfig(
        height=height,
        width=width,
        gcps=gcps,
        crs=crs,
        lens_position=lens_position
    )

    return camera_config , gcps
