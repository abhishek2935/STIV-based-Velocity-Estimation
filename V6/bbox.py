from shapely.geometry import Polygon


def bbox_from_world_coords(coords):
    """
    coords: list of (x, y) tuples in world coordinates
    """
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]

    return Polygon(coords)


def set_bbox_from_world_coords(cam_config, coords):
    cam_config.bbox = bbox_from_world_coords(coords)
    return cam_config


def set_bbox_from_image_corners(cam_config, corners):
    """
    corners: list of [x, y] pixel coordinates
    """
    cam_config.set_bbox_from_corners(corners)
    return cam_config
