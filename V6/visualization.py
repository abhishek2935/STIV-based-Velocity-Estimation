import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def plot_projection(
    frame,
    img_pts,
    img_bbox,
    cam_config,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs,
    title="BBOX Projection"
):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(frame)

    # -------------------------
    # Plot GCPs
    # -------------------------
    ax.scatter(
        img_pts[:, 0],
        img_pts[:, 1],
        s=80,
        color="#2a9df4",
        edgecolor="white",
        linewidth=1.5,
        label="Control points"
    )

    # -------------------------
    # Plot lens position (optional)
    # -------------------------
    # Ensure correct type
    camera_matrix = np.array(camera_matrix, dtype=np.float32)
    dist_coeffs   = np.array(dist_coeffs, dtype=np.float32)

    # -------------------------
    # Plot lens position (optional)
    # -------------------------
    if hasattr(cam_config, "lens_position"):
        lp = np.array(cam_config.lens_position[:2], dtype=np.float32)
        lp3d = np.hstack([lp, 0]).reshape(1, 3)

        lp_img, _ = cv2.projectPoints(
            lp3d,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs
        )

        lp_img = lp_img.reshape(2)

        ax.scatter(
            lp_img[0],
            lp_img[1],
            s=120,
            color="#ff8c00",
            edgecolor="white",
            linewidth=1.5,
            label="Lens position"
        )


    # -------------------------
    # Plot BBOX
    # -------------------------
    poly = patches.Polygon(
        img_bbox,
        closed=True,
        facecolor="#1f77b430",
        edgecolor="#1f77b4",
        linewidth=2,
        label="bbox visible"
    )
    ax.add_patch(poly)

    # -------------------------
    # Formatting
    # -------------------------
    ax.set_title(title)
    ax.set_xlabel("column [-]")
    ax.set_ylabel("row [-]")

    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)

    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.show()
