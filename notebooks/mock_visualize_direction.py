import matplotlib

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_pitch_3d(all_latents, stats, pct=0.15):
    data = torch.stack([t.detach().cpu() for t in all_latents]).numpy()
    data = data.reshape(data.shape[0], -1)
    pitches = np.array([s["pitch"] for s in stats])


    # 2. Reduce dimensionality to 3D using PCA
    pca = PCA(n_components=3)
    coords = pca.fit_transform(data)

    # --- NEW: Arrow Logic ---
    # Find indices for low and high pitch
    indices = np.argsort(pitches)
    num = max(1, int(len(indices) * pct))
    low_idx = indices[:num]
    high_idx = indices[-num:]

    # Calculate centroids in the HIGH-DIMENSIONAL space first
    mean_low_vec = data[low_idx].mean(axis=0)
    mean_high_vec = data[high_idx].mean(axis=0)

    # Project these centroids into the 3D PCA space
    # (Important: use .transform, not .fit_transform)
    low_centroid_3d = pca.transform(mean_low_vec.reshape(1, -1))[0]
    high_centroid_3d = pca.transform(mean_high_vec.reshape(1, -1))[0]
    # ------------------------

    # 3. Infinite Line Logic (Extrapolation)
    direction = high_centroid_3d - low_centroid_3d
    # Create points far out in both directions (e.g., +/- 10x the distance)
    line_start = low_centroid_3d - direction * 5
    line_end = high_centroid_3d + direction * 5
    line_pts = np.vstack([line_start, line_end])

    # 3. Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- Formatting: Hide numbers and labels, keep grid ---
    ax.set_title("Pitch Direction in Speaker Embedding Space", fontsize=15)

    # Remove tick labels (numbers)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Remove axis labels (PC 1, PC 2, etc.)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    # Ensure grid remains visible
    ax.grid(True)
    # ------------------------------------------------------

    cmap = plt.get_cmap('viridis')

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=pitches,
        cmap=cmap,
        alpha=0.6,
        edgecolors='none'
    )


    # --- NEW: Draw the Arrow ---
    # ax.quiver(x, y, z, dx, dy, dz)
    ax.quiver(
        low_centroid_3d[0], low_centroid_3d[1], low_centroid_3d[2],  # Start
        high_centroid_3d[0] - low_centroid_3d[0],                  # DX
        high_centroid_3d[1] - low_centroid_3d[1],                  # DY
        high_centroid_3d[2] - low_centroid_3d[2],                  # DZ
        color='red', linewidth=4, label='Pitch Direction', arrow_length_ratio=0.2, zorder=3
    )
    # ---------------------------

    # --- NEW: Pink Centroids ---
    ax.scatter(
        [low_centroid_3d[0], high_centroid_3d[0]],
        [low_centroid_3d[1], high_centroid_3d[1]],
        [low_centroid_3d[2], high_centroid_3d[2]],
        color='pink', s=200, label='Centroids', edgecolors='black', zorder=2, alpha=0.8
    )

    # --- NEW: The Infinite Line ---
    ax.plot(
        line_pts[:, 0], line_pts[:, 1], line_pts[:, 2],
        color='red', linestyle='-', linewidth=2, alpha=0.8,  label='Pitch Axis'
    )

    ax.set_title("3D PCA Projection: Speaker Identity Space with Pitch Vector", fontsize=15)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")


    ax.set_proj_type('persp', focal_length=0.3)
    margin = 100
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_zlim(-margin, margin)
    ax.view_init(elev=10, azim=108, roll=-2)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0)
    cbar.set_label('Pitch Value (Measured)', rotation=270, labelpad=15)

    plt.show()

# Usage:
# visualize_pitch_3d(all_latents, stats)