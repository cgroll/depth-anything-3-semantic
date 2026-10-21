# %%
#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Point Cloud Cleanup Script - Interactive Style

This script is structured as independent blocks that can be run interactively,
similar to Jupyter notebook cells. Each block is self-contained and can be
executed independently for iterative experimentation.

Workflow:
1. Load point cloud from .glb file
2. Quick downsample to 5% for fast visualization
3. Apply various cleanup methods interactively
4. Save final result

Usage:
    Run blocks individually by uncommenting and executing
"""

# %% BLOCK 1: Imports and Setup
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh

# Configuration
INPUT_PATH = "data/outputs/semantic_3d/point_cloud_semantic.glb"
OUTPUT_PATH = "data/outputs/semantic_3d/point_cloud_cleaned.glb"
VIS_OUTPUT_DIR = "data/outputs/semantic_3d/visualizations"

# Quick downsample ratio (5% = 0.05)
QUICK_DOWNSAMPLE_RATIO = 0.05

# Create visualization output directory
os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

print("✅ Imports loaded")
print(f"Input: {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")
print(f"Visualizations: {VIS_OUTPUT_DIR}")


# %% BLOCK 2: Load Point Cloud
print("\n" + "="*80)
print("LOADING POINT CLOUD")
print("="*80)

# Load using trimesh
print(f"\n📂 Loading from {INPUT_PATH}...")
mesh = trimesh.load(INPUT_PATH)

# Extract points and colors
if isinstance(mesh, trimesh.PointCloud):
    points = mesh.vertices
    colors = mesh.colors[:, :3] if mesh.colors is not None else None
elif isinstance(mesh, trimesh.Scene):
    # Handle scene with multiple geometries
    all_points = []
    all_colors = []
    for geom in mesh.geometry.values():
        if isinstance(geom, trimesh.PointCloud):
            all_points.append(geom.vertices)
            if geom.colors is not None:
                all_colors.append(geom.colors[:, :3])
    points = np.vstack(all_points)
    colors = np.vstack(all_colors) if all_colors else None
else:
    raise ValueError(f"Unexpected geometry type: {type(mesh)}")

print(f"✅ Loaded {len(points):,} points")
print(f"   Has colors: {colors is not None}")

# Create Open3D point cloud
pcd_original = o3d.geometry.PointCloud()
pcd_original.points = o3d.utility.Vector3dVector(points)

if colors is not None:
    # Normalize colors to [0, 1] if needed
    if colors.max() > 1.0:
        colors = colors.astype(np.float64) / 255.0
    pcd_original.colors = o3d.utility.Vector3dVector(colors)

# Print statistics
print(f"\n📊 Original Point Cloud Statistics:")
print(f"   - Number of points: {len(points):,}")
print(f"   - Bounding box:")
print(f"     - X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
print(f"     - Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
print(f"     - Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
center = points.mean(axis=0)
print(f"   - Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")


# %% BLOCK 3: Quick Downsample for Fast Preview (5% of points)
print("\n" + "="*80)
print("QUICK DOWNSAMPLE FOR PREVIEW")
print("="*80)

print(f"\n🔽 Random downsampling to {QUICK_DOWNSAMPLE_RATIO*100:.1f}% of original...")
print(f"   Original: {len(pcd_original.points):,} points")

# Random sampling
num_points = len(pcd_original.points)
num_sample = int(num_points * QUICK_DOWNSAMPLE_RATIO)
indices = np.random.choice(num_points, num_sample, replace=False)
pcd_downsampled = pcd_original.select_by_index(indices)

print(f"   Downsampled: {len(pcd_downsampled.points):,} points")
print(f"   Reduction: {(1 - QUICK_DOWNSAMPLE_RATIO)*100:.1f}%")

# This becomes our working point cloud for fast iteration
pcd = pcd_downsampled


# %% BLOCK 3.5: Align Point Cloud to Ground Plane
print("\n" + "="*80)
print("ALIGNING TO GROUND PLANE")
print("="*80)

print(f"\n🔧 Detecting ground plane using RANSAC...")

# Use RANSAC to find the dominant plane (assumed to be the ground)
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01,  # Points within 1cm of plane
    ransac_n=3,
    num_iterations=1000
)

[a, b, c, d] = plane_model
print(f"   Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
print(f"   Found {len(inliers)} inlier points ({len(inliers)/len(pcd.points)*100:.1f}%)")

# Extract plane normal
plane_normal = np.array([a, b, c])
plane_normal_normalized = plane_normal / np.linalg.norm(plane_normal)
print(f"   Plane normal (normalized): [{plane_normal_normalized[0]:.3f}, {plane_normal_normalized[1]:.3f}, {plane_normal_normalized[2]:.3f}]")

# Get a point on the plane (using the origin projection)
# For plane ax + by + cz + d = 0, closest point to origin is -d * normal / ||normal||^2
plane_point = -d * plane_normal / np.dot(plane_normal, plane_normal)
print(f"   Point on plane: [{plane_point[0]:.3f}, {plane_point[1]:.3f}, {plane_point[2]:.3f}]")

# We want the ground plane to be XY (normal should point to +Z)
# Target normal is [0, 0, 1]
target_normal = np.array([0, 0, 1])

# If the plane normal points downward, flip it
if np.dot(plane_normal_normalized, target_normal) < 0:
    plane_normal_normalized = -plane_normal_normalized
    print(f"   Flipped normal: [{plane_normal_normalized[0]:.3f}, {plane_normal_normalized[1]:.3f}, {plane_normal_normalized[2]:.3f}]")

# Compute rotation matrix to align plane_normal with target_normal
# Using Rodrigues' rotation formula
v = np.cross(plane_normal_normalized, target_normal)
s = np.linalg.norm(v)
c = np.dot(plane_normal_normalized, target_normal)

if s < 1e-6:
    # Vectors are already aligned
    R = np.eye(3)
    print("   Plane is already aligned with XY")
else:
    # Skew-symmetric cross-product matrix
    v_x = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.eye(3) + v_x + np.dot(v_x, v_x) * ((1 - c) / (s * s))
    print("   Computed rotation matrix")

print(f"\n📐 Rotation matrix R:")
print(f"   [{R[0,0]:7.4f}, {R[0,1]:7.4f}, {R[0,2]:7.4f}]")
print(f"   [{R[1,0]:7.4f}, {R[1,1]:7.4f}, {R[1,2]:7.4f}]")
print(f"   [{R[2,0]:7.4f}, {R[2,1]:7.4f}, {R[2,2]:7.4f}]")

print(f"\n🔄 Applying rotation and translation to align ground plane with XY...")

# Apply rotation to point cloud
pcd.rotate(R, center=(0, 0, 0))

# Get the rotated points to determine which side is "up"
points_rotated = np.asarray(pcd.points)

# Calculate distances from each point to the plane after rotation
# After rotation, the plane normal is [0, 0, 1], so distance is just the Z coordinate
# relative to the plane's Z position
plane_point_rotated = R @ plane_point
plane_z = plane_point_rotated[2]

# Distances of all points from the plane along the normal direction
distances = points_rotated[:, 2] - plane_z

# Determine which side has more "elongation" (higher variance in distances)
# Split points into two groups: above and below the plane
above_plane = distances > 0
below_plane = distances < 0

distances_above = distances[above_plane]
distances_below = np.abs(distances[below_plane])

# Calculate statistics for each side
if len(distances_above) > 0:
    mean_above = distances_above.mean()
    max_above = distances_above.max()
    std_above = distances_above.std()
    elongation_above = max_above  # Use max distance as measure of elongation
else:
    mean_above = 0
    max_above = 0
    std_above = 0
    elongation_above = 0

if len(distances_below) > 0:
    mean_below = distances_below.mean()
    max_below = distances_below.max()
    std_below = distances_below.std()
    elongation_below = max_below
else:
    mean_below = 0
    max_below = 0
    std_below = 0
    elongation_below = 0

print(f"\n📊 Analyzing point distribution relative to plane:")
print(f"   Points above plane: {above_plane.sum()} ({above_plane.sum()/len(points_rotated)*100:.1f}%)")
print(f"   - Mean distance: {mean_above:.3f}")
print(f"   - Max distance: {max_above:.3f}")
print(f"   - Std dev: {std_above:.3f}")
print(f"   - Elongation: {elongation_above:.3f}")
print(f"")
print(f"   Points below plane: {below_plane.sum()} ({below_plane.sum()/len(points_rotated)*100:.1f}%)")
print(f"   - Mean distance: {mean_below:.3f}")
print(f"   - Max distance: {max_below:.3f}")
print(f"   - Std dev: {std_below:.3f}")
print(f"   - Elongation: {elongation_below:.3f}")

# If the "below" side has more elongation, flip the Z axis
if elongation_below > elongation_above:
    print(f"\n🔃 Flipping Z axis: objects are on the negative side of the plane")
    # Flip Z axis by applying 180-degree rotation around X axis
    R_flip = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    pcd.rotate(R_flip, center=(0, 0, 0))
    points_rotated = np.asarray(pcd.points)
    # Also flip the plane point
    plane_point_rotated = R_flip @ plane_point_rotated
    plane_z = plane_point_rotated[2]
else:
    print(f"\n✅ Z axis is correctly oriented: objects are on the positive side")

# Translate so the plane is at Z=0
translation = np.array([0, 0, -plane_z])
pcd.translate(translation)
points_rotated = np.asarray(pcd.points)

# Recalculate distances after potential flip
distances_final = points_rotated[:, 2]

print(f"\n📏 After orientation:")
print(f"   Ground plane is at Z = 0")
print(f"   Points range:")
print(f"   - Z min: {points_rotated[:, 2].min():.3f}")
print(f"   - Z max: {points_rotated[:, 2].max():.3f}")
print(f"   - Z mean: {points_rotated[:, 2].mean():.3f}")

# Verify that most points are now above Z=0
points_above_zero = (points_rotated[:, 2] > 0).sum()
print(f"   Points above Z=0: {points_above_zero} ({points_above_zero/len(points_rotated)*100:.1f}%)")

# Print new statistics
points = points_rotated
print(f"\n📊 Aligned Point Cloud:")
print(f"   - Bounding box:")
print(f"     - X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
print(f"     - Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
print(f"     - Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

# Also apply the same transformation to full point clouds
print(f"\n🔄 Applying same transformation to full resolution point clouds...")
pcd_original.rotate(R, center=(0, 0, 0))
if elongation_below > elongation_above:
    pcd_original.rotate(R_flip, center=(0, 0, 0))
pcd_original.translate(translation)

pcd_downsampled.rotate(R, center=(0, 0, 0))
if elongation_below > elongation_above:
    pcd_downsampled.rotate(R_flip, center=(0, 0, 0))
pcd_downsampled.translate(translation)


# %% BLOCK 4: Visualize Current Point Cloud (Elevated 45° Views)
def visualize_point_cloud_elevated(pcd, title="Point Cloud", output_path=None, point_size=1):
    """
    Visualize point cloud from 4 elevated viewpoints at 45 degrees from above.

    Creates 4 views looking down at 45 degrees from North, East, South, and West.

    Args:
        pcd: Open3D point cloud (should be aligned so Z is up)
        title: Title for the visualization
        output_path: Path to save the figure (optional)
        point_size: Size of points in scatter plot
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # Center point cloud at origin for rotation
    center = points.mean(axis=0)
    points_centered = points - center

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(18, 18))

    # Define 4 viewpoints: 0°, 90°, 180°, 270° around Z axis, elevated 45° from horizontal
    elevation = 45  # degrees
    azimuths = [0, 90, 180, 270]
    view_names = ['North (0°)', 'East (90°)', 'South (180°)', 'West (270°)']

    for idx, (azimuth, view_name) in enumerate(zip(azimuths, view_names)):
        ax = fig.add_subplot(2, 2, idx + 1)

        # Convert angles to radians
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)

        # Camera direction (looking toward origin)
        # In spherical coordinates: (azimuth, elevation)
        cam_x = np.cos(elevation_rad) * np.sin(azimuth_rad)
        cam_y = np.cos(elevation_rad) * np.cos(azimuth_rad)
        cam_z = np.sin(elevation_rad)

        # Create viewing transformation
        # We'll project points onto a plane perpendicular to the camera direction

        # Define "up" direction for the camera (tilted Z direction)
        up = np.array([0, 0, 1])

        # Camera forward direction (toward scene)
        forward = -np.array([cam_x, cam_y, cam_z])
        forward = forward / np.linalg.norm(forward)

        # Camera right direction
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        # Camera up direction (perpendicular to forward and right)
        cam_up = np.cross(right, forward)
        cam_up = cam_up / np.linalg.norm(cam_up)

        # Project points onto camera's view plane
        # right direction becomes X axis, cam_up becomes Y axis
        x_proj = np.dot(points_centered, right)
        y_proj = np.dot(points_centered, cam_up)

        # Plot the projected points
        if colors is not None:
            # Sort by depth for proper occlusion
            depth = np.dot(points_centered, forward)
            sort_idx = np.argsort(-depth)  # Back to front

            ax.scatter(x_proj[sort_idx], y_proj[sort_idx],
                      c=colors[sort_idx], s=point_size, alpha=0.6)
        else:
            depth = np.dot(points_centered, forward)
            sort_idx = np.argsort(-depth)
            ax.scatter(x_proj[sort_idx], y_proj[sort_idx],
                      c=depth[sort_idx], s=point_size, alpha=0.6, cmap='viridis')

        ax.set_xlabel('Horizontal')
        ax.set_ylabel('Vertical')
        ax.set_title(f'{view_name}\n(45° elevation)', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

    # Overall title
    fig.suptitle(f'{title}\n{len(points):,} points - Elevated 45° Views',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved visualization to {output_path}")

    plt.close()


def visualize_point_cloud_2d(pcd, title="Point Cloud", output_path=None, point_size=1):
    """
    Visualize point cloud as 2D projections from 4 different angles.

    Args:
        pcd: Open3D point cloud
        title: Title for the visualization
        output_path: Path to save the figure (optional)
        point_size: Size of points in scatter plot
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))

    # View 1: Top view (XY plane, looking down Z axis)
    ax1 = fig.add_subplot(2, 2, 1)
    if colors is not None:
        ax1.scatter(points[:, 0], points[:, 1], c=colors, s=point_size, alpha=0.6)
    else:
        ax1.scatter(points[:, 0], points[:, 1], s=point_size, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Top View (XY plane)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # View 2: Front view (XZ plane, looking along Y axis)
    ax2 = fig.add_subplot(2, 2, 2)
    if colors is not None:
        ax2.scatter(points[:, 0], points[:, 2], c=colors, s=point_size, alpha=0.6)
    else:
        ax2.scatter(points[:, 0], points[:, 2], s=point_size, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('Front View (XZ plane)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # View 3: Side view (YZ plane, looking along X axis)
    ax3 = fig.add_subplot(2, 2, 3)
    if colors is not None:
        ax3.scatter(points[:, 1], points[:, 2], c=colors, s=point_size, alpha=0.6)
    else:
        ax3.scatter(points[:, 1], points[:, 2], s=point_size, alpha=0.6)
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_title('Side View (YZ plane)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # View 4: 3D-like isometric view (XY plane with Z as color intensity)
    ax4 = fig.add_subplot(2, 2, 4)
    if colors is not None:
        # Use depth (Z) to modulate color intensity
        z_normalized = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())
        colors_with_depth = colors * z_normalized[:, np.newaxis]
        ax4.scatter(points[:, 0], points[:, 1], c=colors_with_depth, s=point_size, alpha=0.6)
    else:
        # Use Z as color
        ax4.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=point_size, alpha=0.6, cmap='viridis')
        plt.colorbar(ax4.collections[0], ax=ax4, label='Z depth')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('Top View with Depth (darker = further)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'{title}\n{len(points):,} points', fontsize=16, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved visualization to {output_path}")

    plt.close()


print("\n" + "="*80)
print("VISUALIZING POINT CLOUD")
print("="*80)

print(f"\n📊 Creating elevated 45° views from 4 angles ({len(pcd.points):,} points)...")
output_path = os.path.join(VIS_OUTPUT_DIR, "01_initial_downsampled.png")
visualize_point_cloud_elevated(pcd, title="Initial Downsampled & Aligned Point Cloud (5%)", output_path=output_path)


# %% BLOCK 5: Statistical Outlier Removal
print("\n" + "="*80)
print("STATISTICAL OUTLIER REMOVAL")
print("="*80)

# Parameters
nb_neighbors = 20  # Number of neighbors to analyze
std_ratio = 2.0    # Standard deviation multiplier threshold

print(f"\n🧹 Removing statistical outliers...")
print(f"   Parameters: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
print(f"   Before: {len(pcd.points):,} points")

pcd_cleaned, ind = pcd.remove_statistical_outlier(
    nb_neighbors=nb_neighbors,
    std_ratio=std_ratio
)

print(f"   After: {len(pcd_cleaned.points):,} points")
print(f"   Removed: {len(pcd.points) - len(pcd_cleaned.points):,} points")

# Update working point cloud
pcd = pcd_cleaned

# Visualize result
print(f"\n📊 Visualizing after statistical outlier removal...")
output_path = os.path.join(VIS_OUTPUT_DIR, "02_after_statistical.png")
visualize_point_cloud_elevated(pcd, title="After Statistical Outlier Removal", output_path=output_path)


# %% BLOCK 6: Radius Outlier Removal
print("\n" + "="*80)
print("RADIUS OUTLIER REMOVAL")
print("="*80)

# Parameters
nb_points = 16   # Minimum neighbors within radius
radius = 0.05    # Search radius

print(f"\n🧹 Removing radius outliers...")
print(f"   Parameters: nb_points={nb_points}, radius={radius}")
print(f"   Before: {len(pcd.points):,} points")

pcd_cleaned, ind = pcd.remove_radius_outlier(
    nb_points=nb_points,
    radius=radius
)

print(f"   After: {len(pcd_cleaned.points):,} points")
print(f"   Removed: {len(pcd.points) - len(pcd_cleaned.points):,} points")

# Update working point cloud
pcd = pcd_cleaned

# Visualize result
print(f"\n📊 Visualizing after radius outlier removal...")
output_path = os.path.join(VIS_OUTPUT_DIR, "03_after_radius.png")
visualize_point_cloud_elevated(pcd, title="After Radius Outlier Removal", output_path=output_path)


# %% BLOCK 7: Cluster-Based Filtering (Keep Largest Clusters)
print("\n" + "="*80)
print("CLUSTER-BASED FILTERING")
print("="*80)

# Parameters
eps = 0.1          # DBSCAN distance threshold
min_points = 100   # Minimum points to form a cluster
keep_top_n = 3     # Number of largest clusters to keep

print(f"\n🧹 Filtering by clusters...")
print(f"   Parameters: eps={eps}, min_points={min_points}, keep_top_n={keep_top_n}")
print(f"   Before: {len(pcd.points):,} points")

# Perform DBSCAN clustering
labels = np.array(pcd.cluster_dbscan(
    eps=eps,
    min_points=min_points,
    print_progress=False
))

# Count points in each cluster (label -1 is noise)
unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

if len(unique_labels) > 0:
    print(f"   Found {len(unique_labels)} clusters")

    # Sort clusters by size
    sorted_indices = np.argsort(-counts)  # Descending order
    top_labels = unique_labels[sorted_indices[:keep_top_n]]

    # Keep only points in top clusters
    mask = np.isin(labels, top_labels)
    pcd_cleaned = pcd.select_by_index(np.where(mask)[0])

    print(f"   Kept top {len(top_labels)} cluster(s) with sizes: {counts[sorted_indices[:keep_top_n]].tolist()}")
    print(f"   After: {len(pcd_cleaned.points):,} points")
    print(f"   Removed: {len(pcd.points) - len(pcd_cleaned.points):,} points")

    # Update working point cloud
    pcd = pcd_cleaned
else:
    print("   ⚠️ No clusters found, keeping original point cloud")

# Visualize result
print(f"\n📊 Visualizing after cluster filtering...")
output_path = os.path.join(VIS_OUTPUT_DIR, "04_after_cluster.png")
visualize_point_cloud_elevated(pcd, title="After Cluster-Based Filtering", output_path=output_path)


# %% BLOCK 8: Voxel Downsampling (Further Smoothing)
print("\n" + "="*80)
print("VOXEL DOWNSAMPLING")
print("="*80)

# Parameters
voxel_size = 0.02  # Size of voxel grid cells

print(f"\n🧹 Voxel downsampling...")
print(f"   Parameters: voxel_size={voxel_size}")
print(f"   Before: {len(pcd.points):,} points")

pcd_cleaned = pcd.voxel_down_sample(voxel_size=voxel_size)

print(f"   After: {len(pcd_cleaned.points):,} points")
print(f"   Removed: {len(pcd.points) - len(pcd_cleaned.points):,} points")

# Update working point cloud
pcd = pcd_cleaned

# Visualize result
print(f"\n📊 Visualizing after voxel downsampling...")
output_path = os.path.join(VIS_OUTPUT_DIR, "05_after_voxel.png")
visualize_point_cloud_elevated(pcd, title="After Voxel Downsampling (Final Result)", output_path=output_path)


# %% BLOCK 9: Compare Original vs Cleaned
print("\n" + "="*80)
print("COMPARISON: ORIGINAL VS CLEANED")
print("="*80)

print(f"\n📊 Statistics:")
print(f"   Original (full):      {len(pcd_original.points):,} points")
print(f"   Downsampled (5%):     {len(pcd_downsampled.points):,} points")
print(f"   After cleanup:        {len(pcd.points):,} points")
print(f"   Total reduction:      {(1 - len(pcd.points)/len(pcd_downsampled.points))*100:.1f}%")

# Create side-by-side comparison visualization
print(f"\n📊 Creating comparison visualization...")
points_down = np.asarray(pcd_downsampled.points)
colors_down = np.asarray(pcd_downsampled.colors) if pcd_downsampled.has_colors() else None

points_clean = np.asarray(pcd.points)
colors_clean = np.asarray(pcd.colors) if pcd.has_colors() else None

fig = plt.figure(figsize=(20, 10))

# Downsampled - Top view
ax1 = fig.add_subplot(2, 4, 1)
if colors_down is not None:
    ax1.scatter(points_down[:, 0], points_down[:, 1], c=colors_down, s=1, alpha=0.6)
else:
    ax1.scatter(points_down[:, 0], points_down[:, 1], s=1, alpha=0.6)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title(f'Downsampled - Top View\n{len(points_down):,} points')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# Downsampled - Front view
ax2 = fig.add_subplot(2, 4, 2)
if colors_down is not None:
    ax2.scatter(points_down[:, 0], points_down[:, 2], c=colors_down, s=1, alpha=0.6)
else:
    ax2.scatter(points_down[:, 0], points_down[:, 2], s=1, alpha=0.6)
ax2.set_xlabel('X')
ax2.set_ylabel('Z')
ax2.set_title('Downsampled - Front View')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# Downsampled - Side view
ax3 = fig.add_subplot(2, 4, 3)
if colors_down is not None:
    ax3.scatter(points_down[:, 1], points_down[:, 2], c=colors_down, s=1, alpha=0.6)
else:
    ax3.scatter(points_down[:, 1], points_down[:, 2], s=1, alpha=0.6)
ax3.set_xlabel('Y')
ax3.set_ylabel('Z')
ax3.set_title('Downsampled - Side View')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

# Downsampled - Depth view
ax4 = fig.add_subplot(2, 4, 4)
if colors_down is not None:
    z_norm = (points_down[:, 2] - points_down[:, 2].min()) / (points_down[:, 2].max() - points_down[:, 2].min())
    colors_depth = colors_down * z_norm[:, np.newaxis]
    ax4.scatter(points_down[:, 0], points_down[:, 1], c=colors_depth, s=1, alpha=0.6)
else:
    ax4.scatter(points_down[:, 0], points_down[:, 1], c=points_down[:, 2], s=1, alpha=0.6, cmap='viridis')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_title('Downsampled - Depth View')
ax4.set_aspect('equal')
ax4.grid(True, alpha=0.3)

# Cleaned - Top view
ax5 = fig.add_subplot(2, 4, 5)
if colors_clean is not None:
    ax5.scatter(points_clean[:, 0], points_clean[:, 1], c=colors_clean, s=1, alpha=0.6)
else:
    ax5.scatter(points_clean[:, 0], points_clean[:, 1], s=1, alpha=0.6)
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_title(f'Cleaned - Top View\n{len(points_clean):,} points')
ax5.set_aspect('equal')
ax5.grid(True, alpha=0.3)

# Cleaned - Front view
ax6 = fig.add_subplot(2, 4, 6)
if colors_clean is not None:
    ax6.scatter(points_clean[:, 0], points_clean[:, 2], c=colors_clean, s=1, alpha=0.6)
else:
    ax6.scatter(points_clean[:, 0], points_clean[:, 2], s=1, alpha=0.6)
ax6.set_xlabel('X')
ax6.set_ylabel('Z')
ax6.set_title('Cleaned - Front View')
ax6.set_aspect('equal')
ax6.grid(True, alpha=0.3)

# Cleaned - Side view
ax7 = fig.add_subplot(2, 4, 7)
if colors_clean is not None:
    ax7.scatter(points_clean[:, 1], points_clean[:, 2], c=colors_clean, s=1, alpha=0.6)
else:
    ax7.scatter(points_clean[:, 1], points_clean[:, 2], s=1, alpha=0.6)
ax7.set_xlabel('Y')
ax7.set_ylabel('Z')
ax7.set_title('Cleaned - Side View')
ax7.set_aspect('equal')
ax7.grid(True, alpha=0.3)

# Cleaned - Depth view
ax8 = fig.add_subplot(2, 4, 8)
if colors_clean is not None:
    z_norm_clean = (points_clean[:, 2] - points_clean[:, 2].min()) / (points_clean[:, 2].max() - points_clean[:, 2].min())
    colors_depth_clean = colors_clean * z_norm_clean[:, np.newaxis]
    ax8.scatter(points_clean[:, 0], points_clean[:, 1], c=colors_depth_clean, s=1, alpha=0.6)
else:
    ax8.scatter(points_clean[:, 0], points_clean[:, 1], c=points_clean[:, 2], s=1, alpha=0.6, cmap='viridis')
ax8.set_xlabel('X')
ax8.set_ylabel('Y')
ax8.set_title('Cleaned - Depth View')
ax8.set_aspect('equal')
ax8.grid(True, alpha=0.3)

fig.suptitle('Comparison: Downsampled (top row) vs Cleaned (bottom row)', fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = os.path.join(VIS_OUTPUT_DIR, "06_comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   💾 Saved comparison to {output_path}")

plt.close()


# %% BLOCK 10: Save Cleaned Point Cloud
print("\n" + "="*80)
print("SAVING CLEANED POINT CLOUD")
print("="*80)

print(f"\n💾 Saving to {OUTPUT_PATH}...")

# Convert Open3D point cloud to trimesh
points_out = np.asarray(pcd.points)
colors_out = np.asarray(pcd.colors) if pcd.has_colors() else None

if colors_out is not None:
    # Ensure colors are in [0, 255] range for trimesh
    if colors_out.max() <= 1.0:
        colors_out = (colors_out * 255).astype(np.uint8)

# Create trimesh point cloud
point_cloud_out = trimesh.points.PointCloud(
    vertices=points_out,
    colors=colors_out
)

# Save as .glb
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
point_cloud_out.export(OUTPUT_PATH)

print(f"✅ Saved {len(points_out):,} points to {OUTPUT_PATH}")
print(f"\n🎉 Processing complete!")


# %% BLOCK 11: Apply Cleanup to Full Resolution (Optional)
# Uncomment this block to apply the same cleanup pipeline to the full resolution point cloud
# WARNING: This will be slower but produces the final high-quality result

"""
print("\n" + "="*80)
print("APPLYING CLEANUP TO FULL RESOLUTION")
print("="*80)

print("\n⚠️  Processing full resolution point cloud...")
print(f"   This may take several minutes for {len(pcd_original.points):,} points")

# Use the original full resolution point cloud
pcd_full = pcd_original

# Apply the same cleanup pipeline
print("\n1️⃣ Statistical outlier removal...")
pcd_full, _ = pcd_full.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"   Points: {len(pcd_full.points):,}")

print("\n2️⃣ Radius outlier removal...")
pcd_full, _ = pcd_full.remove_radius_outlier(nb_points=16, radius=0.05)
print(f"   Points: {len(pcd_full.points):,}")

print("\n3️⃣ Cluster-based filtering...")
labels_full = np.array(pcd_full.cluster_dbscan(eps=0.1, min_points=100, print_progress=True))
unique_labels_full, counts_full = np.unique(labels_full[labels_full >= 0], return_counts=True)
if len(unique_labels_full) > 0:
    sorted_indices_full = np.argsort(-counts_full)
    top_labels_full = unique_labels_full[sorted_indices_full[:3]]
    mask_full = np.isin(labels_full, top_labels_full)
    pcd_full = pcd_full.select_by_index(np.where(mask_full)[0])
print(f"   Points: {len(pcd_full.points):,}")

print("\n4️⃣ Voxel downsampling...")
pcd_full = pcd_full.voxel_down_sample(voxel_size=0.02)
print(f"   Points: {len(pcd_full.points):,}")

# Save full resolution cleaned point cloud
output_full = OUTPUT_PATH.replace(".glb", "_full.glb")
print(f"\n💾 Saving full resolution to {output_full}...")

points_full_out = np.asarray(pcd_full.points)
colors_full_out = np.asarray(pcd_full.colors) if pcd_full.has_colors() else None

if colors_full_out is not None:
    if colors_full_out.max() <= 1.0:
        colors_full_out = (colors_full_out * 255).astype(np.uint8)

point_cloud_full_out = trimesh.points.PointCloud(
    vertices=points_full_out,
    colors=colors_full_out
)
point_cloud_full_out.export(output_full)

print(f"✅ Saved {len(points_full_out):,} points to {output_full}")
print(f"🎉 Full resolution processing complete!")
"""
