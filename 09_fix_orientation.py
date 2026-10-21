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
Point Cloud Orientation Fixer - Interactive Style

This script loads a point cloud and allows interactive inspection
and orientation fixing using PyVista.

Workflow:
1. Load the cleaned point cloud
2. Visualize with PyVista interactively
3. Fit ground plane using various methods
4. Save corrected orientation

Usage:
    Run blocks individually by uncommenting and executing
"""

# %% BLOCK 1: Imports and Setup
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import pyvista as pv
import trimesh

# Configuration
INPUT_PATH = "data/outputs/semantic_3d/point_cloud_cleaned.glb"
OUTPUT_PATH = "data/outputs/semantic_3d/point_cloud_oriented.glb"

print(" Imports loaded")
print(f"Input: {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")


# %% BLOCK 2: Load Point Cloud
print("\n" + "="*80)
print("LOADING POINT CLOUD")
print("="*80)

# Load using trimesh
print(f"\n=� Loading from {INPUT_PATH}...")
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

print(f" Loaded {len(points):,} points")
print(f"   Has colors: {colors is not None}")

# Create Open3D point cloud for processing
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

if colors is not None:
    # Normalize colors to [0, 1] if needed
    if colors.max() > 1.0:
        colors = colors.astype(np.float64) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

# Print statistics
print(f"\n=� Point Cloud Statistics:")
print(f"   - Number of points: {len(points):,}")
print(f"   - Bounding box:")
print(f"     - X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
print(f"     - Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
print(f"     - Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
center = points.mean(axis=0)
print(f"   - Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")


# %% BLOCK 3: Visualize with PyVista (Interactive)
print("\n" + "="*80)
print("INTERACTIVE PYVISTA VISUALIZATION")
print("="*80)

print(f"\n<� Creating PyVista point cloud...")

# Create PyVista point cloud
pv_cloud = pv.PolyData(points)

# Add colors if available
if colors is not None:
    # PyVista expects RGB colors in [0, 255] range
    if colors.max() <= 1.0:
        colors_255 = (colors * 255).astype(np.uint8)
    else:
        colors_255 = colors.astype(np.uint8)
    pv_cloud["RGB"] = colors_255

print(f" Created PyVista point cloud with {pv_cloud.n_points:,} points")

# Create plotter
print(f"\n=A  Opening interactive 3D viewer...")
print("   Controls:")
print("   - Left Click + Drag: Rotate")
print("   - Middle Click + Drag: Pan")
print("   - Right Click + Drag: Zoom")
print("   - Scroll: Zoom")
print("   - 'q': Close window")
print("   - 'r': Reset camera")
print("   - 's': Save screenshot")

plotter = pv.Plotter()

# Add point cloud
if colors is not None:
    plotter.add_points(
        pv_cloud,
        rgb=True,
        point_size=5,
        render_points_as_spheres=True
    )
else:
    plotter.add_points(
        pv_cloud,
        color='white',
        point_size=5,
        render_points_as_spheres=True
    )

# Add coordinate axes
plotter.add_axes(
    xlabel='X',
    ylabel='Y',
    zlabel='Z',
    line_width=3,
    cone_radius=0.4
)

# Add grid for reference
plotter.show_grid()

# Set camera position (slightly elevated view)
plotter.camera_position = 'xy'  # Top-down view initially

# Show the plot
print(f"\n=� Launching viewer...")
plotter.show()

print(f" Viewer closed")

# %%

# %% BLOCK 4: Fit Ground Plane using RANSAC
print("\n" + "="*80)
print("GROUND PLANE DETECTION")
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

# %% BLOCK 5: Visualize with Plane and Coordinate Frames
print("\n" + "="*80)
print("VISUALIZING WITH PLANE AND AXES")
print("="*80)

print(f"\n🎨 Creating visualization with fitted plane and coordinate frames...")

# Create PyVista plotter
try:
    plotter2 = pv.Plotter(window_size=[1600, 1200])
except:
    print("   ⚠️  No display detected, using off-screen rendering")
    plotter2 = pv.Plotter(off_screen=True, window_size=[1600, 1200])

# Add original point cloud
pv_cloud2 = pv.PolyData(points)
if colors is not None:
    if colors.max() <= 1.0:
        colors_255_2 = (colors * 255).astype(np.uint8)
    else:
        colors_255_2 = colors.astype(np.uint8)
    pv_cloud2["RGB"] = colors_255_2
    plotter2.add_points(
        pv_cloud2,
        rgb=True,
        point_size=3,
        render_points_as_spheres=True,
        opacity=0.8
    )
else:
    plotter2.add_points(
        pv_cloud2,
        color='lightgray',
        point_size=3,
        render_points_as_spheres=True,
        opacity=0.8
    )

# Extract inlier points (points on the plane)
inlier_points = points[inliers]
pv_inliers = pv.PolyData(inlier_points)
plotter2.add_points(
    pv_inliers,
    color='red',
    point_size=5,
    render_points_as_spheres=True,
    opacity=0.5
)

# Create a plane mesh for visualization
# Get bounding box to size the plane appropriately
bounds = pv_cloud2.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
size = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 1.2

# Create a plane mesh
plane_center = plane_point
plane_mesh = pv.Plane(
    center=plane_center,
    direction=plane_normal_normalized,
    i_size=size,
    j_size=size,
    i_resolution=2,
    j_resolution=2
)

plotter2.add_mesh(
    plane_mesh,
    color='yellow',
    opacity=0.3,
    show_edges=True,
    edge_color='orange',
    line_width=2
)

# Add CURRENT coordinate frame (XYZ in original orientation)
print(f"\n📍 Adding coordinate frames...")

# Origin point (centroid of point cloud)
origin = points.mean(axis=0)

# Current coordinate axes (before rotation)
axis_length = size * 0.3
current_x_axis = np.array([axis_length, 0, 0])
current_y_axis = np.array([0, axis_length, 0])
current_z_axis = np.array([0, 0, axis_length])

# X axis (red)
x_arrow = pv.Arrow(
    start=origin,
    direction=current_x_axis / axis_length,
    scale=axis_length,
    shaft_radius=axis_length * 0.01,
    tip_length=0.2,
    tip_radius=0.04
)
plotter2.add_mesh(x_arrow, color='red', label='Current X')

# Y axis (green)
y_arrow = pv.Arrow(
    start=origin,
    direction=current_y_axis / axis_length,
    scale=axis_length,
    shaft_radius=axis_length * 0.01,
    tip_length=0.2,
    tip_radius=0.04
)
plotter2.add_mesh(y_arrow, color='green', label='Current Y')

# Z axis (blue)
z_arrow = pv.Arrow(
    start=origin,
    direction=current_z_axis / axis_length,
    scale=axis_length,
    shaft_radius=axis_length * 0.01,
    tip_length=0.2,
    tip_radius=0.04
)
plotter2.add_mesh(z_arrow, color='blue', label='Current Z')

# Add TARGET coordinate frame (where we want it to be after rotation)
# This shows where X, Y, Z will point after aligning plane to ground
target_x_axis = R @ current_x_axis
target_y_axis = R @ current_y_axis
target_z_axis = R @ current_z_axis

# Target X axis (magenta/pink)
tx_arrow = pv.Arrow(
    start=origin,
    direction=target_x_axis / axis_length,
    scale=axis_length,
    shaft_radius=axis_length * 0.008,
    tip_length=0.2,
    tip_radius=0.03
)
plotter2.add_mesh(tx_arrow, color='magenta', opacity=0.6, label='Target X')

# Target Y axis (cyan)
ty_arrow = pv.Arrow(
    start=origin,
    direction=target_y_axis / axis_length,
    scale=axis_length,
    shaft_radius=axis_length * 0.008,
    tip_length=0.2,
    tip_radius=0.03
)
plotter2.add_mesh(ty_arrow, color='cyan', opacity=0.6, label='Target Y')

# Target Z axis (purple) - should point UP after rotation
tz_arrow = pv.Arrow(
    start=origin,
    direction=target_z_axis / axis_length,
    scale=axis_length,
    shaft_radius=axis_length * 0.008,
    tip_length=0.2,
    tip_radius=0.03
)
plotter2.add_mesh(tz_arrow, color='purple', opacity=0.6, label='Target Z (should be UP)')

# Add plane normal vector
normal_arrow = pv.Arrow(
    start=plane_center,
    direction=plane_normal_normalized,
    scale=axis_length * 0.5,
    shaft_radius=axis_length * 0.015,
    tip_length=0.25,
    tip_radius=0.05
)
plotter2.add_mesh(normal_arrow, color='yellow', label='Plane Normal')

# Add title
plotter2.add_text(
    f"Ground Plane Detection\n"
    f"Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0\n"
    f"Inliers: {len(inliers)} / {len(points)} ({len(inliers)/len(points)*100:.1f}%)\n"
    f"Solid arrows = Current orientation\n"
    f"Translucent arrows = Target orientation",
    font_size=10,
    position='upper_left'
)

plotter2.show_grid()
plotter2.show()

# Set initial camera position
plotter2.camera_position = [(2, 2, 1.5), origin, (0, 0, 1)]

# Save screenshots
os.makedirs("data/outputs/semantic_3d/plane_fitting", exist_ok=True)

# View 1: Default view
plotter2.screenshot("data/outputs/semantic_3d/plane_fitting/plane_default.png")
print(f"   📸 Saved default view")

# View 2: Top-down view
plotter2.camera_position = [(origin[0], origin[1], origin[2] + size), origin, (0, 1, 0)]
plotter2.screenshot("data/outputs/semantic_3d/plane_fitting/plane_top.png")
print(f"   📸 Saved top view")

# View 3: Side view
plotter2.camera_position = [(origin[0] + size, origin[1], origin[2]), origin, (0, 0, 1)]
plotter2.screenshot("data/outputs/semantic_3d/plane_fitting/plane_side.png")
print(f"   📸 Saved side view")

# View 4: Front view
plotter2.camera_position = [(origin[0], origin[1] + size, origin[2]), origin, (0, 0, 1)]
plotter2.screenshot("data/outputs/semantic_3d/plane_fitting/plane_front.png")
print(f"   📸 Saved front view")

# Try to show interactive window
print(f"\n👁️  Opening interactive viewer...")
print("   Legend:")
print("   - Yellow plane = Detected ground plane")
print("   - Red points = Inliers (points on the plane)")
print("   - RGB arrows (solid) = Current XYZ axes")
print("   - Magenta/Cyan/Purple arrows (translucent) = Target XYZ axes after rotation")
print("   - Yellow arrow from plane = Plane normal (should align with purple Target Z)")
print("")
print("   Verify that:")
print("   1. Yellow plane aligns with the ground")
print("   2. Purple Target Z arrow points upward")
print("   3. Plane normal (yellow) is parallel to Target Z (purple)")

try:
    plotter2.show()
    print(f"✅ Viewer closed")
except:
    print(f"✅ Screenshots saved (no display available for interactive mode)")

print(f"\n📁 Saved visualizations to: data/outputs/semantic_3d/plane_fitting/")

# %% BLOCK 6: Apply Transformation and Re-orient Point Cloud
print("\n" + "="*80)
print("APPLYING TRANSFORMATION TO ORIENT POINT CLOUD")
print("="*80)

print(f"\n🔄 Applying rotation and translation to align ground plane with XY...")

# Apply rotation to point cloud
pcd_oriented = o3d.geometry.PointCloud(pcd)
pcd_oriented.rotate(R, center=(0, 0, 0))

# Get the rotated points to determine which side is "up"
points_rotated = np.asarray(pcd_oriented.points)

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
    elongation_above = 0

if len(distances_below) > 0:
    mean_below = distances_below.mean()
    max_below = distances_below.max()
    std_below = distances_below.std()
    elongation_below = max_below
else:
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
    pcd_oriented.rotate(R_flip, center=(0, 0, 0))
    points_rotated = np.asarray(pcd_oriented.points)
    # Also flip the plane point
    plane_point_rotated = R_flip @ plane_point_rotated
    plane_z = plane_point_rotated[2]
else:
    print(f"\n✅ Z axis is correctly oriented: objects are on the positive side")

# Translate so the plane is at Z=0
translation = np.array([0, 0, -plane_z])
pcd_oriented.translate(translation)
points_rotated = np.asarray(pcd_oriented.points)

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

# %% BLOCK 7: Visualize Properly Oriented Point Cloud
print("\n" + "="*80)
print("VISUALIZING ORIENTED POINT CLOUD")
print("="*80)

print(f"\n🎨 Creating visualization of oriented point cloud...")

# Create PyVista plotter
try:
    plotter3 = pv.Plotter(window_size=[1600, 1200])
except:
    print("   ⚠️  No display detected, using off-screen rendering")
    plotter3 = pv.Plotter(off_screen=True, window_size=[1600, 1200])

# Add oriented point cloud
pv_cloud_oriented = pv.PolyData(points_rotated)
if colors is not None:
    if colors.max() <= 1.0:
        colors_255_3 = (colors * 255).astype(np.uint8)
    else:
        colors_255_3 = colors.astype(np.uint8)
    pv_cloud_oriented["RGB"] = colors_255_3
    plotter3.add_points(
        pv_cloud_oriented,
        rgb=True,
        point_size=5,
        render_points_as_spheres=True
    )
else:
    plotter3.add_points(
        pv_cloud_oriented,
        color='lightgray',
        point_size=5,
        render_points_as_spheres=True
    )

# Add ground plane at Z=0
bounds_oriented = pv_cloud_oriented.bounds
size_oriented = max(bounds_oriented[1] - bounds_oriented[0], bounds_oriented[3] - bounds_oriented[2]) * 1.2

ground_plane = pv.Plane(
    center=[0, 0, 0],
    direction=[0, 0, 1],  # Normal pointing up
    i_size=size_oriented,
    j_size=size_oriented,
    i_resolution=10,
    j_resolution=10
)

plotter3.add_mesh(
    ground_plane,
    color='lightgray',
    opacity=0.3,
    show_edges=True,
    edge_color='gray',
    line_width=1
)

# Add coordinate axes at origin
print(f"\n📍 Adding global coordinate axes...")

axis_length_global = size_oriented * 0.4

# X axis (RED) - points right
x_arrow_global = pv.Arrow(
    start=[0, 0, 0],
    direction=[1, 0, 0],
    scale=axis_length_global,
    shaft_radius=axis_length_global * 0.015,
    tip_length=0.2,
    tip_radius=0.05
)
plotter3.add_mesh(x_arrow_global, color='red', label='X (Right)')

# Y axis (GREEN) - points forward
y_arrow_global = pv.Arrow(
    start=[0, 0, 0],
    direction=[0, 1, 0],
    scale=axis_length_global,
    shaft_radius=axis_length_global * 0.015,
    tip_length=0.2,
    tip_radius=0.05
)
plotter3.add_mesh(y_arrow_global, color='green', label='Y (Forward)')

# Z axis (BLUE) - points up
z_arrow_global = pv.Arrow(
    start=[0, 0, 0],
    direction=[0, 0, 1],
    scale=axis_length_global,
    shaft_radius=axis_length_global * 0.015,
    tip_length=0.2,
    tip_radius=0.05
)
plotter3.add_mesh(z_arrow_global, color='blue', label='Z (Up)')

# Add text labels
plotter3.add_point_labels(
    [[axis_length_global * 1.1, 0, 0]],
    ['X'],
    font_size=24,
    text_color='red',
    bold=True
)
plotter3.add_point_labels(
    [[0, axis_length_global * 1.1, 0]],
    ['Y'],
    font_size=24,
    text_color='green',
    bold=True
)
plotter3.add_point_labels(
    [[0, 0, axis_length_global * 1.1]],
    ['Z (UP)'],
    font_size=24,
    text_color='blue',
    bold=True
)

# Add title
plotter3.add_text(
    f"Properly Oriented Point Cloud\n"
    f"Ground plane at Z=0\n"
    f"X (red) = Right, Y (green) = Forward, Z (blue) = Up\n"
    f"{len(points_rotated):,} points",
    font_size=12,
    position='upper_left'
)

# Add grid and axes
plotter3.show_grid(
    xlabel='X (Right)',
    ylabel='Y (Forward)', 
    zlabel='Z (Up)',
    color='gray',
    font_size=10
)

# Set camera to a nice viewing angle (elevated 45 degrees, looking from southeast)
center = points_rotated.mean(axis=0)
camera_distance = size_oriented * 1.5
plotter3.camera_position = [
    (center[0] + camera_distance * 0.7, center[1] + camera_distance * 0.7, center[2] + camera_distance * 0.5),
    center,
    (0, 0, 1)  # Up direction
]

# Save screenshots
os.makedirs("data/outputs/semantic_3d/oriented", exist_ok=True)

plotter3.show()

# View 1: Default elevated view
plotter3.screenshot("data/outputs/semantic_3d/oriented/oriented_default.png")
print(f"   📸 Saved default view")

# View 2: Top-down view
plotter3.camera_position = [(center[0], center[1], center[2] + size_oriented), center, (0, 1, 0)]
plotter3.screenshot("data/outputs/semantic_3d/oriented/oriented_top.png")
print(f"   📸 Saved top view")

# View 3: Side view (from +X looking toward -X)
plotter3.camera_position = [(center[0] + size_oriented, center[1], center[2] + size_oriented*0.3), center, (0, 0, 1)]
plotter3.screenshot("data/outputs/semantic_3d/oriented/oriented_side_x.png")
print(f"   📸 Saved side view (X)")

# View 4: Front view (from +Y looking toward -Y)
plotter3.camera_position = [(center[0], center[1] + size_oriented, center[2] + size_oriented*0.3), center, (0, 0, 1)]
plotter3.screenshot("data/outputs/semantic_3d/oriented/oriented_front_y.png")
print(f"   📸 Saved front view (Y)")

# Try to show interactive window
print(f"\n👁️  Opening interactive viewer with properly oriented point cloud...")
print("   The point cloud should now be:")
print("   - Ground plane at Z=0 (gray grid)")
print("   - Objects extending upward in +Z direction (blue axis)")
print("   - Meaningful X (red) and Y (green) axes")
print("")
print("   Controls:")
print("   - Left Click + Drag: Rotate")
print("   - Middle Click + Drag: Pan")
print("   - Right Click + Drag: Zoom")
print("   - Scroll: Zoom")
print("   - 'q': Close window")

try:
    plotter3.show()
    print(f"✅ Viewer closed")
except:
    print(f"✅ Screenshots saved (no display available for interactive mode)")

print(f"\n📁 Saved visualizations to: data/outputs/semantic_3d/oriented/")

# %% BLOCK 8: Save Oriented Point Cloud
print("\n" + "="*80)
print("SAVING ORIENTED POINT CLOUD")
print("="*80)

print(f"\n💾 Saving oriented point cloud to {OUTPUT_PATH}...")

# Convert Open3D point cloud to trimesh
points_out = np.asarray(pcd_oriented.points)
colors_out = np.asarray(pcd_oriented.colors) if pcd_oriented.has_colors() else None

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

# Print final statistics
print(f"\n📊 Final Oriented Point Cloud Statistics:")
print(f"   - Number of points: {len(points_out):,}")
print(f"   - Bounding box:")
print(f"     - X: [{points_out[:, 0].min():.3f}, {points_out[:, 0].max():.3f}]")
print(f"     - Y: [{points_out[:, 1].min():.3f}, {points_out[:, 1].max():.3f}]")
print(f"     - Z: [{points_out[:, 2].min():.3f}, {points_out[:, 2].max():.3f}]")
print(f"   - Center: [{points_out.mean(axis=0)[0]:.3f}, {points_out.mean(axis=0)[1]:.3f}, {points_out.mean(axis=0)[2]:.3f}]")
print(f"\n🎉 Point cloud orientation complete!")
