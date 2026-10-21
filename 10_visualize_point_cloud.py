#!/usr/bin/env python3
"""
Simple PyVista Point Cloud Viewer

This script loads and visualizes a point cloud GLB file using PyVista.

Usage:
    python 10_visualize_point_cloud.py
"""

# %%
import os
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

# Set working directory to parent folder of this file
script_dir = Path(__file__).parent
os.chdir(script_dir)
print(f"Working directory: {os.getcwd()}")

# Configuration
INPUT_PATH = "data/outputs/semantic_3d/point_cloud_cleaned.glb"
INPUT_PATH = "data/outputs/semantic_3d/point_cloud_oriented.glb"

print(f"Loading point cloud from: {INPUT_PATH}")

# %% Load the point cloud
print("\nLoading point cloud...")
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

print(f"Loaded {len(points):,} points")
print(f"Has colors: {colors is not None}")

# Print statistics
print(f"\nPoint Cloud Statistics:")
print(f"  - Number of points: {len(points):,}")
print(f"  - Bounding box:")
print(f"    - X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
print(f"    - Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
print(f"    - Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
center = points.mean(axis=0)
print(f"  - Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")

# %% Create PyVista visualization
print("\nCreating PyVista visualization...")

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

print(f"Created PyVista point cloud with {pv_cloud.n_points:,} points")

# %% Visualize interactively
print("\nOpening interactive 3D viewer...")
print("Controls:")
print("  - Left Click + Drag: Rotate")
print("  - Middle Click + Drag: Pan")
print("  - Right Click + Drag: Zoom")
print("  - Scroll: Zoom")
print("  - 'q': Close window")
print("  - 'r': Reset camera")
print("  - 's': Save screenshot")

# Create plotter
plotter = pv.Plotter(window_size=[1600, 1200])

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

# Add title
plotter.add_text(
    f"Point Cloud: {INPUT_PATH}\n{len(points):,} points",
    font_size=12,
    position='upper_left'
)

# Set camera position (slightly elevated view)
plotter.camera_position = 'xy'  # Top-down view initially

# Show the plot
print("\nLaunching viewer...")
plotter.show()

print("\nViewer closed")

# %%
