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
Interactive script for processing video into depth maps and 3D point clouds.

This script extracts frames from a video, runs depth estimation, and exports
results in multiple formats (.glb, .ply, .npz, and individual frames).

Can be run as a regular script or interactively in VSCode/Jupyter using cell blocks.
"""

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

import os
from datetime import datetime

# Input video path
INPUT_VIDEO = "data/inputs/videos/short.mp4"

# Model selection
# Options: "depth-anything/DA3-Small", "depth-anything/DA3-Base",
#          "depth-anything/DA3-Large", "depth-anything/DA3NESTED-GIANT-LARGE"
MODEL_NAME = "depth-anything/DA3-Small"

# Maximum number of frames to extract from video
MAX_FRAMES = 50

# Output folder name
# Set to None for timestamp-based folder, or provide a descriptive name
OUTPUT_FOLDER_NAME = "chair_in_kitchen"  # e.g., "my_video_test" or None

# Processing resolution method
# - "upper_bound_resize": Faster, lower quality
# - "lower_bound_resize": Slower, higher quality
PROCESS_RES_METHOD = "upper_bound_resize"

# Device selection (auto-detected if None)
DEVICE = None  # "cuda" or "cpu" or None for auto-detect

print(" Configuration loaded")
print(f"  - Input video: {INPUT_VIDEO}")
print(f"  - Model: {MODEL_NAME}")
print(f"  - Max frames: {MAX_FRAMES}")
print(f"  - Output folder: {OUTPUT_FOLDER_NAME or 'timestamp-based'}")
print(f"  - Process method: {PROCESS_RES_METHOD}")

# %%
# =============================================================================
# IMPORTS
# =============================================================================

import gc
import glob
import shutil
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from addict import Dict as AddictDict

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

print(" Imports loaded")

# %%
# =============================================================================
# SETUP OUTPUT DIRECTORY
# =============================================================================

def setup_output_directory(base_dir="data/outputs", folder_name=None):
    """
    Create output directory with timestamp or custom name.

    Args:
        base_dir: Base output directory
        folder_name: Custom folder name or None for timestamp

    Returns:
        Path to output directory
    """
    if folder_name is None:
        # Use timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = os.path.join(base_dir, timestamp)
    else:
        # Use custom name
        output_dir = os.path.join(base_dir, folder_name)

        # Check if folder exists
        if os.path.exists(output_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            new_output_dir = os.path.join(base_dir, f"{timestamp}_{folder_name}")
            print(f"�  Warning: Folder '{output_dir}' already exists!")
            print(f"   Using alternative folder: '{new_output_dir}'")
            output_dir = new_output_dir

    os.makedirs(output_dir, exist_ok=True)
    return output_dir

OUTPUT_DIR = setup_output_directory(folder_name=OUTPUT_FOLDER_NAME)
print(f" Output directory: {OUTPUT_DIR}")

# Create subdirectories
SELECTED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "selected_images")
os.makedirs(SELECTED_IMAGES_DIR, exist_ok=True)
print(f" Selected images directory: {SELECTED_IMAGES_DIR}")

# %%
# =============================================================================
# EXTRACT FRAMES FROM VIDEO
# =============================================================================

def extract_frames(video_path, output_dir, max_frames=50):
    """
    Extract frames uniformly from video.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract

    Returns:
        List of paths to extracted frames
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")

    print(f"\n<� Extracting frames from {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Duration: {total_frames / fps:.2f} seconds")

    # Calculate frame indices to extract
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        # Uniformly sample frames
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    print(f"  - Extracting {len(frame_indices)} frames...")

    # Extract frames
    frame_paths = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"    �  Failed to read frame {frame_idx}")
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save frame
        frame_path = os.path.join(output_dir, f"{i:04d}.png")
        imageio.imwrite(frame_path, frame_rgb)
        frame_paths.append(frame_path)

    cap.release()

    print(f" Extracted {len(frame_paths)} frames to {output_dir}")
    return frame_paths

# Extract frames
frame_paths = extract_frames(INPUT_VIDEO, SELECTED_IMAGES_DIR, max_frames=MAX_FRAMES)

print(f"\n=� Extracted frames:")
for i, path in enumerate(frame_paths[:5]):
    print(f"  {i+1}. {os.path.basename(path)}")
if len(frame_paths) > 5:
    print(f"  ... and {len(frame_paths) - 5} more")

# %%
# =============================================================================
# SETUP DEVICE AND GPU MEMORY
# =============================================================================

# Auto-detect device
if DEVICE is None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n=�  Device: {DEVICE}")

# Clear GPU memory before loading model
if DEVICE == "cuda" and torch.cuda.is_available():
    print("\n>� Clearing GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

    print(f"  - Allocated: {allocated:.2f} GB")
    print(f"  - Reserved: {reserved:.2f} GB")
    print(f"  - Total: {total:.2f} GB")
    print(f"  - Free: {total - reserved:.2f} GB")

# %%
# =============================================================================
# LOAD MODEL
# =============================================================================

print(f"\n= Loading model: {MODEL_NAME}")
start_time = time.time()

model = DepthAnything3.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
model.eval()

load_time = time.time() - start_time
print(f" Model loaded in {load_time:.2f} seconds")

# Check GPU memory after loading model
if DEVICE == "cuda" and torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  - GPU memory after loading: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# %%
# =============================================================================
# RUN INFERENCE
# =============================================================================

print(f"\n=� Running multi-view depth estimation...")
print(f"  - Number of images: {len(frame_paths)}")
print(f"  - Processing method: {PROCESS_RES_METHOD}")

inference_start = time.time()

with torch.no_grad():
    prediction = model.inference(
        frame_paths,
        export_dir=None,
        process_res_method=PROCESS_RES_METHOD,
        infer_gs=False,
    )

inference_time = time.time() - inference_start
print(f" Inference completed in {inference_time:.2f} seconds")
print(f"   ({inference_time / len(frame_paths):.2f} seconds per frame)")

# Display prediction summary
print(f"\n=� Prediction Summary:")
print(f"  - Number of views: {prediction.depth.shape[0]}")
print(f"  - Depth shape: {prediction.depth.shape}")
print(f"  - Is metric: {bool(prediction.is_metric)}")

if prediction.scale_factor is not None:
    if isinstance(prediction.scale_factor, dict):
        print(f"  - Scale factor: (dict - not saved)")
    else:
        print(f"  - Scale factor: {prediction.scale_factor:.4f}")

if prediction.extrinsics is not None:
    print(f"  - Camera extrinsics: {prediction.extrinsics.shape}")
if prediction.intrinsics is not None:
    print(f"  - Camera intrinsics: {prediction.intrinsics.shape}")

# %%
# =============================================================================
# SAVE NPZ WITH ALL PREDICTION DATA
# =============================================================================

print(f"\n=� Saving predictions.npz...")
npz_path = os.path.join(OUTPUT_DIR, "predictions.npz")

save_dict = {}

# Save depth maps
if prediction.depth is not None:
    save_dict["depths"] = np.round(prediction.depth, 6)

# Save confidence maps if available
if prediction.conf is not None:
    save_dict["conf"] = np.round(prediction.conf, 2)

# Save camera extrinsics (camera poses)
if prediction.extrinsics is not None:
    save_dict["extrinsics"] = prediction.extrinsics

# Save camera intrinsics
if prediction.intrinsics is not None:
    save_dict["intrinsics"] = prediction.intrinsics

# Save processed images
if prediction.processed_images is not None:
    save_dict["images"] = prediction.processed_images

# Save metric flag
if prediction.is_metric is not None:
    if isinstance(prediction.is_metric, (dict, AddictDict)):
        save_dict["is_metric"] = 0
    else:
        save_dict["is_metric"] = int(prediction.is_metric)
else:
    save_dict["is_metric"] = 0

# Save scale factor if available
if prediction.scale_factor is not None:
    if not isinstance(prediction.scale_factor, dict):
        save_dict["scale_factor"] = float(prediction.scale_factor)

np.savez_compressed(npz_path, **save_dict)
print(f" Saved predictions.npz to {npz_path}")
print(f"   Contains: {list(save_dict.keys())}")

# %%
# =============================================================================
# EXPORT POINT CLOUD (.PLY)
# =============================================================================

print(f"\n  Exporting point cloud to .ply...")

def export_point_cloud_ply(prediction, output_path):
    """Export point cloud to PLY format."""
    # Get depth, images, and camera parameters
    depths = prediction.depth
    images = prediction.processed_images
    intrinsics = prediction.intrinsics
    extrinsics = prediction.extrinsics

    if depths is None or images is None or intrinsics is None or extrinsics is None:
        print("�  Missing required data for point cloud export")
        return

    # Collect all points and colors
    all_points = []
    all_colors = []

    for idx in range(len(depths)):
        depth = depths[idx]
        image = images[idx]
        K = intrinsics[idx]  # 3x3 intrinsic matrix

        # Get extrinsic as 4x4 matrix
        if extrinsics[idx].shape == (3, 4):
            # Add [0, 0, 0, 1] row
            extrinsic = np.vstack([extrinsics[idx], [0, 0, 0, 1]])
        else:
            extrinsic = extrinsics[idx]

        # Create pixel grid
        h, w = depth.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Unproject to camera coordinates
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        z = depth
        x_cam = (x - cx) * z / fx
        y_cam = (y - cy) * z / fy

        # Stack into (H, W, 3)
        points_cam = np.stack([x_cam, y_cam, z], axis=-1)

        # Transform to world coordinates
        points_cam_homo = np.concatenate([
            points_cam.reshape(-1, 3),
            np.ones((h * w, 1))
        ], axis=1)  # (H*W, 4)

        # Apply extrinsic transformation
        points_world = (extrinsic @ points_cam_homo.T).T[:, :3]

        # Get colors
        colors = image.reshape(-1, 3)

        all_points.append(points_world)
        all_colors.append(colors)

    # Concatenate all points and colors
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # Write PLY file
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for point, color in zip(all_points, all_colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} ")
            f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")

    print(f" Exported point cloud with {len(all_points):,} points")

ply_path = os.path.join(OUTPUT_DIR, "point_cloud.ply")
export_point_cloud_ply(prediction, ply_path)
print(f" Saved point cloud to {ply_path}")


# %%
# =============================================================================
# EXPORT GLB FILE
# =============================================================================

print(f"📦 Exporting GLB file...")

from depth_anything_3.utils.export.glb import export_to_glb

glb_path = export_to_glb(
    prediction,
    export_dir=OUTPUT_DIR,
    num_max_points=1_000_000,
    conf_thresh_percentile=40.0,
    show_cameras=True,
    camera_size=0.03,
    export_depth_vis=False,  # Don't export depth visualizations
)

print(f"✅ Saved GLB file to {glb_path}")

# %%
# =============================================================================
# SUMMARY AND CLEANUP
# =============================================================================

print(f"\n{'='*70}")
print(f"<� PROCESSING COMPLETE!")
print(f"{'='*70}")

print(f"\n=� Output directory: {OUTPUT_DIR}")
print(f"\n=� Generated files:")
print(f"  ├── selected_images/      ({len(frame_paths)} PNG files)")
print(f"  ├── predictions.npz       (all prediction data)")
print(f"  ├── point_cloud.ply       (3D point cloud)")
print(f"  └── scene.glb             (3D scene with cameras)")

print(f"\n�  Total time: {time.time() - start_time:.2f} seconds")

# Cleanup GPU memory
if DEVICE == "cuda" and torch.cuda.is_available():
    print(f"\n>� Cleaning up GPU memory...")
    del model
    del prediction
    torch.cuda.empty_cache()
    gc.collect()

    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"  - GPU memory after cleanup: {allocated:.2f} GB allocated")

print(f"\n All done! Check {OUTPUT_DIR} for results.")

# %%
