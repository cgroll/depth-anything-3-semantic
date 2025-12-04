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
Semantic 3D reconstruction script - Part 2: Combine results and generate 3D visualization.

This script:
1. Loads the depth maps, masks, and camera poses from Part 1
2. Projects 2D pixels to 3D points using depth and camera parameters
3. For each 3D point, determines if it was part of the object mask in any view
4. Computes a fraction indicating how many views show the point as part of the object
5. Creates a .glb file with the point cloud, highlighting object points in yellow
6. Creates a video showing the original frames with mask overlays

Usage:
    python 05_semantic_3d_part2.py
    python 05_semantic_3d_part2.py --input-dir data/outputs/semantic_3d
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_predictions(input_dir: str):
    """
    Load predictions from Part 1.

    Args:
        input_dir: Directory containing predictions.npz

    Returns:
        data: Dictionary with all prediction data
    """
    npz_path = os.path.join(input_dir, "predictions.npz")

    if not os.path.exists(npz_path):
        raise ValueError(f"Predictions file not found: {npz_path}")

    print(f"\n📂 Loading predictions from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    print(f"✅ Loaded predictions:")
    print(f"   - Depths shape: {data['depths'].shape}")
    print(f"   - Masks shape: {data['masks'].shape}")
    print(f"   - Images shape: {data['images'].shape if 'images' in data else 'N/A'}")
    print(f"   - Intrinsics shape: {data['intrinsics'].shape if 'intrinsics' in data else 'N/A'}")
    print(f"   - Extrinsics shape: {data['extrinsics'].shape if 'extrinsics' in data else 'N/A'}")

    return data


def project_to_3d(depth_maps, intrinsics, extrinsics, images):
    """
    Project depth maps to 3D points using camera parameters.

    Args:
        depth_maps: (N, H, W) depth maps
        intrinsics: (N, 3, 3) camera intrinsic matrices
        extrinsics: (N, 3, 4) camera extrinsic matrices [R|t]
        images: (N, H, W, 3) RGB images

    Returns:
        points_3d: (M, 3) 3D points in world coordinates
        colors: (M, 3) RGB colors for each point
        point_to_views: (M, K) mapping from 3D points to view indices (padded with -1)
    """
    N, H, W = depth_maps.shape
    print(f"\n🔄 Projecting {N} depth maps to 3D...")

    all_points = []
    all_colors = []
    all_view_indices = []

    for view_idx in range(N):
        if (view_idx + 1) % 10 == 0 or view_idx == 0:
            print(f"   Processing view {view_idx + 1}/{N}...")

        depth = depth_maps[view_idx]
        K = intrinsics[view_idx]
        extrinsic = extrinsics[view_idx]  # (3, 4) [R|t]
        image = images[view_idx]

        # Create pixel coordinates
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        u = u.flatten()
        v = v.flatten()
        depth_flat = depth.flatten()

        # Filter out invalid depths
        valid = depth_flat > 0
        u = u[valid]
        v = v[valid]
        depth_flat = depth_flat[valid]

        # Back-project to camera coordinates
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x_cam = (u - cx) * depth_flat / fx
        y_cam = (v - cy) * depth_flat / fy
        z_cam = depth_flat

        # Stack to camera coordinates (M, 3)
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Transform to world coordinates
        # extrinsic is [R|t] where camera_point = R @ world_point + t
        # So: world_point = R^T @ (camera_point - t)
        R = extrinsic[:, :3]  # (3, 3)
        t = extrinsic[:, 3]   # (3,)

        # Inverse transformation: world = R^T @ (camera - t)
        points_world = (R.T @ (points_cam - t).T).T  # (M, 3)

        # Get colors
        colors = image[v, u, :]  # (M, 3)

        all_points.append(points_world)
        all_colors.append(colors)
        all_view_indices.extend([view_idx] * len(points_world))

    # Concatenate all points
    points_3d = np.vstack(all_points)
    colors = np.vstack(all_colors)
    view_indices = np.array(all_view_indices)

    print(f"✅ Generated {len(points_3d)} 3D points")

    return points_3d, colors, view_indices


def compute_object_fractions(points_3d, view_indices, masks, intrinsics, extrinsics):
    """
    For each 3D point, compute the fraction of views where it appears in the object mask.

    Args:
        points_3d: (M, 3) 3D points in world coordinates
        view_indices: (M,) source view index for each point
        masks: (N, H, W) binary masks
        intrinsics: (N, 3, 3) camera intrinsic matrices
        extrinsics: (N, 3, 4) camera extrinsic matrices

    Returns:
        object_fractions: (M,) fraction of views where each point is in the mask
    """
    N = len(masks)
    M = len(points_3d)

    print(f"\n🔍 Computing object fractions for {M} points across {N} views...")

    # For each point, count how many views see it as part of the object
    in_object_count = np.zeros(M, dtype=np.float32)
    visible_count = np.zeros(M, dtype=np.float32)

    for view_idx in range(N):
        if (view_idx + 1) % 10 == 0 or view_idx == 0:
            print(f"   Processing view {view_idx + 1}/{N}...")

        mask = masks[view_idx]
        K = intrinsics[view_idx]
        extrinsic = extrinsics[view_idx]
        H, W = mask.shape

        # Transform 3D world points to camera coordinates
        # camera_point = R @ world_point + t
        R = extrinsic[:, :3]  # (3, 3)
        t = extrinsic[:, 3]   # (3,)

        # Forward transformation: camera = R @ world + t
        points_cam = (R @ points_3d.T).T + t  # (M, 3)

        # Project to image coordinates
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Avoid division by zero
        Zc = points_cam[:, 2]
        x_img = (points_cam[:, 0] * fx / Zc) + cx
        y_img = (points_cam[:, 1] * fy / Zc) + cy

        # Check if points are in image bounds and in front of camera
        in_bounds = (
            (x_img >= 0) & (x_img < W) &
            (y_img >= 0) & (y_img < H) &
            (Zc > 0.001)  # In front of camera
        )

        # For points in bounds, check if they're in the mask
        x_img_int = np.clip(x_img[in_bounds].astype(np.int32), 0, W - 1)
        y_img_int = np.clip(y_img[in_bounds].astype(np.int32), 0, H - 1)

        in_mask = mask[y_img_int, x_img_int] > 0

        # Update counts
        visible_count[in_bounds] += 1
        in_object_count[in_bounds] += in_mask

    # Compute fraction (avoid division by zero)
    object_fractions = np.zeros(M, dtype=np.float32)
    valid = visible_count > 0
    object_fractions[valid] = in_object_count[valid] / visible_count[valid]

    print(f"✅ Computed object fractions")
    print(f"   - Mean fraction: {object_fractions.mean():.3f}")
    print(f"   - Max fraction: {object_fractions.max():.3f}")
    print(f"   - Points with fraction > 0: {(object_fractions > 0).sum()}")

    return object_fractions


def create_mask_overlay_video(images, masks, output_path: str, fps: int = 10):
    """
    Create a video with mask overlays on images.

    Args:
        images: (N, H, W, 3) RGB images
        masks: (N, H, W) binary masks
        output_path: Path to save video file
        fps: Frames per second for the video
    """
    print(f"\n🎬 Creating video with mask overlays...")

    N, H, W, _ = images.shape
    print(f"   - Frames: {N}")
    print(f"   - Resolution: {W}x{H}")
    print(f"   - FPS: {fps}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    if not video_writer.isOpened():
        raise ValueError(f"Failed to open video writer for {output_path}")

    # Process each frame
    for frame_idx in range(N):
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            print(f"   Processing frame {frame_idx + 1}/{N}...")

        image = images[frame_idx]
        mask = masks[frame_idx]

        # Resize mask to match image dimensions (same as in 06_debug_projection.py)
        if mask.shape != (H, W):
            mask_resized = cv2.resize(mask, (W, H))
        else:
            mask_resized = mask

        # Create overlay using the same method as 06_debug_projection.py
        overlay = image.copy()
        mask_bool = mask_resized > 0.5

        # Create colored overlay (lime green)
        color = np.array([0, 255, 0], dtype=np.uint8)
        overlay[mask_bool] = overlay[mask_bool] * 0.5 + color * 0.5
        overlay = overlay.astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # Write frame
        video_writer.write(overlay_bgr)

    # Release video writer
    video_writer.release()

    print(f"✅ Saved video with mask overlays to {output_path}")


def create_glb_visualization(
    points_3d,
    colors,
    object_fractions,
    output_path: str,
    threshold: float = 0.8,
):
    """
    Create a .glb file with the point cloud, highlighting object points.

    Args:
        points_3d: (M, 3) 3D points
        colors: (M, 3) RGB colors
        object_fractions: (M,) object membership fractions
        output_path: Path to save .glb file
        threshold: Threshold for considering a point as part of the object
    """
    print(f"\n🎨 Creating 3D visualization...")

    # Identify object points
    is_object = object_fractions > threshold
    num_object_points = is_object.sum()

    print(f"   - Total points: {len(points_3d)}")
    print(f"   - Object points (fraction > {threshold}): {num_object_points}")
    print(f"   - Background points: {len(points_3d) - num_object_points}")

    # Create point cloud colors
    # Background points: original RGB colors
    # Object points: yellow with some transparency (but glb doesn't support alpha on points, so just yellow)
    vis_colors = colors.copy().astype(np.float32) / 255.0

    # Set object points to yellow
    vis_colors[is_object] = np.array([1.0, 1.0, 0.0])  # Yellow

    # Create point cloud using trimesh
    # Note: trimesh PointCloud doesn't support per-point alpha, so we just use color
    point_cloud = trimesh.points.PointCloud(
        vertices=points_3d,
        colors=(vis_colors * 255).astype(np.uint8),
    )

    # Save as .glb
    print(f"\n💾 Saving to {output_path}...")
    point_cloud.export(output_path)

    print(f"✅ Saved 3D visualization to {output_path}")

    # Print statistics
    print(f"\n📊 Visualization Statistics:")
    print(f"   - Point cloud bounds:")
    print(f"     - X: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}]")
    print(f"     - Y: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}]")
    print(f"     - Z: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]")
    print(f"   - Object points percentage: {100 * num_object_points / len(points_3d):.1f}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic 3D reconstruction - Part 2: Combine results and generate 3D visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default output from Part 1
  python 05_semantic_3d_part2.py

  # Use custom input directory
  python 05_semantic_3d_part2.py --input-dir path/to/semantic_3d

  # Adjust object threshold
  python 05_semantic_3d_part2.py --threshold 0.7

  # Customize video output
  python 05_semantic_3d_part2.py --video-path output/masks.mp4 --fps 15
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/outputs/semantic_3d",
        help="Directory containing predictions from Part 1 (default: data/outputs/semantic_3d)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/outputs/semantic_3d/point_cloud_semantic.glb",
        help="Path to save .glb file (default: data/outputs/semantic_3d/point_cloud_semantic.glb)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Threshold for object membership fraction (default: 0.8)",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="data/outputs/semantic_3d/mask_overlay_video.mp4",
        help="Path to save video with mask overlays (default: data/outputs/semantic_3d/mask_overlay_video.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for video (default: 10)",
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("SEMANTIC 3D RECONSTRUCTION - PART 2")
    print(f"{'='*80}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_path}")
    print(f"Threshold: {args.threshold}")

    start_time = time.time()

    try:
        # Step 1: Load predictions
        data = load_predictions(args.input_dir)

        depths = data['depths']
        masks = data['masks']
        images = data['images']
        intrinsics = data['intrinsics']
        extrinsics = data['extrinsics']

        # Step 2: Project to 3D
        points_3d, colors, view_indices = project_to_3d(
            depths, intrinsics, extrinsics, images
        )

        # Step 3: Compute object fractions
        object_fractions = compute_object_fractions(
            points_3d, view_indices, masks, intrinsics, extrinsics
        )

        # Step 4: Create .glb visualization
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        create_glb_visualization(
            points_3d,
            colors,
            object_fractions,
            args.output_path,
            threshold=args.threshold,
        )

        # Step 5: Create video with mask overlays
        os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
        create_mask_overlay_video(
            images,
            masks,
            args.video_path,
            fps=args.fps,
        )

        total_time = time.time() - start_time
        print(f"\n✅ Processing complete in {total_time:.2f} seconds")
        print(f"📁 3D visualization saved to: {args.output_path}")
        print(f"📁 Video with mask overlays saved to: {args.video_path}")
        print(f"\nYou can open the .glb file in any 3D viewer (e.g., Blender, online viewers)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
