#!/usr/bin/env python3
"""
Debug script to test 3D to 2D projection with occlusion handling.

This script:
1. Loads ALL frames and creates a complete 3D point cloud
2. For a selected frame, projects the FULL 3D point cloud back to that frame's 2D view
3. Handles occlusion using depth buffering (z-buffer)
4. Compares the re-projected image with the original image
5. Visualizes the difference to check if camera parameters are correct
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh


def load_all_data(input_dir: str):
    """Load all prediction data and point cloud."""
    npz_path = os.path.join(input_dir, "predictions.npz")
    glb_path = os.path.join(input_dir, "point_cloud.glb")

    print(f"\n📂 Loading predictions from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    print(f"✅ Loaded data:")
    print(f"   - Frames: {len(data['depths'])}")
    print(f"   - Depth shape: {data['depths'].shape}")
    print(f"   - Images shape: {data['images'].shape}")

    # Load point cloud from GLB
    print(f"\n📂 Loading point cloud from {glb_path}...")
    loaded = trimesh.load(glb_path)

    # Handle both Scene and PointCloud types
    if isinstance(loaded, trimesh.Scene):
        # GLB files often load as Scene objects
        # Extract the point cloud from the scene
        geometries = list(loaded.geometry.values())
        if len(geometries) == 0:
            raise ValueError("Scene has no geometries")
        point_cloud = geometries[0]
    else:
        point_cloud = loaded

    if isinstance(point_cloud, trimesh.points.PointCloud):
        points_world = point_cloud.vertices
        colors = point_cloud.colors[:, :3]  # RGB only, drop alpha if present
    else:
        raise ValueError(f"Expected PointCloud, got {type(point_cloud)}")

    print(f"✅ Loaded point cloud:")
    print(f"   - Points: {len(points_world)}")
    print(f"   - Colors shape: {colors.shape}")

    return data, points_world, colors




def project_3d_to_2d_with_zbuffer(points_world, colors, K, extrinsic, img_width, img_height):
    """
    Project 3D world points to 2D image with proper depth buffering (occlusion handling).

    Args:
        points_world: (M, 3) 3D points in world coordinates
        colors: (M, 3) RGB colors
        K: (3, 3) intrinsic matrix
        extrinsic: (3, 4) extrinsic matrix [R|t]
        img_width, img_height: Output image dimensions

    Returns:
        rendered_image: (H, W, 3) rendered image from 3D projection
        depth_buffer: (H, W) depth map
        valid_mask: (H, W) boolean mask of rendered pixels
    """
    print(f"\n🔄 Projecting 3D points to 2D with z-buffering...")

    # Transform world to camera
    R = extrinsic[:, :3]
    t = extrinsic[:, 3]
    points_cam = (R @ points_world.T).T + t

    # Project to image
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    Zc = points_cam[:, 2]
    u = (points_cam[:, 0] * fx / Zc) + cx
    v = (points_cam[:, 1] * fy / Zc) + cy

    # Filter points in front of camera and within image bounds
    valid = (
        (Zc > 0.001) &
        (u >= 0) & (u < img_width) &
        (v >= 0) & (v < img_height)
    )

    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = Zc[valid]
    colors_valid = colors[valid]

    print(f"   Points in view: {len(u_valid)} / {len(points_world)}")

    # Convert to integer pixel coordinates
    u_int = u_valid.astype(np.int32)
    v_int = v_valid.astype(np.int32)

    # Initialize z-buffer and image
    depth_buffer = np.full((img_height, img_width), np.inf, dtype=np.float32)
    rendered_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Sort points by depth (farthest first) for proper z-buffering
    sort_idx = np.argsort(-z_valid)  # Negative for descending order

    # Render with z-buffering
    for idx in sort_idx:
        u_px = u_int[idx]
        v_px = v_int[idx]
        z_px = z_valid[idx]

        # Only render if this point is closer than what's already in the buffer
        if z_px < depth_buffer[v_px, u_px]:
            depth_buffer[v_px, u_px] = z_px
            rendered_image[v_px, u_px] = colors_valid[idx]

    # Create valid mask (pixels that were rendered)
    valid_mask = depth_buffer < np.inf

    print(f"   Rendered pixels: {valid_mask.sum()} / {img_width * img_height}")
    print(f"   Coverage: {100 * valid_mask.sum() / (img_width * img_height):.1f}%")

    return rendered_image, depth_buffer, valid_mask


def compare_images(original, rendered, valid_mask, frame_idx, output_dir):
    """Create comparison visualization."""
    print(f"\n🎨 Creating comparison visualization...")

    # Compute difference (only where valid)
    diff = np.zeros_like(original)
    diff[valid_mask] = np.abs(original[valid_mask].astype(np.float32) -
                               rendered[valid_mask].astype(np.float32)).astype(np.uint8)

    # Compute metrics
    mse = np.mean(diff[valid_mask]**2) if valid_mask.sum() > 0 else 0
    mae = np.mean(diff[valid_mask]) if valid_mask.sum() > 0 else 0

    print(f"   Difference metrics (on valid pixels):")
    print(f"     MAE: {mae:.2f}")
    print(f"     MSE: {mse:.2f}")
    print(f"     RMSE: {np.sqrt(mse):.2f}")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title(f'Original Image (Frame {frame_idx})')
    axes[0, 0].axis('off')

    # Rendered image
    axes[0, 1].imshow(rendered)
    axes[0, 1].set_title('Rendered from 3D Point Cloud')
    axes[0, 1].axis('off')

    # Difference
    axes[0, 2].imshow(diff)
    axes[0, 2].set_title(f'Absolute Difference\nMAE={mae:.2f}, RMSE={np.sqrt(mse):.2f}')
    axes[0, 2].axis('off')

    # Valid mask
    axes[1, 0].imshow(valid_mask, cmap='gray')
    axes[1, 0].set_title(f'Valid Pixels\n{valid_mask.sum()} pixels ({100*valid_mask.sum()/(valid_mask.size):.1f}%)')
    axes[1, 0].axis('off')

    # Overlay: show where rendering differs significantly
    overlay = original.copy()
    high_diff = (diff.mean(axis=2) > 30) & valid_mask  # Significant difference
    overlay[high_diff] = [255, 0, 0]  # Mark in red
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f'High Difference Regions (>30)\n{high_diff.sum()} pixels')
    axes[1, 1].axis('off')

    # Difference heatmap
    diff_gray = diff.mean(axis=2)
    im = axes[1, 2].imshow(diff_gray, cmap='hot', vmin=0, vmax=50)
    axes[1, 2].set_title('Difference Heatmap')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, f'debug_3d_to_2d_frame_{frame_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to {output_path}")

    plt.show()

    return mae, np.sqrt(mse)


def main():
    parser = argparse.ArgumentParser(
        description="Debug 3D to 2D projection with occlusion handling"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/outputs/semantic_3d",
        help="Directory containing predictions.npz",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index to test projection for (default: 0)",
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("DEBUG: 3D to 2D Projection with Z-Buffer")
    print(f"{'='*80}")
    print(f"Target frame: {args.frame_idx}")

    try:
        # Load data and point cloud
        data, points_world, colors = load_all_data(args.input_dir)

        depths = data['depths']
        masks = data['masks']
        images = data['images']
        intrinsics = data['intrinsics']
        extrinsics = data['extrinsics']

        # Get target frame data
        frame_idx = args.frame_idx
        original_image = images[frame_idx]
        K = intrinsics[frame_idx]
        extrinsic = extrinsics[frame_idx]
        H, W = original_image.shape[:2]

        print(f"\n🎯 Target frame {frame_idx}:")
        print(f"   Image size: {W}x{H}")
        print(f"   Using camera parameters from frame {frame_idx}")

        # Project full 3D point cloud back to this frame's view
        rendered_image, depth_buffer, valid_mask = project_3d_to_2d_with_zbuffer(
            points_world, colors, K, extrinsic, W, H
        )

        # Compare
        mae, rmse = compare_images(
            original_image, rendered_image, valid_mask,
            frame_idx, args.input_dir
        )

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")

        if mae < 10 and rmse < 15:
            print("✅ Camera parameters look correct!")
            print("   The 3D reconstruction matches the original images well.")
        elif mae < 30 and rmse < 40:
            print("⚠️  Camera parameters are approximately correct")
            print("   There are some differences, but the overall structure matches.")
        else:
            print("❌ Significant differences detected!")
            print("   The camera parameters may be incorrect, or there are issues with")
            print("   the 3D reconstruction. Check the visualization for details.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
