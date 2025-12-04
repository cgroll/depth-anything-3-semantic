#!/usr/bin/env python3
"""
Debug script to test 3D to 2D projection for a single frame.

This script (similar to 07_debug_3d_to_2d.py):
1. Loads the FULL 3D point cloud from point_cloud.glb (all frames combined)
2. Projects the FULL 3D point cloud back to a selected frame's 2D view
3. Handles occlusion using depth buffering (z-buffer)
4. Compares the rendered image with the original image
5. Shows depth maps with object mask outlines
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def load_all_data(input_dir: str):
    """Load all prediction data and point cloud (same as 07_debug_3d_to_2d.py)."""
    npz_path = os.path.join(input_dir, "predictions.npz")
    glb_path = os.path.join(input_dir, "point_cloud.glb")

    print(f"\n📂 Loading predictions from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    print(f"✅ Loaded data:")
    print(f"   - Frames: {len(data['depths'])}")

    # Load point cloud from GLB
    print(f"\n📂 Loading FULL 3D point cloud from {glb_path}...")
    loaded = trimesh.load(glb_path)

    # Handle both Scene and PointCloud types
    if isinstance(loaded, trimesh.Scene):
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

    print(f"✅ Loaded FULL point cloud from ALL frames:")
    print(f"   - Points: {len(points_world)}")

    return data, points_world, colors


def project_3d_to_2d_with_zbuffer(points_world, colors, K, extrinsic, img_width, img_height):
    """
    Project 3D world points to 2D image with proper depth buffering (same as 07_debug_3d_to_2d.py).
    """
    print(f"\n🔄 Projecting FULL 3D point cloud to 2D with z-buffering...")

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
    sort_idx = np.argsort(-z_valid)

    # Render with z-buffering
    for idx in sort_idx:
        u_px = u_int[idx]
        v_px = v_int[idx]
        z_px = z_valid[idx]

        if z_px < depth_buffer[v_px, u_px]:
            depth_buffer[v_px, u_px] = z_px
            rendered_image[v_px, u_px] = colors_valid[idx]

    print(f"   Rendered pixels: {(depth_buffer < np.inf).sum()} / {img_width * img_height}")
    print(f"   Coverage: {100 * (depth_buffer < np.inf).sum() / (img_width * img_height):.1f}%")

    return rendered_image, depth_buffer


def test_single_frame_projection(input_dir: str, frame_idx: int = 0):
    """
    Test camera projection for a single frame using FULL 3D point cloud.

    Args:
        input_dir: Directory containing predictions.npz and point_cloud.glb
        frame_idx: Index of frame to test
    """
    # Load data AND FULL 3D point cloud from GLB (all frames combined)
    data, points_world_glb, colors_glb = load_all_data(input_dir)

    depths = data['depths']
    masks = data['masks']
    images = data['images']
    intrinsics = data['intrinsics']
    extrinsics = data['extrinsics']

    print(f"\n🔍 Using frame index: {frame_idx}")

    # Get single frame data
    depth = depths[frame_idx]
    mask = masks[frame_idx]
    image = images[frame_idx]
    K = intrinsics[frame_idx]
    extrinsic = extrinsics[frame_idx]

    H, W = depth.shape
    print(f"\n🎯 Frame {frame_idx} properties:")
    print(f"   - Image size: {W}x{H}")
    print(f"   - Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
    print(f"   - Mask pixels: {mask.sum()} / {H*W} ({100*mask.sum()/(H*W):.1f}%)")

    # Use the dedicated z-buffer projection function (same as 07_debug_3d_to_2d.py)
    rendered_image, depth_buffer = project_3d_to_2d_with_zbuffer(
        points_world_glb, colors_glb, K, extrinsic, W, H
    )

    # Visualize
    print(f"\n🎨 Creating visualization...")

    # Debug: Check dimensions
    print(f"   Image shape: {image.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Rendered image shape: {rendered_image.shape}")

    # Resize mask to match image dimensions (same as in 03_yolo.py line 96)
    if mask.shape != (H, W):
        print(f"   ⚠️  Resizing mask from {mask.shape} to ({H}, {W})")
        mask_resized = cv2.resize(mask, (W, H))
    else:
        mask_resized = mask

    # First: Create a test figure with just image + mask overlay (using method from 03_yolo.py)
    fig_test = plt.figure(figsize=(10, 10))
    ax_test = fig_test.add_subplot(111)

    # Create overlay using the same method as 03_yolo.py
    test_overlay = image.copy()
    mask_bool = mask_resized > 0.5

    # Create colored overlay (lime green)
    color = np.array([0, 255, 0], dtype=np.uint8)
    overlay_temp = test_overlay.copy()
    overlay_temp[mask_bool] = overlay_temp[mask_bool] * 0.5 + color * 0.5
    test_overlay = overlay_temp.astype(np.uint8)

    ax_test.imshow(test_overlay)
    ax_test.set_title(f'TEST: Image with Mask Overlay (Frame {frame_idx})')
    ax_test.axis('off')

    # Save test figure
    test_output_path = os.path.join(input_dir, f'test_mask_overlay_frame_{frame_idx}.png')
    plt.savefig(test_output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved TEST mask overlay to {test_output_path}")
    plt.show()

    # Now create the main visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Get mask contours from the resized mask
    contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Found {len(contours)} contours")

    # Row 1, Col 1: Original image WITH mask contour overlay
    axes[0, 0].imshow(image)
    for contour in contours:
        # Contours are in (x, y) format, which matches image coordinates
        axes[0, 0].plot(contour[:, 0, 0], contour[:, 0, 1], 'lime', linewidth=3, alpha=1.0)
    axes[0, 0].set_title(f'Original Image + Mask Outline (Frame {frame_idx})')
    axes[0, 0].axis('off')

    # Row 1, Col 2: Re-projected image from FULL 3D point cloud
    axes[0, 1].imshow(rendered_image)
    axes[0, 1].set_title('Rendered from FULL 3D Point Cloud')
    axes[0, 1].axis('off')

    # Row 1, Col 3: Re-projected image WITH mask contour overlay
    axes[0, 2].imshow(rendered_image)
    for contour in contours:
        axes[0, 2].plot(contour[:, 0, 0], contour[:, 0, 1], 'lime', linewidth=3, alpha=1.0)
    axes[0, 2].set_title('Rendered + Mask Outline')
    axes[0, 2].axis('off')

    # Row 2, Col 1: Depth map
    im_depth1 = axes[1, 0].imshow(depth, cmap='viridis')
    axes[1, 0].set_title('Depth Map')
    axes[1, 0].axis('off')
    plt.colorbar(im_depth1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Row 2, Col 2: Depth map + mask contour
    im_depth2 = axes[1, 1].imshow(depth, cmap='viridis')
    for contour in contours:
        axes[1, 1].plot(contour[:, 0, 0], contour[:, 0, 1], 'lime', linewidth=3, alpha=1.0)
    axes[1, 1].set_title('Depth Map + Object Mask Outline')
    axes[1, 1].axis('off')
    plt.colorbar(im_depth2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Row 2, Col 3: Difference between original and rendered
    valid_render_mask = depth_buffer < np.inf
    diff = np.zeros_like(image)
    if valid_render_mask.sum() > 0:
        diff[valid_render_mask] = np.abs(image[valid_render_mask].astype(np.float32) -
                                         rendered_image[valid_render_mask].astype(np.float32)).astype(np.uint8)

    mae = np.mean(diff[valid_render_mask]) if valid_render_mask.sum() > 0 else 0
    mse = np.mean(diff[valid_render_mask]**2) if valid_render_mask.sum() > 0 else 0
    axes[1, 2].imshow(diff)
    axes[1, 2].set_title(f'Difference (Original - Rendered)\nMAE={mae:.2f}, RMSE={np.sqrt(mse):.2f}')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(input_dir, f'debug_frame_{frame_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to {output_path}")
    plt.show()

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\nRendering quality:")
    print(f"  - Pixels rendered: {valid_render_mask.sum()} / {H*W} ({100*valid_render_mask.sum()/(H*W):.1f}%)")
    print(f"  - Mean Absolute Error: {mae:.2f}")
    print(f"  - RMSE: {np.sqrt(mse):.2f}")

    if mae < 10 and np.sqrt(mse) < 15:
        print("\n✅ Camera parameters look correct!")
        print("   The 3D reconstruction matches the original image well.")
    elif mae < 30 and np.sqrt(mse) < 40:
        print("\n⚠️  Camera parameters are approximately correct")
        print("   There are some differences, but the overall structure matches.")
    else:
        print("\n❌ Significant differences detected!")
        print("   The camera parameters may be incorrect, or there are issues with")
        print("   the 3D reconstruction. Check the visualization for details.")

    print(f"\nVisualization output:")
    print(f"  - Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug camera projection for a single frame"
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
        help="Frame index to test (default: 0)",
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("DEBUG: Camera Projection Test")
    print(f"{'='*80}")

    try:
        test_single_frame_projection(args.input_dir, args.frame_idx)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
