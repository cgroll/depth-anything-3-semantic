#!/usr/bin/env python3
"""
Example usage of dev.py outputs for downstream tasks.

This script demonstrates how to load and use the depth estimation results
produced by dev.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_predictions(predictions_path: str):
    """Load predictions from npz file."""
    data = np.load(predictions_path)

    print("Available data keys:", list(data.keys()))
    print("\nData shapes:")
    for key in data.keys():
        if hasattr(data[key], 'shape'):
            print(f"  {key}: {data[key].shape}")
        else:
            print(f"  {key}: {data[key]}")

    return data


def visualize_depth_maps(data, output_dir="visualizations"):
    """Visualize depth maps."""
    depths = data['depths']
    num_views = depths.shape[0]

    Path(output_dir).mkdir(exist_ok=True)

    # Create a grid of depth visualizations
    fig, axes = plt.subplots(1, min(num_views, 3), figsize=(15, 5))
    if num_views == 1:
        axes = [axes]

    for i in range(min(num_views, 3)):
        im = axes[i].imshow(depths[i], cmap='turbo')
        axes[i].set_title(f'View {i}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = Path(output_dir) / "depth_maps.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved depth visualization to: {save_path}")
    plt.close()


def analyze_camera_poses(data):
    """Analyze camera poses from extrinsics."""
    if 'extrinsics' not in data:
        print("No camera extrinsics available")
        return

    extrinsics = data['extrinsics']  # (N, 4, 4)
    intrinsics = data['intrinsics']  # (N, 3, 3)

    print("\n" + "="*60)
    print("Camera Analysis")
    print("="*60)

    # Extract camera positions
    camera_positions = extrinsics[:, :3, 3]

    print(f"\nNumber of views: {len(camera_positions)}")
    print(f"\nCamera positions:")
    for i, pos in enumerate(camera_positions):
        print(f"  View {i}: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")

    # Compute camera trajectory statistics
    if len(camera_positions) > 1:
        distances = np.linalg.norm(np.diff(camera_positions, axis=0), axis=1)
        print(f"\nCamera trajectory:")
        print(f"  Total path length: {distances.sum():.3f} units")
        print(f"  Average distance between views: {distances.mean():.3f} units")
        print(f"  Min distance: {distances.min():.3f} units")
        print(f"  Max distance: {distances.max():.3f} units")

    # Extract focal lengths
    print(f"\nCamera intrinsics (focal lengths):")
    for i in range(len(intrinsics)):
        fx = intrinsics[i, 0, 0]
        fy = intrinsics[i, 1, 1]
        cx = intrinsics[i, 0, 2]
        cy = intrinsics[i, 1, 2]
        print(f"  View {i}: fx={fx:7.2f}, fy={fy:7.2f}, cx={cx:7.2f}, cy={cy:7.2f}")


def depth_statistics(data):
    """Compute and display depth statistics."""
    depths = data['depths']

    print("\n" + "="*60)
    print("Depth Statistics")
    print("="*60)

    for i in range(len(depths)):
        depth = depths[i]
        print(f"\nView {i}:")
        print(f"  Shape: {depth.shape}")
        print(f"  Min depth: {depth.min():.3f}")
        print(f"  Max depth: {depth.max():.3f}")
        print(f"  Mean depth: {depth.mean():.3f}")
        print(f"  Median depth: {np.median(depth):.3f}")

        # Compute depth percentiles
        p25, p75 = np.percentile(depth, [25, 75])
        print(f"  25th percentile: {p25:.3f}")
        print(f"  75th percentile: {p75:.3f}")


def export_to_custom_format(data, output_path: str):
    """Example: Export to custom format for your application."""
    # Example: Export first view's depth as a simple text file
    depths = data['depths']

    # Save as simple text (for debugging/inspection)
    np.savetxt(output_path, depths[0], fmt='%.6f')
    print(f"\nExported first depth map to: {output_path}")


def main():
    """Main example."""
    import sys

    # Default path
    predictions_path = "data/outputs/depth_estimation/predictions.npz"

    # Check if custom path provided
    if len(sys.argv) > 1:
        predictions_path = sys.argv[1]

    # Check if file exists
    if not Path(predictions_path).exists():
        print(f"Error: Predictions file not found: {predictions_path}")
        print("\nPlease run dev.py first to generate predictions:")
        print("  python3 dev.py")
        print("\nOr specify a custom path:")
        print(f"  python3 {sys.argv[0]} path/to/predictions.npz")
        return 1

    print("="*60)
    print("Depth Anything 3 - Prediction Analysis")
    print("="*60)
    print(f"\nLoading predictions from: {predictions_path}\n")

    # Load predictions
    data = load_predictions(predictions_path)

    # Analyze camera poses
    analyze_camera_poses(data)

    # Compute depth statistics
    depth_statistics(data)

    # Visualize depth maps
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    visualize_depth_maps(data)

    # Export example
    export_to_custom_format(data, "depth_view0.txt")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

    return 0


if __name__ == "__main__":
    exit(main())
