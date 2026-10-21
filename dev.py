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
Development script for multi-view depth estimation.

This script processes images from data/inputs and saves depth estimation results
to data/outputs/depth_estimation.

Usage:
    python dev.py
    python dev.py --input-dir custom/input/path --output-dir custom/output/path
    python dev.py --model-dir depth-anything/DA3-LARGE
"""

import argparse
import gc
import glob
import os
import time
from pathlib import Path

import imageio
import numpy as np
import torch
import trimesh
from addict import Dict as AddictDict

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export.glb import (
    export_to_glb,
    _depths_to_world_points_with_colors,
    _filter_and_downsample,
    get_conf_thresh,
    set_sky_depth,
)
from depth_anything_3.utils.visualize import visualize_depth


def _can_export_3d(prediction: Prediction) -> bool:
    """Check if prediction has all required data for 3D export."""
    return (
        prediction.depth is not None
        and prediction.intrinsics is not None
        and prediction.extrinsics is not None
        and prediction.conf is not None
        and prediction.processed_images is not None
    )


def export_point_cloud_ply(
    prediction: Prediction,
    output_path: str,
    num_max_points: int = 1_000_000,
    conf_thresh: float = 1.05,
    conf_thresh_percentile: float = 40.0,
    ensure_thresh_percentile: float = 90.0,
    sky_depth_def: float = 98.0,
):
    """
    Export point cloud to PLY file (without camera poses).

    Args:
        prediction: Prediction object with depth, cameras, and images
        output_path: Output PLY file path
        num_max_points: Maximum number of points after downsampling
        conf_thresh: Base confidence threshold
        conf_thresh_percentile: Lower percentile for adaptive threshold
        ensure_thresh_percentile: Upper percentile clamp for threshold
        sky_depth_def: Percentile for sky depth values
    """
    # Ensure required data is available
    assert prediction.depth is not None, "Depth is required"
    assert prediction.intrinsics is not None, "Intrinsics are required"
    assert prediction.extrinsics is not None, "Extrinsics are required"
    assert prediction.conf is not None, "Confidence is required"
    assert prediction.processed_images is not None, "Processed images are required"

    # Sky processing
    if getattr(prediction, "sky_mask", None) is not None:
        set_sky_depth(prediction, prediction.sky_mask, sky_depth_def)

    # Get adaptive confidence threshold
    conf_thr = get_conf_thresh(
        prediction,
        getattr(prediction, "sky_mask", None),
        conf_thresh,
        conf_thresh_percentile,
        ensure_thresh_percentile,
    )

    # Back-project to world coordinates
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,
        prediction.processed_images,
        prediction.conf,
        conf_thr,
    )

    # Downsample
    points, colors = _filter_and_downsample(points, colors, num_max_points)

    # Create point cloud and save
    if points.shape[0] > 0:
        pc = trimesh.points.PointCloud(vertices=points, colors=colors)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pc.export(output_path)
        return output_path
    else:
        raise ValueError("No valid points to export")


def save_depth_maps(
    prediction,
    output_dir: str,
    save_raw: bool = True,
    save_visualizations: bool = True,
    save_npz: bool = True,
    save_glb: bool = True,
    save_ply: bool = True,
):
    """
    Save depth estimation results.

    Args:
        prediction: Prediction object from model inference
        output_dir: Output directory to save results
        save_raw: Whether to save raw depth maps as .npy files
        save_visualizations: Whether to save depth visualizations as images
        save_npz: Whether to save all prediction data as .npz file
        save_glb: Whether to save GLB file with cameras and point cloud
        save_ply: Whether to save PLY file with point cloud only
    """
    os.makedirs(output_dir, exist_ok=True)

    num_images = prediction.depth.shape[0]
    print(f"\n📊 Saving {num_images} depth maps to {output_dir}")

    # Save raw depth maps
    if save_raw:
        raw_dir = os.path.join(output_dir, "depth_raw")
        os.makedirs(raw_dir, exist_ok=True)
        for idx in range(num_images):
            depth_path = os.path.join(raw_dir, f"{idx:04d}.npy")
            np.save(depth_path, prediction.depth[idx])
        print(f"  ✓ Saved raw depth maps to {raw_dir}/")

    # Save depth visualizations
    if save_visualizations and prediction.processed_images is not None:
        vis_dir = os.path.join(output_dir, "depth_vis")
        os.makedirs(vis_dir, exist_ok=True)

        for idx in range(num_images):
            # Create depth visualization
            depth_vis = visualize_depth(prediction.depth[idx])
            image_vis = prediction.processed_images[idx]

            # Convert to uint8
            depth_vis = depth_vis.astype(np.uint8)
            image_vis = image_vis.astype(np.uint8)

            # Concatenate image and depth side by side
            vis_image = np.concatenate([image_vis, depth_vis], axis=1)

            # Save visualization
            save_path = os.path.join(vis_dir, f"{idx:04d}.jpg")
            imageio.imwrite(save_path, vis_image, quality=95)

        print(f"  ✓ Saved depth visualizations to {vis_dir}/")

    # Save camera parameters and all prediction data
    if save_npz:
        npz_path = os.path.join(output_dir, "predictions.npz")

        save_dict = {}

        # Helper function to convert to numpy array safely
        def to_numpy(arr):
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                return arr
            if hasattr(arr, 'cpu'):  # torch tensor
                return arr.cpu().numpy()
            return np.asarray(arr)

        # Save depth maps
        if prediction.depth is not None:
            save_dict["depths"] = np.round(to_numpy(prediction.depth), 6)

        # Save confidence maps if available
        if prediction.conf is not None:
            save_dict["conf"] = np.round(to_numpy(prediction.conf), 2)

        # Save camera extrinsics (camera poses)
        # Note: Saved as (N, 3, 4) [R|t] format from API
        # When reloading for inference, convert to (N, 4, 4) by adding [0,0,0,1] bottom row
        if prediction.extrinsics is not None:
            save_dict["extrinsics"] = to_numpy(prediction.extrinsics)

        # Save camera intrinsics
        if prediction.intrinsics is not None:
            save_dict["intrinsics"] = to_numpy(prediction.intrinsics)

        # Save processed images
        if prediction.processed_images is not None:
            save_dict["images"] = to_numpy(prediction.processed_images)

        # Save metric flag (handle both int/bool and addict.Dict types)
        if prediction.is_metric is not None:
            if isinstance(prediction.is_metric, (dict, AddictDict)):
                # If it's a dict or addict.Dict, treat as False (0)
                # Empty addict.Dict means not metric
                save_dict["is_metric"] = 0
            else:
                save_dict["is_metric"] = int(prediction.is_metric)
        else:
            save_dict["is_metric"] = 0

        # Save scale factor if available
        if prediction.scale_factor is not None:
            if isinstance(prediction.scale_factor, dict):
                # Skip dict scale factors
                pass
            else:
                save_dict["scale_factor"] = float(prediction.scale_factor)

        np.savez_compressed(npz_path, **save_dict)
        print(f"  ✓ Saved predictions data to {npz_path}")

    # Export GLB (3D scene with cameras and point cloud)
    if save_glb and _can_export_3d(prediction):
        try:
            print(f"\n🎨 Exporting GLB (point cloud + cameras)...")
            glb_path = export_to_glb(
                prediction,
                output_dir,
                num_max_points=1_000_000,
                show_cameras=True,
                camera_size=0.03,
                export_depth_vis=False,  # Already saved above
            )
            print(f"  ✓ Saved GLB to {glb_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to export GLB: {e}")

    # Export PLY (point cloud only)
    if save_ply and _can_export_3d(prediction):
        try:
            print(f"\n🎨 Exporting PLY (point cloud only)...")
            ply_path = os.path.join(output_dir, "point_cloud.ply")
            export_point_cloud_ply(prediction, ply_path)
            print(f"  ✓ Saved PLY to {ply_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to export PLY: {e}")

    # Print summary
    print(f"\n📈 Prediction Summary:")
    print(f"  - Number of views: {num_images}")
    print(f"  - Depth shape: {prediction.depth.shape}")
    print(f"  - Is metric: {bool(prediction.is_metric)}")
    if prediction.scale_factor is not None:
        print(f"  - Scale factor: {prediction.scale_factor:.4f}")
    if prediction.extrinsics is not None:
        print(f"  - Camera extrinsics: {prediction.extrinsics.shape}")
    if prediction.intrinsics is not None:
        print(f"  - Camera intrinsics: {prediction.intrinsics.shape}")


def process_images(
    input_dir: str,
    output_dir: str,
    model_dir: str = "depth-anything/DA3NESTED-GIANT-LARGE",
    process_res_method: str = "lower_bound_resize",
    device: str = None,
    save_glb: bool = True,
    save_ply: bool = True,
):
    """
    Process images from input directory and save depth estimation results.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output results
        model_dir: Model directory or Hugging Face model ID
        process_res_method: Image processing method
            - "lower_bound_resize": High resolution (slower, better quality)
            - "upper_bound_resize": Low resolution (faster, lower quality)
            - "upper_bound_crop": Crop to fit resolution
        device: Device to run inference on (cuda/cpu). Auto-detected if None.
        save_glb: Whether to export GLB file with cameras and point cloud
        save_ply: Whether to export PLY file with point cloud only
    """
    # Check input directory
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Get list of images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {input_dir}")

    print(f"\n🖼️  Found {len(image_paths)} images in {input_dir}")
    for i, path in enumerate(image_paths[:5]):  # Show first 5
        print(f"  {i+1}. {os.path.basename(path)}")
    if len(image_paths) > 5:
        print(f"  ... and {len(image_paths) - 5} more")

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clear GPU memory at startup
    if device == "cuda" and torch.cuda.is_available():
        print("\n🧹 Clearing GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"  GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    print(f"\n🔧 Configuration:")
    print(f"  - Model: {model_dir}")
    print(f"  - Device: {device}")
    print(f"  - Processing method: {process_res_method}")
    print(f"  - Output directory: {output_dir}")

    # Load model
    print(f"\n🔄 Loading model from {model_dir}...")
    start_time = time.time()
    model = DepthAnything3.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")

    # Run inference
    print(f"\n🚀 Running multi-view depth estimation...")
    inference_start = time.time()

    with torch.no_grad():
        prediction = model.inference(
            image_paths,
            export_dir=None,
            process_res_method=process_res_method,
            infer_gs=False,  # Don't compute 3D Gaussian Splatting for faster processing
        )

    inference_time = time.time() - inference_start
    print(f"✅ Inference completed in {inference_time:.2f} seconds")
    print(f"   ({inference_time / len(image_paths):.2f} seconds per image)")

    # Save results
    save_depth_maps(
        prediction,
        output_dir,
        save_raw=True,
        save_visualizations=True,
        save_npz=True,
        save_glb=save_glb,
        save_ply=save_ply,
    )

    # Cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - start_time
    print(f"\n✅ Processing complete in {total_time:.2f} seconds")
    print(f"📁 Results saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-view depth estimation with Depth Anything 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process images from data/inputs to data/outputs/depth_estimation
  python dev.py

  # Use custom input/output directories
  python dev.py --input-dir path/to/images --output-dir path/to/output

  # Use a different model
  python dev.py --model-dir depth-anything/DA3-LARGE

  # Use high-resolution processing (slower but better quality)
  python dev.py --high-res

  # Use CPU instead of GPU
  python dev.py --device cpu
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/inputs",
        help="Directory containing input images (default: data/inputs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs/depth_estimation",
        help="Directory to save output results (default: data/outputs/depth_estimation)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default='depth-anything/DA3-Base', # "depth-anything/DA3NESTED-GIANT-LARGE",
        help="Model directory or Hugging Face model ID (default: depth-anything/DA3NESTED-GIANT-LARGE)",
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Use high-resolution processing (slower but better quality)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to run inference on (default: auto-detect)",
    )
    parser.add_argument(
        "--no-glb",
        action="store_true",
        help="Skip GLB export (point cloud with cameras)",
    )
    parser.add_argument(
        "--no-ply",
        action="store_true",
        help="Skip PLY export (point cloud only)",
    )

    args = parser.parse_args()

    # Determine processing method
    process_res_method = "lower_bound_resize" if args.high_res else "upper_bound_resize"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process images
    try:
        process_images(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            process_res_method=process_res_method,
            device=args.device,
            save_glb=not args.no_glb,
            save_ply=not args.no_ply,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
