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
from addict import Dict as AddictDict

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


def save_depth_maps(
    prediction,
    output_dir: str,
    save_raw: bool = True,
    save_visualizations: bool = True,
    save_npz: bool = True,
):
    """
    Save depth estimation results.

    Args:
        prediction: Prediction object from model inference
        output_dir: Output directory to save results
        save_raw: Whether to save raw depth maps as .npy files
        save_visualizations: Whether to save depth visualizations as images
        save_npz: Whether to save all prediction data as .npz file
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

        # Save depth maps
        if prediction.depth is not None:
            save_dict["depths"] = np.round(prediction.depth, 6)

        # Save confidence maps if available
        if prediction.conf is not None:
            save_dict["conf"] = np.round(prediction.conf, 2)

        # Save camera extrinsics (camera poses)
        # Note: Saved as (N, 3, 4) [R|t] format from API
        # When reloading for inference, convert to (N, 4, 4) by adding [0,0,0,1] bottom row
        if prediction.extrinsics is not None:
            save_dict["extrinsics"] = prediction.extrinsics

        # Save camera intrinsics
        if prediction.intrinsics is not None:
            save_dict["intrinsics"] = prediction.intrinsics

        # Save processed images
        if prediction.processed_images is not None:
            save_dict["images"] = prediction.processed_images

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
        default='depth-anything/DA3-Small', # "depth-anything/DA3NESTED-GIANT-LARGE",
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
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
