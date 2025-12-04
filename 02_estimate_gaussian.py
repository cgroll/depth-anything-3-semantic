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
3D Gaussian Splatting estimation script.

This script takes depth estimation outputs from dev.py and generates 3D Gaussian
Splatting representations, saving them as .ply files for use in 3DGS viewers.

Usage:
    python3 02_estimate_gaussian.py
    python3 02_estimate_gaussian.py --input-dir path/to/depth_estimation
    python3 02_estimate_gaussian.py --output path/to/output.ply
"""

import argparse
import gc
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export.gs import export_to_gs_ply


def load_predictions_and_images(predictions_path: str, images_dir: str = None):
    """
    Load predictions from dev.py output.

    Args:
        predictions_path: Path to predictions.npz file
        images_dir: Optional directory containing original images

    Returns:
        Tuple of (prediction data dict, image paths list)
    """
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    print(f"📂 Loading predictions from {predictions_path}")
    data = np.load(predictions_path)

    # If images_dir not provided, try to find it relative to predictions
    if images_dir is None:
        predictions_dir = os.path.dirname(predictions_path)
        # Look for original images in parent directories
        possible_dirs = [
            os.path.join(predictions_dir, "..", "..", "inputs"),
            os.path.join(predictions_dir, "images"),
            predictions_dir,
        ]
        for dir_path in possible_dirs:
            dir_path = os.path.normpath(dir_path)
            if os.path.exists(dir_path):
                image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
                image_paths = []
                for ext in image_extensions:
                    image_paths.extend(glob.glob(os.path.join(dir_path, ext)))
                if len(image_paths) > 0:
                    images_dir = dir_path
                    break

    # Get image paths
    image_paths = []
    if images_dir and os.path.exists(images_dir):
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        image_paths = sorted(image_paths)
        print(f"  ✓ Found {len(image_paths)} images in {images_dir}")
    else:
        print("  ⚠️  No image directory found, using processed images from predictions")
        image_paths = None

    print(f"  ✓ Loaded prediction data:")
    for key in data.keys():
        if hasattr(data[key], 'shape'):
            print(f"    - {key}: {data[key].shape}")
        else:
            print(f"    - {key}: {data[key]}")

    return data, image_paths


def run_gaussian_inference(
    image_paths: list,
    predictions_data: dict,
    model_dir: str = "depth-anything/DA3NESTED-GIANT-LARGE",
    device: str = None,
):
    """
    Run model inference with 3D Gaussian Splatting enabled.

    Args:
        image_paths: List of image file paths
        predictions_data: Previously computed prediction data (with depths, cameras)
        model_dir: Model directory or Hugging Face model ID
        device: Device to run inference on (cuda/cpu)

    Returns:
        Prediction object with Gaussian splatting data
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clear GPU memory at startup
    if device == "cuda" and torch.cuda.is_available():
        print("\n🧹 Clearing GPU memory...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Get current GPU memory stats
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"  GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    print(f"\n🔧 Configuration:")
    print(f"  - Model: {model_dir}")
    print(f"  - Device: {device}")
    print(f"  - Number of images: {len(image_paths)}")

    # Load model
    print(f"\n🔄 Loading model from {model_dir}...")

    # Validate model supports Gaussian Splatting
    model_name_lower = model_dir.lower()
    is_giant = "da3-giant" in model_name_lower or "da3_giant" in model_name_lower
    is_nested = "nested" in model_name_lower

    if not (is_giant or is_nested):
        print(f"\n❌ ERROR: Model '{model_dir}' does NOT support 3D Gaussian Splatting!")
        print("   According to official documentation, only these models support 3DGS:")
        print("   - depth-anything/DA3-GIANT (1.15B params, requires 12-16GB GPU)")
        print("   - depth-anything/DA3NESTED-GIANT-LARGE (1.40B params, requires 18-24GB GPU)")
        print("\n   Other models (DA3-LARGE, DA3-SMALL, DA3MONO-LARGE, DA3METRIC-LARGE)")
        print("   do NOT have the required gs_head and cam_enc modules for 3DGS.")
        print("\n   For 8GB GPUs, use: --device cpu")
        return None

    start_time = time.time()
    model = DepthAnything3.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")

    # Convert predictions_data to expected format for re-inference
    extrinsics = predictions_data.get('extrinsics', None)
    intrinsics = predictions_data.get('intrinsics', None)

    # Convert extrinsics from (N, 3, 4) to (N, 4, 4) if needed
    if extrinsics is not None and extrinsics.shape[1:] == (3, 4):
        print("  ℹ️  Converting extrinsics from (N, 3, 4) to (N, 4, 4) format...")
        # Add bottom row [0, 0, 0, 1] to make it a homogeneous transformation matrix
        N = extrinsics.shape[0]
        extrinsics_4x4 = np.zeros((N, 4, 4), dtype=extrinsics.dtype)
        extrinsics_4x4[:, :3, :] = extrinsics  # Copy [R|t]
        extrinsics_4x4[:, 3, 3] = 1.0  # Set bottom-right to 1
        extrinsics = extrinsics_4x4
        print(f"  ✓ Converted extrinsics to shape: {extrinsics.shape}")

    # Run inference with 3DGS enabled
    print(f"\n🚀 Running 3D Gaussian Splatting inference...")
    inference_start = time.time()

    with torch.no_grad():
        prediction = model.inference(
            image_paths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            export_dir=None,
            process_res_method="lower_bound_resize",  # High-res for better quality
            infer_gs=True,  # Enable 3D Gaussian Splatting
        )

    inference_time = time.time() - inference_start
    print(f"✅ Gaussian inference completed in {inference_time:.2f} seconds")

    # Clean up GPU memory
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    return prediction


def save_gaussian_ply(
    prediction: Prediction,
    output_path: str,
    gs_views_interval: int = 1,
):
    """
    Save 3D Gaussian Splatting as PLY file.

    Args:
        prediction: Prediction object with Gaussian data
        output_path: Output PLY file path
        gs_views_interval: Export GS every N views (default: 1 = all views)
    """
    if prediction.gaussians is None:
        raise ValueError(
            "No Gaussian data in prediction!\n"
            "This usually means:\n"
            "  1. The model doesn't support 3D Gaussian Splatting\n"
            "     → Use --model-dir depth-anything/DA3NESTED-GIANT-LARGE\n"
            "  2. Or infer_gs=True was not enabled (should be automatic in this script)\n"
            "\nSupported models (with 3DGS capability):\n"
            "  - depth-anything/DA3-GIANT (smallest, 1.15B params)\n"
            "  - depth-anything/DA3NESTED-GIANT-LARGE (best, 1.40B params, recommended)\n"
            "  - depth-anything/DA3MONO-LARGE"
        )

    print(f"\n💾 Saving Gaussian Splatting to {output_path}")

    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert depth to tensor format expected by export function
    gs_world = prediction.gaussians
    pred_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(gs_world.means)

    # Prepare output directory for export_to_gs_ply
    temp_export_dir = os.path.dirname(output_path)
    if not temp_export_dir:
        temp_export_dir = "."

    # Export using the library function
    export_to_gs_ply(
        prediction,
        export_dir=temp_export_dir,
        gs_views_interval=gs_views_interval,
    )

    # The export function saves to gs_ply/0000.ply, so we need to rename it
    exported_ply = os.path.join(temp_export_dir, "gs_ply", "0000.ply")
    if os.path.exists(exported_ply):
        # If output path is in gs_ply subfolder, just use it
        if os.path.dirname(output_path).endswith("gs_ply"):
            if exported_ply != output_path:
                os.rename(exported_ply, output_path)
        else:
            # Move to desired output path
            os.rename(exported_ply, output_path)
            # Clean up empty directory
            gs_ply_dir = os.path.join(temp_export_dir, "gs_ply")
            if os.path.exists(gs_ply_dir) and not os.listdir(gs_ply_dir):
                os.rmdir(gs_ply_dir)

    print(f"  ✓ Saved to {output_path}")

    # Print Gaussian statistics
    print(f"\n📊 Gaussian Statistics:")
    print(f"  - Number of Gaussians: {gs_world.means.shape[0]:,}")
    print(f"  - Means shape: {gs_world.means.shape}")
    print(f"  - Scales shape: {gs_world.scales.shape}")
    print(f"  - Rotations shape: {gs_world.rotations.shape}")
    print(f"  - Harmonics shape: {gs_world.harmonics.shape}")
    print(f"  - Opacities shape: {gs_world.opacities.shape}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="3D Gaussian Splatting estimation from depth predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process predictions from dev.py output
  python3 02_estimate_gaussian.py

  # Use custom input directory
  python3 02_estimate_gaussian.py --input-dir path/to/depth_estimation

  # Specify custom output path
  python3 02_estimate_gaussian.py --output path/to/output.ply

  # Use original images directory
  python3 02_estimate_gaussian.py --images-dir data/inputs

  # Use different model
  python3 02_estimate_gaussian.py --model-dir depth-anything/DA3-LARGE

  # CPU mode
  python3 02_estimate_gaussian.py --device cpu

  # Export every 2nd view (for dense sequences)
  python3 02_estimate_gaussian.py --gs-views-interval 2
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/outputs/depth_estimation",
        help="Directory containing predictions.npz from dev.py (default: data/outputs/depth_estimation)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory containing original input images (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/outputs/gaussian_splatting/scene.ply",
        help="Output PLY file path (default: data/outputs/gaussian_splatting/scene.ply)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3-GIANT",
        help="Model directory or Hugging Face model ID (default: depth-anything/DA3-GIANT). "
             "NOTE: Only DA3-GIANT (1.15B, 12-16GB GPU) and DA3NESTED-GIANT-LARGE (1.40B, 18-24GB GPU) support 3D Gaussian Splatting! "
             "Use --device cpu for GPUs with <12GB memory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to run inference on (default: auto-detect)",
    )
    parser.add_argument(
        "--gs-views-interval",
        type=int,
        default=1,
        help="Export Gaussians every N views (default: 1 = all views). Higher values reduce file size.",
    )

    args = parser.parse_args()

    print("="*60)
    print("3D Gaussian Splatting Estimation")
    print("="*60)

    # Initial GPU cleanup and check
    if torch.cuda.is_available() and args.device != "cpu":
        print("\n🧹 Initial GPU cleanup...")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

        # Check GPU memory
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory_gb = gpu_props.total_memory / (1024**3)
        print(f"  ✓ GPU cache cleared")
        print(f"  📊 GPU: {gpu_props.name}")
        print(f"  💾 Total GPU Memory: {total_memory_gb:.1f} GB")

        # Check GPU memory requirements based on model
        model_requirements = {
            "depth-anything/DA3NESTED-GIANT-LARGE": (18, 24),
            "depth-anything/DA3-GIANT": (12, 16),
        }

        if args.model_dir in model_requirements:
            min_mem, max_mem = model_requirements[args.model_dir]
            if total_memory_gb < min_mem:
                print(f"\n⚠️  WARNING: GPU has only {total_memory_gb:.1f}GB memory!")
                print(f"   {args.model_dir.split('/')[-1]} requires ~{min_mem}-{max_mem}GB GPU memory.")
                print("\n💡 Recommended alternatives for your GPU:")
                if total_memory_gb >= 12:
                    print("   - Use: --model-dir depth-anything/DA3-GIANT (requires ~12-16GB)")
                else:
                    print("   - Use CPU mode: --device cpu")
                    print("   - Or skip 3DGS: Use dev.py for depth estimation only")

                response = input("\n   Continue anyway? (y/N): ").strip().lower()
                if response != 'y':
                    print("\n   Exiting. Please run with appropriate model or --device cpu")
                    return 1

    # Find predictions file
    predictions_path = os.path.join(args.input_dir, "predictions.npz")

    try:
        # Load predictions and find images
        predictions_data, image_paths = load_predictions_and_images(
            predictions_path,
            images_dir=args.images_dir,
        )

        # If no images found, check if processed images are in predictions
        if image_paths is None or len(image_paths) == 0:
            if 'images' in predictions_data:
                print("  ⚠️  Using processed images from predictions.npz")
                # We need to save these images temporarily for inference
                temp_images_dir = os.path.join(args.input_dir, "temp_images")
                os.makedirs(temp_images_dir, exist_ok=True)

                image_paths = []
                import imageio
                for i, img in enumerate(predictions_data['images']):
                    img_path = os.path.join(temp_images_dir, f"{i:04d}.png")
                    imageio.imwrite(img_path, img)
                    image_paths.append(img_path)
                print(f"  ✓ Saved {len(image_paths)} temporary images to {temp_images_dir}")
            else:
                raise ValueError(
                    "No images found. Please specify --images-dir or ensure "
                    "predictions.npz contains 'images' field."
                )

        # Run Gaussian inference
        prediction = run_gaussian_inference(
            image_paths=image_paths,
            predictions_data=predictions_data,
            model_dir=args.model_dir,
            device=args.device,
        )

        # Save Gaussian PLY
        save_gaussian_ply(
            prediction=prediction,
            output_path=args.output,
            gs_views_interval=args.gs_views_interval,
        )

        # Clean up temp images if created
        temp_images_dir = os.path.join(args.input_dir, "temp_images")
        if os.path.exists(temp_images_dir):
            import shutil
            shutil.rmtree(temp_images_dir)
            print("  ✓ Cleaned up temporary images")

        # Cleanup CUDA cache
        if args.device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n" + "="*60)
        print("✅ Gaussian Splatting estimation complete!")
        print("="*60)
        print(f"\n📁 Output saved to: {args.output}")
        print("\nYou can view this PLY file in:")
        print("  - 3DGS viewers (e.g., https://antimatter15.com/splat/)")
        print("  - MeshLab, CloudCompare, or other point cloud viewers")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
