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
Semantic 3D reconstruction script - Part 1: Extract masks and compute depth.

This script:
1. Loads a video and extracts frames
2. Runs YOLO instance segmentation on each frame
3. Extracts the mask for the primary detected object (chair)
4. Runs Depth Anything 3 multi-view depth estimation
5. Saves all results to disk for later processing

Usage:
    python 04_semantic_3d.py
    python 04_semantic_3d.py --video-path data/inputs/videos/custom.mp4
"""

import argparse
import gc
import os
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import trimesh
from addict import Dict as AddictDict
from ultralytics import YOLO

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


def extract_frames_from_video(video_path: str, max_frames: int = None, frames_per_second: float = 2.0):
    """
    Extract frames from video file at a specified sampling rate.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (None for all frames)
        frames_per_second: Number of frames to sample per second (default: 2.0)

    Returns:
        frames: List of frames as numpy arrays (BGR format)
        video_info: Dictionary with video metadata
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n🎥 Video properties: {width}x{height} @ {fps} FPS")
    print(f"   Total frames in video: {total_frames}")
    print(f"   Sampling rate: {frames_per_second} frames/second")

    # Calculate frame interval for sampling
    frame_interval = max(1, int(fps / frames_per_second))
    print(f"   Taking every {frame_interval}th frame")

    frames = []
    frame_count = 0
    frames_extracted = 0

    # Calculate expected number of frames
    expected_frames = int(total_frames / frame_interval)
    if max_frames is not None:
        expected_frames = min(expected_frames, max_frames)
    print(f"   Expected to extract ~{expected_frames} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"   Reached end of video at frame {frame_count}")
            break

        # Sample frame at interval
        if frame_count % frame_interval == 0:
            frames.append(frame)
            frames_extracted += 1

            # Check if we've reached max_frames
            if max_frames is not None and frames_extracted >= max_frames:
                print(f"   Reached max_frames limit ({max_frames})")
                break

            # Progress indicator
            if frames_extracted % 20 == 0:
                print(f"   Extracted {frames_extracted} frames...")

        frame_count += 1

    cap.release()
    print(f"✅ Extracted {len(frames)} frames from {frame_count} total frames")

    video_info = {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "extracted_frames": len(frames),
        "frame_interval": frame_interval,
        "frames_per_second": frames_per_second,
    }

    return frames, video_info


def run_yolo_segmentation(frames, model_path: str = "yolov8m-seg.pt"):
    """
    Run YOLO instance segmentation on frames and extract primary object masks.

    Args:
        frames: List of frames as numpy arrays
        model_path: Path to YOLO model weights

    Returns:
        masks: List of binary masks (H, W) for the primary detected object
        segmentation_results: List of full YOLO results for each frame
    """
    print(f"\n🔄 Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    print("✅ YOLO model loaded")

    print(f"\n🔍 Running instance segmentation on {len(frames)} frames...")
    masks = []
    segmentation_results = []

    for i, frame in enumerate(frames):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   Processing frame {i+1}/{len(frames)}...")

        results = model(frame, verbose=False)
        result = results[0]
        segmentation_results.append(result)

        # Extract mask for the primary object (highest confidence)
        if result.masks is not None and len(result.masks) > 0:
            # Get the object with highest confidence
            boxes = result.boxes.data.cpu().numpy()
            confidences = boxes[:, 4]
            best_idx = np.argmax(confidences)

            # Extract and resize mask
            mask_data = result.masks.data.cpu().numpy()[best_idx]
            height, width = frame.shape[:2]
            mask_resized = cv2.resize(mask_data, (width, height))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            # Get class name for logging
            cls_id = int(boxes[best_idx, 5])
            cls_name = model.names[cls_id]
            conf = confidences[best_idx]

            if (i + 1) % 10 == 0 or i == 0:
                print(f"     Detected: {cls_name} (confidence: {conf:.2f})")

            masks.append(mask_binary)
        else:
            # No objects detected - create empty mask
            height, width = frame.shape[:2]
            masks.append(np.zeros((height, width), dtype=np.uint8))
            if (i + 1) % 10 == 0 or i == 0:
                print(f"     No objects detected")

    print(f"✅ Segmentation complete")

    # Clear GPU memory after YOLO processing
    if torch.cuda.is_available():
        print("\n🧹 Clearing GPU memory after YOLO...")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return masks, segmentation_results


def save_frames_to_temp_dir(frames, temp_dir: str):
    """
    Save frames to temporary directory for Depth Anything 3 processing.

    Args:
        frames: List of frames as numpy arrays (BGR format)
        temp_dir: Directory to save temporary frames

    Returns:
        image_paths: List of paths to saved images
    """
    os.makedirs(temp_dir, exist_ok=True)

    print(f"\n💾 Saving frames to {temp_dir}...")
    image_paths = []

    for i, frame in enumerate(frames):
        # Convert BGR to RGB for saving
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_path = os.path.join(temp_dir, f"{i:04d}.jpg")
        imageio.imwrite(image_path, frame_rgb, quality=95)
        image_paths.append(image_path)

    print(f"✅ Saved {len(image_paths)} frames")
    return image_paths


def run_depth_estimation(
    image_paths,
    model_dir: str = "depth-anything/DA3-Small",
    process_res_method: str = "upper_bound_resize",
    device: str = None,
):
    """
    Run Depth Anything 3 multi-view depth estimation.

    Args:
        image_paths: List of paths to input images
        model_dir: Model directory or Hugging Face model ID
        process_res_method: Image processing method
        device: Device to run inference on (cuda/cpu)

    Returns:
        prediction: Prediction object with depth maps, camera poses, etc.
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clear GPU memory at startup
    if device == "cuda" and torch.cuda.is_available():
        print("\n🧹 Clearing GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    print(f"\n🔧 Depth estimation configuration:")
    print(f"   - Model: {model_dir}")
    print(f"   - Device: {device}")
    print(f"   - Processing method: {process_res_method}")

    # Load model
    print(f"\n🔄 Loading Depth Anything 3 model...")
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
            infer_gs=False,  # Don't compute 3D Gaussian Splatting
        )

    inference_time = time.time() - inference_start
    print(f"✅ Inference completed in {inference_time:.2f} seconds")
    print(f"   ({inference_time / len(image_paths):.2f} seconds per image)")

    # Cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    return prediction


def create_point_cloud_glb(depths, intrinsics, extrinsics, images, output_path: str):
    """
    Create a 3D point cloud from depth maps and save as GLB.

    Args:
        depths: (N, H, W) depth maps
        intrinsics: (N, 3, 3) camera intrinsic matrices
        extrinsics: (N, 3, 4) camera extrinsic matrices [R|t]
        images: (N, H, W, 3) RGB images
        output_path: Path to save GLB file
    """
    N = len(depths)
    all_points = []
    all_colors = []

    for frame_idx in range(N):
        depth = depths[frame_idx]
        K = intrinsics[frame_idx]
        extrinsic = extrinsics[frame_idx]
        image = images[frame_idx]

        H, W = depth.shape

        # Create pixel grid
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth.flatten()

        # Filter valid depths
        valid = depth_flat > 0
        u_valid = u_flat[valid]
        v_valid = v_flat[valid]
        depth_valid = depth_flat[valid]

        # Unproject to camera coordinates
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x_cam = (u_valid - cx) * depth_valid / fx
        y_cam = (v_valid - cy) * depth_valid / fy
        z_cam = depth_valid

        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Transform to world coordinates
        # Extrinsic matrix: camera = R @ world + t
        # Inverse: world = R^T @ (camera - t)
        R = extrinsic[:, :3]
        t = extrinsic[:, 3]
        points_world = (R.T @ (points_cam - t).T).T

        # Get colors
        colors = image[v_valid, u_valid, :]

        all_points.append(points_world)
        all_colors.append(colors)

    # Concatenate all points
    points_3d = np.vstack(all_points)
    colors_3d = np.vstack(all_colors)

    # Create point cloud and save
    point_cloud = trimesh.points.PointCloud(
        vertices=points_3d,
        colors=colors_3d,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    point_cloud.export(output_path)

    print(f"    Created point cloud with {len(points_3d)} points")
    print(f"    World bounds:")
    print(f"      X: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}]")
    print(f"      Y: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}]")
    print(f"      Z: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]")


def save_results(
    masks,
    prediction,
    video_info,
    output_dir: str,
):
    """
    Save all results to disk.

    Args:
        masks: List of binary masks for primary object
        prediction: Prediction object from Depth Anything 3
        video_info: Dictionary with video metadata
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n💾 Saving results to {output_dir}...")

    # Save masks
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        mask_path = os.path.join(masks_dir, f"{i:04d}.png")
        # Scale to 0-255 for visualization
        imageio.imwrite(mask_path, (mask * 255).astype(np.uint8))
    print(f"  ✓ Saved {len(masks)} masks to {masks_dir}/")

    # Save depth maps (raw)
    depth_dir = os.path.join(output_dir, "depth_raw")
    os.makedirs(depth_dir, exist_ok=True)
    for i in range(prediction.depth.shape[0]):
        depth_path = os.path.join(depth_dir, f"{i:04d}.npy")
        np.save(depth_path, prediction.depth[i])
    print(f"  ✓ Saved depth maps to {depth_dir}/")

    # Save depth visualizations
    if prediction.processed_images is not None:
        vis_dir = os.path.join(output_dir, "depth_vis")
        os.makedirs(vis_dir, exist_ok=True)

        for i in range(prediction.depth.shape[0]):
            depth_vis = visualize_depth(prediction.depth[i])
            image_vis = prediction.processed_images[i]

            depth_vis = depth_vis.astype(np.uint8)
            image_vis = image_vis.astype(np.uint8)

            # Concatenate image and depth side by side
            vis_image = np.concatenate([image_vis, depth_vis], axis=1)

            save_path = os.path.join(vis_dir, f"{i:04d}.jpg")
            imageio.imwrite(save_path, vis_image, quality=95)

        print(f"  ✓ Saved depth visualizations to {vis_dir}/")

    # Save all prediction data as NPZ
    npz_path = os.path.join(output_dir, "predictions.npz")

    save_dict = {
        "depths": np.round(prediction.depth, 6),
        "masks": np.array(masks),
    }

    # Save video info as separate scalar arrays (numpy-compatible)
    if video_info is not None:
        save_dict["video_fps"] = np.array(video_info["fps"])
        save_dict["video_width"] = np.array(video_info["width"])
        save_dict["video_height"] = np.array(video_info["height"])
        save_dict["video_total_frames"] = np.array(video_info["total_frames"])
        save_dict["video_extracted_frames"] = np.array(video_info["extracted_frames"])
        save_dict["video_frame_interval"] = np.array(video_info["frame_interval"])
        save_dict["video_frames_per_second"] = np.array(video_info["frames_per_second"])

    if prediction.conf is not None:
        save_dict["conf"] = np.round(prediction.conf, 2)

    if prediction.extrinsics is not None:
        save_dict["extrinsics"] = prediction.extrinsics

    if prediction.intrinsics is not None:
        save_dict["intrinsics"] = prediction.intrinsics

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

    # Save 3D point cloud as GLB
    print(f"\n🎨 Creating 3D point cloud...")
    glb_path = os.path.join(output_dir, "point_cloud.glb")
    create_point_cloud_glb(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,
        prediction.processed_images,
        glb_path,
    )
    print(f"  ✓ Saved 3D point cloud to {glb_path}")

    # Print summary
    print(f"\n📈 Results Summary:")
    print(f"  - Number of frames: {len(masks)}")
    print(f"  - Depth shape: {prediction.depth.shape}")
    print(f"  - Masks shape: {np.array(masks).shape}")
    print(f"  - Is metric: {bool(prediction.is_metric)}")
    if prediction.scale_factor is not None:
        print(f"  - Scale factor: {prediction.scale_factor:.4f}")
    if prediction.extrinsics is not None:
        print(f"  - Camera extrinsics: {prediction.extrinsics.shape}")
    if prediction.intrinsics is not None:
        print(f"  - Camera intrinsics: {prediction.intrinsics.shape}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic 3D reconstruction - Part 1: Extract masks and compute depth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default video (2 frames per second by default)
  python 04_semantic_3d.py

  # Use custom video
  python 04_semantic_3d.py --video-path path/to/video.mp4

  # Sample at different rate (e.g., 1 frame per second)
  python 04_semantic_3d.py --frames-per-second 1.0

  # Process only first 30 frames
  python 04_semantic_3d.py --max-frames 30

  # Use different model
  python 04_semantic_3d.py --model-dir depth-anything/DA3-LARGE
        """,
    )

    parser.add_argument(
        "--video-path",
        type=str,
        default="data/inputs/videos/short.mp4",
        help="Path to input video (default: data/inputs/videos/short.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs/semantic_3d",
        help="Directory to save output results (default: data/outputs/semantic_3d)",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="data/temp/frames",
        help="Temporary directory for extracted frames (default: data/temp/frames)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8m-seg.pt",
        help="YOLO model weights path (default: yolov8m-seg.pt)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3-Small",
        help="Depth model directory or Hugging Face ID (default: depth-anything/DA3-Small)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: None - all frames)",
    )
    parser.add_argument(
        "--frames-per-second",
        type=float,
        default=2.0,
        help="Number of frames to sample per second (default: 2.0)",
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

    # Check video exists
    if not os.path.exists(args.video_path):
        print(f"\n❌ Error: Video not found at {args.video_path}")
        return 1

    print(f"\n{'='*80}")
    print("SEMANTIC 3D RECONSTRUCTION - PART 1")
    print(f"{'='*80}")
    print(f"Video: {args.video_path}")
    print(f"Output: {args.output_dir}")

    start_time = time.time()

    try:
        # Step 1: Extract frames from video
        frames, video_info = extract_frames_from_video(
            args.video_path,
            max_frames=args.max_frames,
            frames_per_second=args.frames_per_second,
        )

        # Step 2: Run YOLO instance segmentation
        masks, segmentation_results = run_yolo_segmentation(
            frames,
            model_path=args.yolo_model,
        )

        # Step 3: Save frames to temporary directory
        image_paths = save_frames_to_temp_dir(frames, args.temp_dir)

        # Step 4: Run depth estimation
        process_res_method = "lower_bound_resize" if args.high_res else "upper_bound_resize"
        prediction = run_depth_estimation(
            image_paths,
            model_dir=args.model_dir,
            process_res_method=process_res_method,
            device=args.device,
        )

        # Step 5: Save all results
        save_results(
            masks,
            prediction,
            video_info,
            args.output_dir,
        )

        total_time = time.time() - start_time
        print(f"\n✅ Processing complete in {total_time:.2f} seconds")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"\nNext step: Run 05_semantic_3d_part2.py to generate 3D visualization")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
