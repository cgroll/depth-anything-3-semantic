# Quick Start Guide - Depth Anything 3 Development Scripts

This guide provides a quick reference for using the development scripts for multi-view depth estimation and 3D Gaussian Splatting.

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch with CUDA support
- Required packages (install via `pip install -r requirements.txt`)

## 🚀 Quick Start

### Option 1: Complete Pipeline (Depth + Gaussian Splatting)

```bash
# 1. Place your images in data/inputs/
cp /path/to/your/images/* data/inputs/

# 2. Run depth estimation
python3 dev.py

# 3. Generate 3D Gaussian Splatting
python3 02_estimate_gaussian.py

# 4. View results in a 3DGS viewer (e.g., https://antimatter15.com/splat/)
# Upload: data/outputs/gaussian_splatting/scene.ply
```

### Option 2: Depth Estimation Only

```bash
# Place images and run
cp /path/to/your/images/* data/inputs/
python3 dev.py

# Results in: data/outputs/depth_estimation/
```

## 📁 Directory Structure

```
depth-anything-3/
├── dev.py                          # Main depth estimation script
├── 02_estimate_gaussian.py         # 3D Gaussian Splatting script
├── example_usage.py                # Example for analyzing results
├── data/
│   ├── inputs/                     # Place your input images here
│   └── outputs/
│       ├── depth_estimation/       # Depth maps and camera data
│       │   ├── depth_raw/          # Raw depth maps (.npy)
│       │   ├── depth_vis/          # Depth visualizations (.jpg)
│       │   └── predictions.npz     # All prediction data
│       └── gaussian_splatting/     # 3D Gaussian Splatting outputs
│           └── scene.ply           # PLY file for 3DGS viewers
```

## 🔧 Common Commands

### Depth Estimation (dev.py)

**Outputs**: Depth maps (.npy), visualizations (.jpg), camera data (.npz), **GLB file (3D scene)**, and **PLY file (point cloud)**

```bash
# Basic usage (generates depth maps, GLB, and PLY files)
python3 dev.py

# High-resolution processing (better quality, slower)
python3 dev.py --high-res

# Use smaller/faster model
python3 dev.py --model-dir depth-anything/DA3-Small

# Custom input/output directories
python3 dev.py --input-dir path/to/images --output-dir path/to/output

# Skip 3D exports for faster processing
python3 dev.py --no-glb  # Skip GLB export (scene with cameras)
python3 dev.py --no-ply  # Skip PLY export (point cloud only)
python3 dev.py --no-glb --no-ply  # Skip both 3D exports

# CPU mode (no GPU)
python3 dev.py --device cpu
```

### 3D Gaussian Splatting (02_estimate_gaussian.py)

**⚠️ Important**: Only works with DA3-GIANT, DA3NESTED, or DA3MONO models!

**GPU Requirements**: Minimum 8GB GPU (12GB+ recommended)

```bash
# Basic usage (uses DA3MONO-LARGE, optimized for 8-12GB GPUs)
python3 02_estimate_gaussian.py

# Custom paths
python3 02_estimate_gaussian.py \
    --input-dir path/to/depth_estimation \
    --output path/to/scene.ply

# Use best quality model (requires 16GB+ GPU)
python3 02_estimate_gaussian.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE

# Use CPU mode (for GPUs with <8GB)
python3 02_estimate_gaussian.py --device cpu

# Specify images directory explicitly
python3 02_estimate_gaussian.py --images-dir data/inputs

# Reduce file size for dense sequences (export every 2nd view)
python3 02_estimate_gaussian.py --gs-views-interval 2
```

### Analyze Results (example_usage.py)

```bash
# Analyze depth estimation results
python3 example_usage.py

# Analyze custom results
python3 example_usage.py path/to/predictions.npz
```

## 📊 Output Formats

### Depth Estimation Outputs (dev.py)

1. **depth_raw/** - Raw depth maps
   - Format: `.npy` (NumPy arrays)
   - Load: `depth = np.load('depth_raw/0000.npy')`
   - Values: Metric depth in meters

2. **depth_vis/** - Visualizations
   - Format: `.jpg` images
   - Content: Input image + colored depth map side-by-side

3. **predictions.npz** - Complete data
   - Contains: depths, cameras (extrinsics, intrinsics), images, confidence
   - Load: `data = np.load('predictions.npz')`

4. **scene.glb** - 3D scene with cameras and point cloud
   - Format: GLB (Binary glTF)
   - Contains: Point cloud + camera wireframes
   - View: Drag-and-drop into 3D viewers (Blender, online glTF viewers)

5. **point_cloud.ply** - Point cloud only
   - Format: PLY (Polygon File Format)
   - Contains: 3D points with RGB colors
   - View: MeshLab, CloudCompare, Open3D

### Gaussian Splatting Outputs (02_estimate_gaussian.py)

1. **scene.ply** - 3D Gaussian Splatting file
   - Format: PLY (Polygon File Format)
   - Contains: Gaussian means, scales, rotations, colors, opacities
   - View: Upload to 3DGS viewers (https://antimatter15.com/splat/)

## 🎯 Use Cases & Examples

### 1. Multi-View 3D Reconstruction

```bash
# Capture photos of an object from multiple angles
# Save to data/inputs/

# Run complete pipeline
python3 dev.py --high-res
python3 02_estimate_gaussian.py

# View in 3DGS viewer for novel view synthesis
```

### 2. Scene Understanding

```bash
# Process a sequence of images
python3 dev.py --input-dir scene_images/

# Analyze camera poses and depth statistics
python3 example_usage.py data/outputs/depth_estimation/predictions.npz
```

### 3. Batch Processing

```bash
# Process multiple scenes
for scene in scene1 scene2 scene3; do
    python3 dev.py \
        --input-dir data/inputs/$scene \
        --output-dir data/outputs/$scene/depth

    python3 02_estimate_gaussian.py \
        --input-dir data/outputs/$scene/depth \
        --output data/outputs/$scene/scene.ply
done
```

### 4. Video to 3D

```bash
# Extract frames from video first (using ffmpeg)
ffmpeg -i input.mp4 -vf fps=2 data/inputs/frame_%04d.png

# Run pipeline
python3 dev.py
python3 02_estimate_gaussian.py
```

## 🎨 Viewing Results

### Depth Maps

```python
import numpy as np
import matplotlib.pyplot as plt

# Load depth map
depth = np.load('data/outputs/depth_estimation/depth_raw/0000.npy')

# Visualize
plt.imshow(depth, cmap='turbo')
plt.colorbar(label='Depth (meters)')
plt.show()
```

### 3D Gaussian Splatting

**Web Viewers (Easiest):**
1. Go to https://antimatter15.com/splat/
2. Drag and drop `scene.ply`
3. Interact with the 3D scene

**Desktop Viewers:**
- **MeshLab**: Open PLY file directly
- **CloudCompare**: Open PLY, visualize point cloud
- **Open3D**: Python library for 3D visualization

### Camera Poses

```python
import numpy as np

# Load camera data
data = np.load('data/outputs/depth_estimation/predictions.npz')
extrinsics = data['extrinsics']  # (N, 4, 4)

# Get camera positions
positions = extrinsics[:, :3, 3]
print("Camera positions:", positions)
```

## ⚙️ Performance Tips

### Speed Optimization

1. **Use smaller models** for faster processing:
   - `DA3-Small`: ~2x faster
   - `DA3-BASE`: ~1.5x faster
   - `DA3NESTED-GIANT-LARGE`: Best quality (default)

2. **Reduce resolution** for faster processing:
   - Default (low-res): Fast preview
   - `--high-res`: Better quality, 2-3x slower

3. **GPU acceleration**: Always use CUDA if available
   - Check: `nvidia-smi`
   - 10-20x faster than CPU

### Memory Optimization

If you encounter OOM (Out of Memory) errors:

1. Use smaller model: `--model-dir depth-anything/DA3-Small`
2. Process fewer images at once
3. Don't use `--high-res` flag
4. For Gaussian Splatting: Use `--gs-views-interval 2` or higher

## 🐛 Troubleshooting

### "No images found"
- Check image formats (supported: jpg, png, bmp, tiff)
- Verify images are in the correct directory
- Check file permissions

### "CUDA out of memory"
- Reduce number of images
- Use smaller model: `--model-dir depth-anything/DA3-Small`
- Use CPU mode: `--device cpu`

### "Predictions file not found"
- Make sure `dev.py` completed successfully
- Check the path to predictions.npz
- Verify `--input-dir` points to correct location

### "No Gaussian data in prediction"
- This means `infer_gs=True` was not enabled
- Re-run with Gaussian inference enabled
- This is expected - Gaussian inference requires additional computation

## 📚 Additional Resources

- **Full Documentation**: See [DEV_README.md](DEV_README.md)
- **GitHub**: https://github.com/ByteDance-Seed/Depth-Anything-3
- **Paper**: Depth Anything V3 on arXiv
- **Models**: https://huggingface.co/depth-anything

## 💡 Tips & Tricks

1. **Image Order**: Images are processed in alphabetical order. Rename files if you need a specific order.

2. **First Frame Selection**: In Gradio app, you can select which frame to use as the reference view.

3. **Dense Sequences**: For videos with many frames, use `--gs-views-interval 2` or higher to reduce Gaussian file size.

4. **Quality vs Speed**: For quick previews, use default settings. For final results, use `--high-res`.

5. **Camera Poses**: The estimated camera poses are in camera-to-world format. Extract position with `extrinsics[:, :3, 3]`.

## 🔄 Integration Workflow

### With Gradio App

```bash
# Save to Gradio workspace
python3 dev.py --output-dir workspace/gradio/input_images/my_scene

# Launch Gradio app
python3 app.py

# Load scene in browser - predictions will be cached
```

### With Other Tools

```bash
# Export to standard formats
python3 dev.py

# Use predictions.npz in your own scripts
python3 your_custom_script.py data/outputs/depth_estimation/predictions.npz
```

## 🎓 Example Workflows

### Complete 3D Reconstruction Pipeline

```bash
# 1. Capture/collect images
# 2. Organize in data/inputs/
# 3. Run depth estimation
python3 dev.py --high-res

# 4. Generate Gaussian Splatting
python3 02_estimate_gaussian.py

# 5. Analyze results
python3 example_usage.py

# 6. View in 3DGS viewer
# Open https://antimatter15.com/splat/ and upload scene.ply
```

---

For more detailed information, see [DEV_README.md](DEV_README.md).
