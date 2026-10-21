# Development Script for Multi-View Depth Estimation

This guide explains how to use the `dev.py` script for processing multi-view images with Depth Anything 3.

## Quick Start

1. **Place your images in the input directory:**
   ```bash
   # Copy your images to data/inputs/
   cp /path/to/your/images/* data/inputs/
   ```

2. **Run the script:**
   ```bash
   python dev.py
   ```

3. **Find results in the output directory:**
   ```
   data/outputs/depth_estimation/
   ├── depth_raw/          # Raw depth maps (.npy files)
   ├── depth_vis/          # Depth visualizations (.jpg files)
   └── predictions.npz     # All prediction data (depths, cameras, etc.)
   ```

## Usage Examples

### Basic Usage
Process images from `data/inputs/` and save to `data/outputs/depth_estimation/`:
```bash
python dev.py
```

### Custom Directories
Use custom input and output directories:
```bash
python dev.py --input-dir path/to/images --output-dir path/to/output
```

### High-Resolution Processing
Use high-resolution processing for better quality (slower):
```bash
python dev.py --high-res
```

### Different Model
Use a different model from Hugging Face:
```bash
# Use the LARGE model (faster, smaller)
python dev.py --model-dir depth-anything/DA3-LARGE

# Use the BASE model (even faster)
python dev.py --model-dir depth-anything/DA3-BASE
```

### CPU Mode
Force CPU processing (useful if GPU not available):
```bash
python dev.py --device cpu
```

## Output Format

The script saves three types of outputs:

### 1. Raw Depth Maps (`depth_raw/`)
- Format: `.npy` files (NumPy arrays)
- Shape: `(H, W)` - height × width
- Values: Metric depth values in meters (if using metric model)
- Load with: `depth = np.load('depth_raw/0000.npy')`

### 2. Depth Visualizations (`depth_vis/`)
- Format: `.jpg` images
- Content: Side-by-side comparison of input image and colored depth map
- Colors: Closer objects = warmer colors (red), farther objects = cooler colors (blue)

### 3. Predictions Data (`predictions.npz`)
- Format: Compressed NumPy archive
- Contains:
  - `depths`: All depth maps (N, H, W)
  - `extrinsics`: Camera poses (N, 4, 4)
  - `intrinsics`: Camera intrinsics (N, 3, 3)
  - `images`: Processed input images (N, H, W, 3)
  - `conf`: Confidence maps if available (N, H, W)
  - `is_metric`: Whether depth is in metric scale (bool)
  - `scale_factor`: Metric scale factor if available

Load with:
```python
import numpy as np
data = np.load('predictions.npz')
depths = data['depths']
extrinsics = data['extrinsics']
intrinsics = data['intrinsics']
```

## Advanced Usage

### Processing Multiple Sequences

Process multiple image sequences in a loop:

```bash
#!/bin/bash
for scene in scene1 scene2 scene3; do
    python dev.py \
        --input-dir data/inputs/$scene \
        --output-dir data/outputs/$scene
done
```

### Batch Processing Script

Create a Python script for batch processing:

```python
import subprocess
import os

scenes = ['scene1', 'scene2', 'scene3']

for scene in scenes:
    input_dir = f'data/inputs/{scene}'
    output_dir = f'data/outputs/{scene}'

    if os.path.exists(input_dir):
        print(f"Processing {scene}...")
        subprocess.run([
            'python', 'dev.py',
            '--input-dir', input_dir,
            '--output-dir', output_dir
        ])
```

## Understanding the Output

### Camera Extrinsics
Camera extrinsics define the camera's position and orientation in world space:
- Shape: `(N, 4, 4)` - 4×4 transformation matrix for each of N views
- Format: Camera-to-world transformation matrix
- Top-left 3×3: Rotation matrix
- Top-right 3×1: Translation vector (camera position)

### Camera Intrinsics
Camera intrinsics define the camera's internal parameters:
- Shape: `(N, 3, 3)` - 3×3 matrix for each of N views
- Format: Camera calibration matrix
- `K[0,0]`: Focal length in x (fx)
- `K[1,1]`: Focal length in y (fy)
- `K[0,2]`: Principal point x (cx)
- `K[1,2]`: Principal point y (cy)

### Using the Results

Load and visualize depth maps:
```python
import numpy as np
import matplotlib.pyplot as plt

# Load predictions
data = np.load('data/outputs/depth_estimation/predictions.npz')
depths = data['depths']

# Visualize first depth map
plt.figure(figsize=(10, 8))
plt.imshow(depths[0], cmap='turbo')
plt.colorbar(label='Depth (meters)')
plt.title('Depth Map - View 0')
plt.show()
```

Load and use camera poses:
```python
import numpy as np

# Load camera data
data = np.load('data/outputs/depth_estimation/predictions.npz')
extrinsics = data['extrinsics']  # (N, 4, 4)
intrinsics = data['intrinsics']  # (N, 3, 3)

# Get camera position for first view
camera_pos = extrinsics[0, :3, 3]
print(f"Camera position: {camera_pos}")

# Get focal length for first view
fx = intrinsics[0, 0, 0]
fy = intrinsics[0, 1, 1]
print(f"Focal length: fx={fx:.2f}, fy={fy:.2f}")
```

## Performance Tips

1. **GPU Acceleration**: CUDA GPU highly recommended for faster processing
   - Typical speed: ~2-5 seconds per image on GPU
   - CPU processing: ~30-60 seconds per image

2. **Resolution Settings**:
   - Default (low-res): Faster, good for previews
   - `--high-res`: Better quality, slower processing

3. **Model Selection**:
   - `DA3NESTED-GIANT-LARGE`: Best quality, slowest (default)
   - `DA3-LARGE`: Good balance of speed and quality
   - `DA3-BASE`: Fastest, lower quality

4. **Batch Size**: The script processes all images at once for better multi-view consistency

## Troubleshooting

### Out of Memory Error
If you get OOM errors with many/large images:
1. Try using `--high-res` without the flag (default low-res)
2. Use a smaller model: `--model-dir depth-anything/DA3-BASE`
3. Process fewer images at once

### No Images Found
Ensure your images are in supported formats:
- Supported: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`
- Case-insensitive matching

### CUDA Errors
If you get CUDA errors:
1. Check CUDA is properly installed: `nvidia-smi`
2. Try CPU mode: `--device cpu`
3. Update PyTorch to match your CUDA version

## 3D Gaussian Splatting (02_estimate_gaussian.py)

After running `dev.py`, you can generate 3D Gaussian Splatting representations for novel view synthesis.

**⚠️ Important**: Only **DA3-GIANT**, **DA3NESTED**, and **DA3MONO** models support 3D Gaussian Splatting!

### Basic Usage

```bash
# Generate Gaussian Splatting from depth predictions
python3 02_estimate_gaussian.py
```

This will:
1. Load predictions from `data/outputs/depth_estimation/predictions.npz`
2. Run 3DGS inference using the depth and camera data (with DA3MONO-LARGE model, optimized for 8-12GB GPUs)
3. Save the result as `data/outputs/gaussian_splatting/scene.ply`

**GPU Requirements:**
- **8-12GB GPU**: Use DA3MONO-LARGE (default)
- **16GB+ GPU**: Can use DA3NESTED-GIANT-LARGE for best quality
- **<8GB GPU**: Use CPU mode or skip 3DGS

### Advanced Options

```bash
# Custom input/output paths
python3 02_estimate_gaussian.py \
    --input-dir path/to/depth_estimation \
    --output path/to/output.ply

# Specify original images directory
python3 02_estimate_gaussian.py --images-dir data/inputs

# Use best quality model (requires 16GB+ GPU)
python3 02_estimate_gaussian.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE

# Use CPU mode (for GPUs with <8GB memory)
python3 02_estimate_gaussian.py --device cpu

# For dense sequences, export every 2nd view to reduce file size
python3 02_estimate_gaussian.py --gs-views-interval 2
```

### Viewing the Results

The generated `.ply` file can be viewed in:

1. **Web-based 3DGS viewers:**
   - https://antimatter15.com/splat/
   - https://playcanvas.com/supersplat/editor

2. **Desktop viewers:**
   - MeshLab
   - CloudCompare
   - Open3D

3. **3DGS-specific tools:**
   - Gaussian Splatting SIBR Viewer
   - Nerfstudio viewer

### Pipeline Example

Complete pipeline from images to 3D Gaussian Splatting:

```bash
# Step 1: Place your images
cp /path/to/your/images/* data/inputs/

# Step 2: Run depth estimation
python3 dev.py

# Step 3: Generate Gaussian Splatting
python3 02_estimate_gaussian.py

# Results will be in:
# - data/outputs/depth_estimation/ (depth maps, cameras)
# - data/outputs/gaussian_splatting/scene.ply (3DGS file)
```

## Integration with Gradio App

The depth estimation results can be loaded into the Gradio app for 3D visualization:

1. Save your results to the workspace directory:
   ```bash
   python3 dev.py --output-dir workspace/gradio/input_images/my_scene
   ```

2. The Gradio app will detect the cached `predictions.npz` and load it automatically

## References

- GitHub: https://github.com/ByteDance-Seed/Depth-Anything-3
- Paper: Depth Anything V3 (arXiv)
- Hugging Face: https://huggingface.co/depth-anything
