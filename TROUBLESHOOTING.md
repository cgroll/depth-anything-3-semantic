# Troubleshooting Guide

Common issues and solutions for the Depth Anything 3 development scripts.

## 3D Gaussian Splatting Issues

### "No Gaussian data in prediction" Error

**Error message:**
```
❌ Error: No Gaussian data in prediction!
This usually means:
  1. The model doesn't support 3D Gaussian Splatting
     → Use --model-dir depth-anything/DA3NESTED-GIANT-LARGE
  2. Or infer_gs=True was not enabled (should be automatic in this script)

Supported models:
  - depth-anything/DA3-GIANT (smallest with 3DGS, 1.15B params)
  - depth-anything/DA3NESTED-GIANT-LARGE (best, 1.40B params, recommended)
  - depth-anything/DA3MONO-LARGE
```

**Cause:** You're using a model that doesn't support 3D Gaussian Splatting.

**Solution:**
```bash
# Use the correct model
python3 02_estimate_gaussian.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE
```

**Explanation:** Only these model variants support 3D Gaussian Splatting:
- ✅ **DA3-GIANT** - Smallest with 3DGS (1.15B params)
- ✅ **DA3NESTED-GIANT-LARGE** - Best quality, supports metric depth + 3DGS (1.40B params)
- ✅ **DA3MONO-LARGE** - Alternative with 3DGS support
- ❌ **DA3-Small, DA3-BASE, DA3-LARGE** - No 3DGS support

### Model Compatibility Table

| Model | Depth Estimation | 3D Gaussian Splatting | Metric Depth | Params |
|-------|-----------------|---------------------|--------------|--------|
| DA3-Small | ✅ | ❌ | ❌ | 0.08B |
| DA3-BASE | ✅ | ❌ | ❌ | 0.12B |
| DA3-LARGE | ✅ | ❌ | ❌ | 0.35B |
| DA3-GIANT | ✅ | ✅ | ❌ | 1.15B |
| DA3METRIC-LARGE | ✅ | ❌ | ✅ | 0.35B |
| DA3MONO-LARGE | ✅ | ✅ | ✅ | 0.35B |
| DA3NESTED-GIANT-LARGE | ✅ | ✅ | ✅ | 1.40B |

**Recommendations:**
- **For 3DGS + Best Quality**: Use `DA3NESTED-GIANT-LARGE` (1.40B params)
- **For 3DGS + Less Memory**: Use `DA3-GIANT` (1.15B params)

## Extrinsics Format Issues

### "Expected size for first two dimensions of batch2 tensor to be: [N, 4] but got: [N, 3]"

**Cause:** Extrinsics are in (N, 3, 4) format but model expects (N, 4, 4).

**Status:** ✅ **FIXED** - The script now automatically converts extrinsics format.

**What happens:** The script detects (N, 3, 4) format and adds the bottom row [0, 0, 0, 1] to make it (N, 4, 4).

You should see:
```
ℹ️  Converting extrinsics from (N, 3, 4) to (N, 4, 4) format...
✓ Converted extrinsics to shape: (16, 4, 4)
```

If this doesn't work, please report it as a bug.

## GPU Memory Issues

### "CUDA out of memory"

**Quick fixes:**

1. **Use smaller model:**
   ```bash
   # For dev.py (depth estimation only)
   python3 dev.py --model-dir depth-anything/DA3-Small

   # For 02_estimate_gaussian.py (requires GS-capable model)
   python3 02_estimate_gaussian.py --model-dir depth-anything/DA3MONO-LARGE
   ```

2. **Don't use high-res:**
   ```bash
   # Default is low-res (faster, less memory)
   python3 dev.py
   ```

3. **Process fewer images:**
   - Reduce number of input images
   - Split dataset into batches

4. **Use CPU:**
   ```bash
   python3 dev.py --device cpu
   python3 02_estimate_gaussian.py --device cpu
   ```

5. **Clear GPU memory between runs:**
   ```bash
   python3 -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"
   ```

### Memory Requirements

**Depth Estimation (dev.py):**
- DA3-Small: ~4-6GB
- DA3-BASE: ~6-8GB
- DA3-LARGE: ~10-14GB
- DA3NESTED-GIANT-LARGE: ~16-20GB

**Gaussian Splatting (02_estimate_gaussian.py):**
- DA3MONO-LARGE: ~12-16GB
- DA3NESTED-GIANT-LARGE: ~18-24GB

Add 2-4GB if using `--high-res`.

## Input/Output Issues

### "No images found"

**Causes:**
1. Wrong directory path
2. Unsupported image format
3. File permissions

**Solutions:**
```bash
# Check directory exists and has images
ls -la data/inputs/

# Check file formats (supported: jpg, png, bmp, tiff)
file data/inputs/*

# Fix permissions
chmod 644 data/inputs/*
```

### "Predictions file not found"

**Error:** `predictions.npz` not found.

**Solution:**
```bash
# Make sure dev.py completed successfully
python3 dev.py

# Check file was created
ls -la data/outputs/depth_estimation/predictions.npz

# Then run Gaussian Splatting
python3 02_estimate_gaussian.py
```

## Model Loading Issues

### Model download is slow

**Solution:** Models are cached after first download to:
```bash
~/.cache/huggingface/hub/
```

First run will download ~2-5GB, subsequent runs use the cache.

### Model fails to load

**Causes:**
1. No internet connection (first time)
2. Corrupted cache
3. Wrong model name

**Solutions:**
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/hub/models--depth-anything*

# Verify model name is correct
python3 02_estimate_gaussian.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE
```

## Pipeline Issues

### Running dev.py and 02_estimate_gaussian.py together

**Problem:** Both scripts try to use GPU simultaneously.

**Solution:** Run sequentially, not in parallel:
```bash
# Good: Sequential
python3 dev.py && python3 02_estimate_gaussian.py

# Bad: Parallel
python3 dev.py & python3 02_estimate_gaussian.py  # Don't do this!
```

### Results don't match between runs

**Possible causes:**
1. Different image order (files sorted alphabetically)
2. Different models used
3. Different resolution settings

**Solution:** Use consistent settings:
```bash
# Always use same model and settings
python3 dev.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE --high-res
python3 02_estimate_gaussian.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE
```

## Viewing Issues

### PLY file won't open in viewer

**Causes:**
1. Incomplete generation (script failed)
2. Viewer doesn't support Gaussian Splatting format
3. File too large

**Solutions:**

1. **Check file was created successfully:**
   ```bash
   ls -lh data/outputs/gaussian_splatting/scene.ply
   ```

2. **Use correct viewer:**
   - Web: https://antimatter15.com/splat/
   - Desktop: MeshLab, CloudCompare

3. **Reduce file size:**
   ```bash
   python3 02_estimate_gaussian.py --gs-views-interval 2
   ```

### Web viewer crashes

**Cause:** PLY file too large for browser.

**Solutions:**
1. Use `--gs-views-interval 2` or higher
2. Process fewer images
3. Use desktop viewer instead

## Performance Issues

### Inference is very slow

**Expected speeds (GPU):**
- Depth estimation: 2-5 seconds per image
- Gaussian Splatting: 30-60 seconds total

**If slower:**

1. **Check GPU is being used:**
   ```bash
   nvidia-smi
   # Should show python process using GPU
   ```

2. **Use GPU instead of CPU:**
   ```bash
   # Don't use --device cpu unless necessary
   python3 dev.py
   ```

3. **Close other GPU applications:**
   ```bash
   nvidia-smi
   # Kill other GPU processes if needed
   ```

## Development Issues

### Python import errors

**Error:** `ModuleNotFoundError: No module named 'depth_anything_3'`

**Solution:**
```bash
# Make sure you're in the correct directory
cd /path/to/depth-anything-3

# Install in development mode
pip install -e .
```

### Script changes not taking effect

**Cause:** Python cached bytecode.

**Solution:**
```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Or restart Python
```

## Getting Help

If none of these solutions work:

1. **Check the logs:** Look for error messages in the terminal output
2. **Verify setup:** Run `python3 test_gaussian.py` to verify pipeline
3. **Report issue:** Create an issue at https://github.com/ByteDance-Seed/Depth-Anything-3/issues

Include:
- Error message (full traceback)
- Command you ran
- Python version: `python3 --version`
- PyTorch version: `python3 -c "import torch; print(torch.__version__)"`
- CUDA version: `nvidia-smi`
- GPU memory: Output from script showing memory stats
