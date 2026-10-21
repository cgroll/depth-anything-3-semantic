# GPU Memory Recommendations

This guide helps you choose the right model based on your GPU memory.

## Quick Reference

| Your GPU Memory | Depth Estimation (dev.py) | 3D Gaussian Splatting (02_estimate_gaussian.py) |
|-----------------|---------------------------|------------------------------------------------|
| **< 8GB** | ✅ DA3-Small (default) | ❌ Not supported on GPU (use CPU mode) |
| **8-12GB** | ✅ DA3-Small/BASE/LARGE | ❌ Not supported on GPU (use CPU mode) |
| **12-16GB** | ✅ Any model | ✅ DA3-GIANT (only option) |
| **16-24GB** | ✅ Any model | ✅ DA3-GIANT or DA3NESTED-GIANT-LARGE |
| **24GB+** | ✅ Any model | ✅ Any 3DGS model (DA3NESTED recommended) |

## Important: 3D Gaussian Splatting Model Requirements

⚠️ **Only 2 models support 3D Gaussian Splatting:**
- **DA3-GIANT** (1.15B params, requires 12-16GB GPU)
- **DA3NESTED-GIANT-LARGE** (1.40B params, requires 18-24GB GPU)

**Models that DO NOT support 3DGS:**
- ❌ DA3-SMALL (0.08B)
- ❌ DA3-BASE (0.12B)
- ❌ DA3-LARGE (0.35B)
- ❌ DA3MONO-LARGE (0.35B) - despite the name suggesting otherwise
- ❌ DA3METRIC-LARGE (0.35B)

These smaller models lack the required `gs_head` and `cam_enc` modules for 3D Gaussian Splatting.

## Automated GPU Check

Run this to see what your GPU supports:

```bash
python3 check_gpu.py
```

This will analyze your GPU and recommend the best model.

## Model Details

### Depth Estimation Models (dev.py)

| Model | GPU Memory | Speed | Quality | Command |
|-------|------------|-------|---------|----|
| **DA3-Small** | ~4-6GB | Fastest | Good | `python3 dev.py` (default) |
| DA3-BASE | ~6-8GB | Fast | Better | `python3 dev.py --model-dir depth-anything/DA3-BASE` |
| DA3-LARGE | ~10-14GB | Medium | Best | `python3 dev.py --model-dir depth-anything/DA3-LARGE` |

### 3D Gaussian Splatting Models (02_estimate_gaussian.py)

| Model | GPU Memory | Parameters | Quality | Command |
|-------|------------|------------|---------|----|
| **DA3-GIANT** | ~12-16GB | 1.15B | Good | `python3 02_estimate_gaussian.py` (default) |
| DA3NESTED-GIANT-LARGE | ~18-24GB | 1.40B | Best | `python3 02_estimate_gaussian.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE` |

**Note**: DA3-GIANT is the smallest and only option for GPUs with 12-16GB memory.

## What Happens with Insufficient Memory

If you try to use a model that's too large for your GPU:

1. **During download**: Process may be killed by OOM (Out of Memory) killer
2. **During model loading**: `CUDA out of memory` error
3. **During inference**: Process crashes or system becomes unresponsive

**Solution**: Use the GPU memory check built into `02_estimate_gaussian.py` or run `check_gpu.py` first.

## GPU Memory Check Feature

The `02_estimate_gaussian.py` script now includes automatic GPU memory checking:

```bash
python3 02_estimate_gaussian.py
```

**Output example (8GB GPU):**
```
============================================================
3D Gaussian Splatting Estimation
============================================================

🧹 Initial GPU cleanup...
  ✓ GPU cache cleared
  📊 GPU: NVIDIA GeForce RTX 3070
  💾 Total GPU Memory: 8.0 GB

⚠️  WARNING: GPU has only 8.0GB memory!
   DA3-GIANT requires ~12-16GB GPU memory.

💡 Recommended alternatives for your GPU:
   - Use CPU mode: --device cpu
   - Or skip 3DGS: Use dev.py for depth estimation only

   Continue anyway? (y/N):
```

You can:
- Type `N` (or Enter) to exit and use CPU mode or skip 3DGS
- Type `y` to continue anyway (will likely fail with OOM)

## Recommendations by GPU Model

### Common Consumer GPUs

**NVIDIA RTX 3060 (12GB):**
```bash
# Depth estimation
python3 dev.py --model-dir depth-anything/DA3-LARGE

# Gaussian Splatting
python3 02_estimate_gaussian.py  # Uses DA3-GIANT
```

**NVIDIA RTX 3070/3080 (8-10GB):**
```bash
# Depth estimation
python3 dev.py  # Uses DA3-Small (default)

# Gaussian Splatting - NOT SUPPORTED ON GPU
# Use CPU mode (very slow):
python3 02_estimate_gaussian.py --device cpu
```

**NVIDIA RTX 3090/4090 (24GB):**
```bash
# Depth estimation
python3 dev.py --model-dir depth-anything/DA3-LARGE

# Gaussian Splatting (use best quality)
python3 02_estimate_gaussian.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE
```

**NVIDIA RTX 4060/4070 (8-12GB):**
```bash
# Depth estimation
python3 dev.py  # Uses DA3-Small

# Gaussian Splatting
# 8GB: Use CPU mode
python3 02_estimate_gaussian.py --device cpu

# 12GB: Use DA3-GIANT
python3 02_estimate_gaussian.py  # Uses DA3-GIANT
```

### Professional GPUs

**NVIDIA A100/H100 (40-80GB):**
```bash
# Use best models
python3 dev.py --model-dir depth-anything/DA3-LARGE --high-res
python3 02_estimate_gaussian.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE
```

**NVIDIA V100 (16-32GB):**
```bash
# Depth estimation
python3 dev.py --model-dir depth-anything/DA3-LARGE

# Gaussian Splatting
python3 02_estimate_gaussian.py --model-dir depth-anything/DA3-GIANT
# OR for 32GB:
python3 02_estimate_gaussian.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE
```

## CPU Mode (For GPUs with <12GB)

If your GPU has insufficient memory for 3DGS or you don't have a GPU:

```bash
# Depth estimation on CPU (slow but works)
python3 dev.py --device cpu

# Gaussian Splatting on CPU (VERY slow, 30-60 minutes)
python3 02_estimate_gaussian.py --device cpu
```

**CPU Performance:**
- Depth estimation: ~30-60 seconds per image
- Gaussian Splatting: ~30-60 minutes total

Compare to GPU:
- Depth estimation: ~2-5 seconds per image
- Gaussian Splatting: ~2-5 minutes total

## Memory Optimization Tips

### 1. Reduce Image Count
Process fewer images at once:
```bash
# Instead of 50 images, use 20
ls data/inputs/*.jpg | head -20 | xargs -I {} cp {} data/inputs_subset/
python3 dev.py --input-dir data/inputs_subset
```

### 2. Don't Use High-Res
Avoid `--high-res` flag if memory-constrained:
```bash
# Good for 8GB GPU
python3 dev.py

# May cause OOM on 8GB GPU
python3 dev.py --high-res
```

### 3. Use Smaller GS View Interval
For Gaussian Splatting, export fewer views:
```bash
python3 02_estimate_gaussian.py --gs-views-interval 2
```

This uses every 2nd view instead of all views, reducing memory usage.

### 4. Close Other GPU Applications
```bash
# Check what's using GPU
nvidia-smi

# Kill unnecessary GPU processes
kill <PID>
```

## Troubleshooting

### "Killed" during model download
**Cause**: System ran out of memory during download

**Solution**:
```bash
# Clear incomplete download
rm -rf ~/.cache/huggingface/hub/models--depth-anything--DA3-GIANT/

# Use CPU mode instead
python3 02_estimate_gaussian.py --device cpu
```

### "CUDA out of memory" error
**Solution**: Use CPU mode
```bash
python3 02_estimate_gaussian.py --device cpu
```

### Process becomes unresponsive
**Solution**:
1. Press Ctrl+C to kill process
2. Clear GPU memory: `python3 -c "import torch; torch.cuda.empty_cache()"`
3. Use CPU mode

### "'NoneType' object is not callable" at cam_enc
**Cause**: Using a model that doesn't support 3DGS (e.g., DA3MONO-LARGE)

**Solution**: Use DA3-GIANT or DA3NESTED-GIANT-LARGE:
```bash
python3 02_estimate_gaussian.py --model-dir depth-anything/DA3-GIANT
```

## Summary

**Default models are optimized for typical hardware:**
- `dev.py`: DA3-Small (works on 4GB+ GPU)
- `02_estimate_gaussian.py`: DA3-GIANT (requires 12GB+ GPU)

**For 8GB GPUs:**
- ✅ Depth estimation: Use DA3-Small
- ❌ 3D Gaussian Splatting: Not supported on GPU - use `--device cpu`

**Automatic safety features:**
- GPU memory check warns before attempting large models
- Interactive prompt allows you to abort if insufficient memory detected
- Clear error messages when using unsupported models

**Run `python3 check_gpu.py` for personalized recommendations!**
