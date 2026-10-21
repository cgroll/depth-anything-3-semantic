# GPU Memory Management

This document describes the GPU memory management strategies implemented in the development scripts.

## Overview

Both `dev.py` and `02_estimate_gaussian.py` implement aggressive GPU memory management to ensure:
- Clean startup state
- Efficient memory usage during inference
- Proper cleanup after completion
- Reduced risk of OOM (Out of Memory) errors

## Implementation Details

### 1. Startup Cleanup

Both scripts clear GPU memory at startup before loading models:

```python
# Clear GPU memory at startup
if device == "cuda" and torch.cuda.is_available():
    print("\n🧹 Clearing GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    print(f"  GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

**Benefits:**
- Frees memory from previous runs
- Shows current memory state for debugging
- Ensures clean starting point

### 2. Post-Inference Cleanup

After inference completes, memory is cleaned up:

```python
# Cleanup
if device == "cuda":
    torch.cuda.empty_cache()
    gc.collect()
```

**Benefits:**
- Releases model tensors from GPU
- Frees cached memory
- Prepares for next operation

### 3. Additional Cleanup in 02_estimate_gaussian.py

The Gaussian Splatting script includes additional cleanup:

**Initial cleanup in main():**
```python
# Initial GPU cleanup
if torch.cuda.is_available() and args.device != "cpu":
    print("\n🧹 Initial GPU cleanup...")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    print("  ✓ GPU cache cleared")
```

**Post-inference cleanup in run_gaussian_inference():**
```python
# Clean up GPU memory
if device == "cuda" and torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
```

**Benefits:**
- Double cleanup ensures maximum memory availability
- Important for 3DGS which is memory-intensive

## Memory Monitoring

The scripts display GPU memory statistics at startup:

```
🧹 Clearing GPU memory...
  GPU memory: 0.23GB allocated, 0.50GB reserved
```

**Interpreting the output:**
- **Allocated**: Memory currently in use by tensors
- **Reserved**: Memory reserved by PyTorch allocator (may be higher than allocated)

## Best Practices

### 1. Run Scripts Sequentially

If running both scripts, run them sequentially rather than concurrently:

```bash
# Good: Sequential execution
python3 dev.py
python3 02_estimate_gaussian.py

# Avoid: Concurrent execution
python3 dev.py & python3 02_estimate_gaussian.py  # Don't do this!
```

### 2. Monitor Memory Usage

Check GPU memory before and during runs:

```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Or check once
nvidia-smi
```

### 3. Reduce Memory Usage if Needed

If you encounter OOM errors:

1. **Use smaller models:**
   ```bash
   python3 dev.py --model-dir depth-anything/DA3-Small
   python3 02_estimate_gaussian.py --model-dir depth-anything/DA3-BASE
   ```

2. **Process fewer images:**
   - Reduce number of input images
   - Use `--gs-views-interval 2` to export fewer Gaussians

3. **Don't use high-res:**
   ```bash
   # Default is low-res (faster, less memory)
   python3 dev.py

   # Avoid this if memory-constrained
   python3 dev.py --high-res
   ```

4. **Use CPU mode:**
   ```bash
   python3 dev.py --device cpu
   python3 02_estimate_gaussian.py --device cpu
   ```

### 4. Clear Memory Between Runs

If running multiple experiments, clear memory between runs:

```bash
# Run first experiment
python3 dev.py --output-dir outputs/exp1

# Clear Python cache and GPU
python3 -c "import torch; torch.cuda.empty_cache()"

# Run second experiment
python3 dev.py --output-dir outputs/exp2
```

## Memory Requirements

Typical GPU memory usage:

### dev.py (Depth Estimation)
- **DA3-Small**: ~4-6GB
- **DA3-BASE**: ~6-8GB
- **DA3-LARGE**: ~10-14GB
- **DA3NESTED-GIANT-LARGE**: ~16-20GB

Additional factors:
- High-res processing: +2-4GB
- Number of images: +100-200MB per image

### 02_estimate_gaussian.py (3DGS)
- **DA3-BASE**: ~8-12GB
- **DA3-LARGE**: ~12-16GB
- **DA3NESTED-GIANT-LARGE**: ~18-24GB

Additional factors:
- Number of views: +200-500MB per view
- Gaussian density: varies significantly

## Troubleshooting

### "CUDA out of memory" Error

**Solution 1: Use smaller model**
```bash
python3 dev.py --model-dir depth-anything/DA3-Small
```

**Solution 2: Clear GPU memory**
```bash
# Kill any other GPU processes
nvidia-smi
# Look for PID of GPU processes
kill <PID>

# Then run your script
python3 dev.py
```

**Solution 3: Reduce batch size**
- Process fewer images at once
- Split your dataset into smaller chunks

**Solution 4: Use CPU**
```bash
python3 dev.py --device cpu
```

### Memory Not Being Freed

If you notice memory not being freed after script completion:

```bash
# Manually clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Or restart Python kernel if in Jupyter/IPython
```

### Memory Leak Detection

To check for memory leaks:

```python
import torch
import gc

# Before operation
torch.cuda.reset_peak_memory_stats()
initial = torch.cuda.memory_allocated()

# Run your code here
# ...

# After operation
final = torch.cuda.memory_allocated()
peak = torch.cuda.max_memory_allocated()

print(f"Initial: {initial/1024**3:.2f}GB")
print(f"Final: {final/1024**3:.2f}GB")
print(f"Peak: {peak/1024**3:.2f}GB")
print(f"Leaked: {(final - initial)/1024**3:.2f}GB")
```

## Advanced Configuration

### Environment Variables

Control PyTorch memory behavior:

```bash
# Limit PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Enable memory debugging
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Then run scripts
python3 dev.py
```

### Automatic Mixed Precision (AMP)

The scripts use automatic mixed precision by default for better memory efficiency:

```python
# In model forward pass
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
with torch.autocast(device_type=device.type, dtype=autocast_dtype):
    result = model(...)
```

**Benefits:**
- ~30-50% less memory usage
- ~20-40% faster inference
- Minimal impact on accuracy

## Summary

The scripts implement comprehensive GPU memory management:

✅ **Startup cleanup** - Clear memory before loading models
✅ **Post-inference cleanup** - Free memory after processing
✅ **Memory monitoring** - Display current usage
✅ **Garbage collection** - Python GC + CUDA cache clearing
✅ **Synchronization** - Ensure cleanup completes

This ensures efficient, reliable operation even on memory-constrained systems.
