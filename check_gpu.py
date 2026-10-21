#!/usr/bin/env python3
"""
Check GPU capabilities and recommend appropriate model.
"""

import subprocess

def get_gpu_memory():
    """Get GPU memory in GB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        memory_mb = float(result.stdout.strip())
        return memory_mb / 1024  # Convert to GB
    except Exception as e:
        print(f"Could not detect GPU: {e}")
        return 0

def recommend_model(gpu_memory_gb):
    """Recommend model based on GPU memory."""
    print("="*60)
    print("GPU Memory Analysis")
    print("="*60)
    print(f"\n🔍 Detected GPU Memory: {gpu_memory_gb:.1f} GB")

    recommendations = []

    if gpu_memory_gb >= 24:
        recommendations.append({
            'name': 'DA3NESTED-GIANT-LARGE',
            'params': '1.40B',
            'memory': '~18-24GB',
            'capabilities': '3DGS + Metric Depth (Best Quality)',
            'command': 'python3 02_estimate_gaussian.py'
        })

    if gpu_memory_gb >= 16:
        recommendations.append({
            'name': 'DA3-GIANT',
            'params': '1.15B',
            'memory': '~12-16GB',
            'capabilities': '3DGS (Good Quality)',
            'command': 'python3 02_estimate_gaussian.py --model-dir depth-anything/DA3-GIANT'
        })

    if gpu_memory_gb >= 12:
        recommendations.append({
            'name': 'DA3-GIANT',
            'params': '1.15B',
            'memory': '~12-16GB',
            'capabilities': '3DGS (Smallest model with 3DGS support)',
            'command': 'python3 02_estimate_gaussian.py --model-dir depth-anything/DA3-GIANT'
        })

    if gpu_memory_gb < 12:
        print("\n⚠️  WARNING: Your GPU does NOT have enough memory for 3D Gaussian Splatting!")
        print(f"   Available: {gpu_memory_gb:.1f}GB")
        print("   Required: At least 12GB for DA3-GIANT (smallest 3DGS model)")
        print("\n💡 Recommendations:")
        print("   1. Use CPU mode (very slow, 30-60 min):")
        print("      python3 02_estimate_gaussian.py --device cpu --model-dir depth-anything/DA3-GIANT")
        print("\n   2. Use depth estimation only (without 3DGS):")
        print("      python3 dev.py --model-dir depth-anything/DA3-Small")
        print("\n   3. Upgrade to a GPU with at least 12GB memory")
        print("\n   Note: DA3MONO-LARGE and other smaller models do NOT support 3DGS.")
        return

    print("\n✅ Recommended Models for Your GPU:\n")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} ({rec['params']} parameters)")
        print(f"   Memory Required: {rec['memory']}")
        print(f"   Capabilities: {rec['capabilities']}")
        print(f"   Command: {rec['command']}")
        print()

    if recommendations:
        print("🎯 Best Choice:")
        best = recommendations[0]
        print(f"   {best['name']}")
        print(f"   {best['command']}")

def main():
    gpu_memory = get_gpu_memory()

    if gpu_memory == 0:
        print("❌ No GPU detected or nvidia-smi not available")
        print("\n💡 Use CPU mode:")
        print("   python3 02_estimate_gaussian.py --device cpu --model-dir depth-anything/DA3MONO-LARGE")
        return

    recommend_model(gpu_memory)

    print("\n" + "="*60)
    print("For more information, see:")
    print("  - TROUBLESHOOTING.md (Model Compatibility Table)")
    print("  - GPU_MEMORY_MANAGEMENT.md")
    print("="*60)

if __name__ == "__main__":
    main()
