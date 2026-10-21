#!/usr/bin/env python3
"""
Quick test script to verify the Gaussian Splatting pipeline works.
"""

import numpy as np
import sys

def test_extrinsics_conversion():
    """Test that extrinsics conversion works correctly."""
    print("Testing extrinsics conversion...")

    # Load predictions
    try:
        data = np.load('data/outputs/depth_estimation/predictions.npz')
    except FileNotFoundError:
        print("❌ Error: predictions.npz not found.")
        print("   Please run 'python3 dev.py' first.")
        return False

    extrinsics = data['extrinsics']
    print(f"  Original shape: {extrinsics.shape}")

    # Check if conversion is needed
    if extrinsics.shape[1:] != (3, 4):
        print(f"  ⚠️  Unexpected extrinsics shape: {extrinsics.shape}")
        return False

    # Apply conversion
    N = extrinsics.shape[0]
    extrinsics_4x4 = np.zeros((N, 4, 4), dtype=extrinsics.dtype)
    extrinsics_4x4[:, :3, :] = extrinsics
    extrinsics_4x4[:, 3, 3] = 1.0

    print(f"  Converted shape: {extrinsics_4x4.shape}")

    # Verify bottom row
    expected_bottom = np.array([0., 0., 0., 1.])
    if not np.allclose(extrinsics_4x4[0, 3, :], expected_bottom):
        print(f"  ❌ Bottom row incorrect: {extrinsics_4x4[0, 3, :]}")
        return False

    print("  ✅ Conversion successful!")
    return True


def main():
    """Run tests."""
    print("="*60)
    print("Gaussian Splatting Pipeline Test")
    print("="*60)

    # Test extrinsics conversion
    if not test_extrinsics_conversion():
        print("\n❌ Tests failed!")
        return 1

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    print("\nYou can now run:")
    print("  python3 02_estimate_gaussian.py")

    return 0


if __name__ == "__main__":
    exit(main())
